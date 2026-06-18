import gc
from shutil import ignore_patterns
import argparse
import json
import sys
import os
import os.path as osp
import datetime
import shutil
from typing import List

# from PIL import Image
import toml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
# import torch.distributed as dist 
from accelerate.utils import set_seed

from diffusers import DDPMScheduler, DDIMScheduler
from accelerate import DistributedType
from diffusers.utils import logging
# from diffusers.models import AutoencoderKL

import library.train_util as train_util
import library.chinese_sdxl_train_util as chinese_sdxl_train_util
# import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
)

from model_lib.nets.layers.ema import LitEma, load_litema, save_litema, ema_scope
from removal.v1_2 import (
    RemovalDataset, RemovalDataset_v1_2,
    load_cfg,
    build_removal_model,
    load_removal_model,
)

from utils_train import (
    build_accelerator,
    build_dataloader,
    build_vae,
    build_models,
    save,
    common_arguments,
    build_progress_bar
)
from model_lib.nets.utils import CustomOutput
from utils_infer import encode_clean_latents #, predict_noise

import warnings
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides.*")
warnings.filterwarnings("ignore", message="Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale.")

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def build_teacher_model(args, weight_dtype, accelerator):
    teacher_cfg = load_cfg(args.teacher_config_path)

    teacher_model = build_removal_model(teacher_cfg, args.num_embeddings)

    if args.teacher_weight_path:
        accelerator.print(f"==> Loading teacher model from: {args.teacher_weight_path}")
        state_dict = torch.load(args.teacher_weight_path, map_location=accelerator.device)
        teacher_model.load_state_dict(state_dict)

    accelerator.print(f"weight_dtype:{weight_dtype}")
    if getattr(teacher_model, 'unet', None):
        accelerator.print(f"unet:{teacher_model.unet.dtype}")
    else:
        accelerator.print(f"diff_model:{teacher_model.diff_model.dtype}")

    teacher_model.requires_grad_(False).eval()
    teacher_model.to(accelerator.device, dtype=torch.float32)
    if accelerator.is_main_process:
        from pprint import pprint
        pprint("Teacher Model Config:")
        pprint(teacher_model.diff_model.config)

    # set xformer/mem_eff_attn
    accelerator.print(f"Enable memory efficient attention, mem_eff_attn:{args.mem_eff_attn}, xformers:{args.xformers}")
    chinese_sdxl_train_util.set_diffusers_xformers_flag(teacher_model.diff_model, True)

    return teacher_model



def cal_KD_loss(pred: CustomOutput, target: CustomOutput, args):
    loss_dict = dict()

    # get feat KD loss from intermediate layers.
    if args.kl_feat_loss or args.mse_feat_loss: 
        feat_loss_list = []
        assert len(args.feat_index_S) == len(args.feat_loss_weight)
        assert len(args.feat_index_S) == len(args.feat_index_T)
        for _is, _it, _weight in zip(args.feat_index_S, args.feat_index_T, args.feat_loss_weight):
            feat_S, feat_T = pred.block_outputs[_is], target.block_outputs[_it]
            if args.kl_feat_loss: 
                with torch.no_grad():
                    probs_T = torch.softmax(feat_T /  args.kl_temp, dim=1)
                log_probs_S = torch.log_softmax(feat_S /  args.kl_temp, dim=1)
                feat_loss = torch.nn.functional.kl_div(log_probs_S, probs_T, reduction='batchmean')
            elif args.mse_feat_loss: 
                feat_loss = torch.nn.functional.mse_loss(feat_S, feat_T, reduction='mean')
            else:
                print("no available KD_loss type!")
            feat_loss_list.append(feat_loss * _weight)
        loss_dict["loss_featkd"] = sum(feat_loss_list)
    else:
        loss_dict["loss_featkd"] = 0

    loss_outkd = torch.nn.functional.mse_loss(pred.sample.float(), target.sample.float(), reduction="mean")
    loss_dict["loss_outkd"] = loss_outkd
    loss_kd = sum([ v for k,v in loss_dict.items()])
    return loss_kd, loss_dict 


def cal_task_loss(pred: CustomOutput, target: torch.Tensor, args):
    '''
        refer to task loss between gt noise and student pred noise in SnapGen.
    '''
    loss_dict = dict()
    if args.task_loss:
        loss_task = torch.nn.functional.mse_loss(pred.sample.float(), target.float(),reduction='mean')
        loss_dict['loss_task'] = loss_task
    else:
        loss_task = 0
        loss_dict['loss_task'] = loss_task
    return loss_task, loss_dict

def cal_elatentlpips_loss(
    pred: CustomOutput, target: torch.Tensor, encoder_model:torch.nn.Module,  
    noise_scheduler, timesteps, noisy_latents, args = None):
    '''
        refer to task loss between gt noise and student pred noise in SnapGen.
    '''

    loss_dict = dict()
    if args.elatentlpips_loss:
        # Compute the perceptual distance between the two latent representations
        # Note: Set `normalize=True` if the latents (latent0 and latent1) are not already normalized 
        # by `vae.config.scaling_factor` and `vae.config.shift_factor`.
        noise_pred = pred.sample
        noisy_latents_pred = torch.stack([
            noise_scheduler.step(n, t, noisy_latent).pred_original_sample \
                for (n, t, noisy_latent) in zip(noise_pred, timesteps, noisy_latents)
        ])
        target_latents_pred = torch.stack([
            noise_scheduler.step(tgt, t, noisy_latent).pred_original_sample \
                for (tgt, t, noisy_latent) in zip(target.float(), timesteps, noisy_latents)
        ])
        loss_elatentlpips = encoder_model(noisy_latents_pred, target_latents_pred, normalize=True, ensembling=True).mean()
        loss_dict['loss_elatentlpips'] = loss_elatentlpips
    else:
        loss_elatentlpips = 0
        loss_dict['loss_elatentlpips'] = loss_elatentlpips
    return loss_elatentlpips, loss_dict



def cal_adaptive_weights_type8(featkd_loss, task_loss, outkd_loss, elatentlpips_loss, last_featkd_layer=None, outkd_layer=None):
    assert last_featkd_layer is not None, "need last_featkd_layer's parameter to get gradient"
    assert outkd_layer is not None, "need outkd_layer's parameter to get gradient"
    
    from torch.autograd import grad as get_grad
    from torch import norm as get_norm

    feat_grad_featkd = get_grad(featkd_loss, last_featkd_layer, retain_graph=True)[0]
    feat_grad_outkd  = get_grad(outkd_loss,  last_featkd_layer, retain_graph=True)[0]
    feat_grad_task   = get_grad(task_loss,   last_featkd_layer, retain_graph=True)[0]
    feat_grad_elatentlpips = get_grad(elatentlpips_loss,   last_featkd_layer, retain_graph=True)[0]

    out_grad_outkd  = get_grad(outkd_loss, outkd_layer, retain_graph=True)[0]
    out_grad_task   = get_grad(task_loss,  outkd_layer, retain_graph=True)[0]
    out_grad_elatentlpips = get_grad(elatentlpips_loss,  outkd_layer, retain_graph=True)[0]

    out_weight_outkd = get_norm(out_grad_task) / (get_norm(out_grad_outkd) + 1e-6)
    out_weight_outkd = torch.clamp(out_weight_outkd, 0.0, 1e6).detach()

    out_weight_elatentlpips = get_norm(out_grad_task) / (get_norm(out_grad_elatentlpips) + 1e-6)
    out_weight_elatentlpips = torch.clamp(out_weight_elatentlpips, 0.0, 1e6).detach()

    feat_weight_task = get_norm(feat_grad_featkd) / (get_norm(feat_grad_task) + 1e-4)
    feat_weight_task = torch.clamp(feat_weight_task, 0.0, 1e4).detach()  

    return feat_weight_task, out_weight_outkd, out_weight_elatentlpips, \
        get_norm(feat_grad_featkd), get_norm(feat_grad_task), get_norm(feat_grad_outkd), get_norm(feat_grad_elatentlpips), \
        get_norm(out_grad_task), get_norm(out_grad_outkd), get_norm(out_grad_elatentlpips)




def train_distillation(args):
    chinese_sdxl_train_util.verify_sdxl_training_args(args,False)

    if args.seed is not None:
        set_seed(args.seed)

    accelerator = build_accelerator(args, fsdp_plugin=None)

    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    student, vae = build_models(args, weight_dtype, accelerator)
    teacher = build_teacher_model(args, weight_dtype, accelerator)
    del vae.decoder # cause not need docoder in current training paradigm

    # EMA
    if args.use_model_ema:
        student_ema = LitEma(student, decay=args.ema_decay).to(accelerator.device)
        accelerator.print(f"Keeping EMAs of {len(list(student_ema.buffers()))}.")

    # torch.compile
    if args.use_compile:
        student.compile(backend="cudagraphs", 
                        fullgraph=False,
                        dynamic=False)

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    
    if args.resume_from_ckpt:
        accelerator.print(f"==> resume ckpt from : {args.resume_from_ckpt}")
        msg = load_removal_model(
            student, args.resume_from_ckpt, accelerator.device, strict=True
        )
        accelerator.print(f'load state dict {msg}')

        if args.use_model_ema:
            load_path = args.resume_from_ckpt.replace(
                "diffusion_pytorch_model.bin",
                "diffusion_pytorch_model.EMA.bin"
            )
            load_litema(student_ema, load_path, map_location=accelerator.device)
            accelerator.print(f'load EMA state dict {msg}')

    # E-Latent-LPIPS
    if args.elatentlpips_loss:
        from elatentlpips import ELatentLPIPS
        # Initialize E-LatentLPIPS with the specified encoder model (options: sd15, sd21, sdxl, sd3, flux)
        # The 'augment' parameter can be set to one of the following: b, bg, bgc, bgco
        elatentlpips_model = ELatentLPIPS(encoder="sdxl", augment="bg").eval()
        elatentlpips_model = accelerator.prepare(elatentlpips_model)
    else:
        elatentlpips_model = None


    # training_models
    training_models = []
    params_to_optimize = []
    named_params_to_optimize = []

    training_models.append(student)
    params_to_optimize.append({"params": list(student.parameters()), "lr": args.learning_rate})
    named_params_to_optimize.append({"params": list(student.named_parameters()), "lr": args.learning_rate})

    n_params = 0
    for params in params_to_optimize:
        for p in params["params"]:
            n_params += p.numel()
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    accelerator.print("prepare optimizer, data loader etc.")
    
    _, _, optimizer = train_util.get_optimizer(args, 
        trainable_params=params_to_optimize, 
        named_trainable_params=named_params_to_optimize)
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)
    
    executor = ThreadPoolExecutor(max_workers=1) 

    student, optimizer, lr_scheduler = accelerator.prepare(
        student, optimizer, lr_scheduler
    )

    teacher = accelerator.prepare(teacher)
    
    if accelerator.is_main_process:
        init_kwargs = {}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers("finetuning" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs)

    loss_total = 0
    accumulate_loss = 0
    for m in training_models:
        m.train()

    dataset_class = eval(args.data_type)
    train_dataloader, _ = build_dataloader(args, 
        dataset_class, accelerator)

    global_step = args.global_step
    pbar = build_progress_bar(
        range(args.max_train_steps),  args.global_step, 
        disable=not accelerator.is_local_main_process)


    for step in range(args.global_step, args.max_train_steps):
        with accelerator.accumulate(training_models[0]): 
            batch = next(train_dataloader)
            latents, masked_image_latents = encode_clean_latents(batch, vae, weight_dtype, accelerator)

            # resize mask
            masks = batch["masks"]
            h, w = masks.shape[-2:]
            vae_ds_ratio = 2 ** (len(vae.config.block_out_channels) - 1)
            size = (h // vae_ds_ratio, w // vae_ds_ratio)
            resized_masks = F.interpolate(masks, size=size).to(accelerator.device, dtype=weight_dtype)
            
            # Sample noise
            noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)
            noisy_latents = noisy_latents.to(weight_dtype)

            # Predict the noise residual
            with accelerator.autocast():
                latent_model_input = torch.cat([
                 noisy_latents, resized_masks, masked_image_latents], dim=1)

                pred_S = student(
                    latent_model_input, timesteps=timesteps, input_ids=batch["input_ids"])

                pred_T = teacher(
                    latent_model_input, timesteps=timesteps, input_ids=batch["input_ids"])

            # target = noise
            loss_kd, loss_dict_kd = cal_KD_loss(pred_S, pred_T, args)
            loss_task, loss_dict_task = cal_task_loss(pred_S, noise, args)
            loss_elatentlpips, loss_dict_elatentlpips = cal_elatentlpips_loss(
                pred_S, noise, elatentlpips_model, 
                noise_scheduler = noise_scheduler, 
                timesteps = timesteps, 
                noisy_latents = noisy_latents, 
                args = args)

            loss_dict = loss_dict_kd | loss_dict_task | loss_dict_elatentlpips

            raw_student = accelerator.unwrap_model(student)
            feat_weight_task, \
            out_weight_outkd, \
            out_weight_elatentlpips, \
            feat_gnorm_featkd, \
            feat_gnorm_task,  \
            feat_gnorm_outkd, \
            feat_gnorm_elatentlpips, \
            out_gnorm_task,   \
            out_gnorm_outkd, \
            out_gnorm_elatentlpips  = cal_adaptive_weights_type8(
                loss_dict["loss_featkd"], 
                loss_dict["loss_task"], 
                loss_dict["loss_outkd"], 
                loss_dict["loss_elatentlpips"], 
                last_featkd_layer = raw_student.diff_model.down_blocks[2].attentions[1].proj_out.weight,
                outkd_layer = raw_student.diff_model.conv_out.conv_pw.weight)

            loss = loss_dict["loss_featkd"] * args.KD_loss_weight \
                    + feat_weight_task * ( \
                        loss_dict["loss_task"]  * args.task_loss_weight \
                        + loss_dict["loss_outkd"] * out_weight_outkd * args.KD_loss_weight \
                        + loss_dict["loss_elatentlpips"] * out_weight_elatentlpips * args.elatentlpips_loss_weight)

            accelerator.backward(loss)
            if args.max_grad_norm != 0.0:
                grad_norm = accelerator.clip_grad_norm_(
                    student.parameters(), args.max_grad_norm).item()

        optimizer.step()

        if args.use_model_ema:
            raw_student = accelerator.unwrap_model(student)
            student_ema(accelerator.unwrap_model(raw_student))

        lr_scheduler.step()
        optimizer.zero_grad()

        current_loss = loss.detach()
        accumulate_loss += current_loss
        
        # logging        
        if accelerator.sync_gradients: 
            loss_total += accumulate_loss #current_loss
            logs = {
                "avr_loss": loss_total.item() / (step + 1 - args.global_step),
                "loss": accumulate_loss.item() / accelerator.gradient_accumulation_steps, #current_loss,
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "grad_norm": grad_norm,
                'global_step': global_step,
                "feat_gnorm_featkd": feat_gnorm_featkd.item(),
                "feat_gnorm_task": feat_gnorm_task.item(),
                "feat_gnorm_outkd": feat_gnorm_outkd.item(),
                "feat_gnorm_elatentlpips": feat_gnorm_elatentlpips.item(),
                "out_gnorm_task": out_gnorm_task.item(),
                "out_gnorm_outkd": out_gnorm_outkd.item(),
                "out_gnorm_elatentlpips": out_gnorm_elatentlpips.item(),
                "feat_weight_task": feat_weight_task.item(),
                "out_weight_outkd": out_weight_outkd.item(),
                "out_weight_elatentlpips": out_weight_elatentlpips.item()
            }
            logs |= { k:v.item() for k,v in loss_dict.items()}
            pbar.set_postfix(**logs, refresh=False)

            if args.logging_dir:
                tb_logs = logs | {"rank": accelerator.process_index,}
                executor.submit(accelerator.log, tb_logs, step=global_step)

            accumulate_loss = 0

        # save model by step
        if (global_step != args.global_step \
            and args.save_every_n_steps \
            and global_step % args.save_every_n_steps == 0):
                save_path = osp.join(args.output_dir, "ckpt", f"exp-step{global_step:08d}", f"diffusion_pytorch_model.bin")
                save(student, save_path, accelerator)

                if args.use_model_ema:
                    save_path = osp.join(args.output_dir, "ckpt", f"exp-step{global_step:08d}", f"diffusion_pytorch_model.EMA.bin")
                    if accelerator.is_main_process:
                        save_litema(student_ema, save_path)
                    accelerator.print(f"d[info]: EMA Model saved at: {save_path}\n")

        pbar.update()
        global_step += 1

    # save the final model
    save_path = osp.join(args.output_dir, "ckpt", f"exp-step{global_step:08d}", f"diffusion_pytorch_model.bin")
    save(student, save_path, accelerator)

    if args.use_model_ema:
        save_path = osp.join(args.output_dir, "ckpt", f"exp-step{global_step:08d}", f"diffusion_pytorch_model.EMA.bin")
        save_litema(student_ema, save_path)
        accelerator.print(f"d[info]: EMA Model saved at: {save_path}\n")

    accelerator.wait_for_everyone()
    accelerator.end_training()

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_training_arguments(parser, False)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    # config_util.add_config_arguments(parser)

    common_arguments(parser)
   
    '''add teacher config path'''
    parser.add_argument('--teacher_config_path', type=str, default=None)
    parser.add_argument('--teacher_weight_path', type=str, default=None)
    
    parser.add_argument('--kl_feat_loss', action='store_true', 
        help='enable KLDivLoss for feat and output KD.')
    parser.add_argument('--kl_tempeature', type=float, default=1.0, dest = 'kl_temp', 
        help='temperature for the smoothment of soft label feature.')
    
    parser.add_argument('--mse_feat_loss', action='store_true', 
        help='enable MSELoss for feat and output KD.')
    
    parser.add_argument('--feat_index_T', nargs='*', type=int, default=[4,], 
        help='index list of Teacher intermediate feautures for KD.')
    parser.add_argument('--feat_index_S', nargs='*', type=int, default=[4,], 
        help='index list of Student intermediate feautures for KD.')
    parser.add_argument('--feat_loss_weight', nargs='*',type=float, default=[0.2,], 
        help='loss weights of intermediate feautures for KD.')


    parser.add_argument('--task_loss', action='store_true', 
        help='enable MSELoss for output and gt_noise.')
    parser.add_argument('--task_loss_weight', type=float, default=1.0, 
        help='weight multiplied to loss_task.')
    parser.add_argument('--KD_loss_weight', type=float, default=1.0, 
        help='weight multiplied to loss_kd.')
    
    parser.add_argument('--elatentlpips_loss', action='store_true', 
        help='enable MSELoss for output and gt_noise.')
    parser.add_argument('--elatentlpips_loss_weight', type=float, default=1.0, 
        help='weight multiplied to loss_task.')
    

    parser.add_argument('--use_model_ema', action='store_true', 
        help='enable EMA on training model.')
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    
    parser.add_argument('--use_compile', action='store_true', 
        help='use torch.compile on foward & backward.')
    
    # datatype
    parser.add_argument('--data_type', 
        type=str, default="RemovalDataset",
        choices=['RemovalDataset', 'RemovalDataset_v1_2'], 
        help='different mask assignment strategy.')
    
    return parser


if __name__ == "__main__":
#    timeout_seconds = 1800
#    timeout_timedelta = datetime.timedelta(seconds=timeout_seconds)
#    torch.distributed.init_process_group(backend='nccl', timeout=timeout_timedelta)

    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    
    train_distillation(args)


