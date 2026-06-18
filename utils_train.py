import shutil
import os
import os.path as osp
from pathlib import Path
from typing import Dict, List, Union

import argparse

import torch
from accelerate import Accelerator, DistributedType
from diffusers.models import AutoencoderKL


from library import train_util, chinese_sdxl_train_util

from removal.v1_2 import load_cfg, build_removal_model

def build_accelerator(args: argparse.Namespace, **kwargs) -> Accelerator:
    accelerator = train_util.prepare_accelerator(args, **kwargs)
    accelerator.print("prepare accelerator done")
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        # deepspeedの場合はtrain_micro_batch_size_per_gpuを設定しておく
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1
    return accelerator


def build_vae(model_cfg: Dict) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(model_cfg["vae"]['model_dir'])
    return vae


def build_models(args: argparse.Namespace, weight_dtype: str, accelerator: Accelerator) -> List[torch.nn.Module]:
    model_cfg = load_cfg(args.model_config_path)

    if args.pretrained_model_name_or_path: # interface for from_pretrained mathod
        model_cfg['vae']['model_dir'] = osp.join(args.pretrained_model_name_or_path, 'vae')
        model_cfg['model']['model_dir'] = osp.join(args.pretrained_model_name_or_path, 'unet')

    removal_model = build_removal_model(model_cfg, args.num_embeddings)
    vae = build_vae(model_cfg)

    accelerator.print(f"weight_dtype:{weight_dtype}")
    accelerator.print(f"vae:{vae.dtype}")
    if getattr(removal_model, 'unet', None):
        accelerator.print(f"unet:{removal_model.unet.dtype}")
    else:
        accelerator.print(f"diff_model:{removal_model.diff_model.dtype}")

    if args.reset_unet_parameters:
        accelerator.print("==> reset unet parameters")
        for name, layer in removal_model.unet.named_modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                if accelerator.is_main_process:
                    accelerator.print(f'parameters reset: {name}')

    vae.requires_grad_(False).eval()
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
    vae.to(accelerator.device, dtype=vae_dtype)

    removal_model.to(accelerator.device, dtype=torch.float32)


    if accelerator.is_main_process:
        from pprint import pprint
        pprint("Model Config:")
        pprint(removal_model.diff_model.config)

    if args.gradient_checkpointing:
        removal_model.diff_model.enable_gradient_checkpointing()

    # set xformer/mem_eff_attn
    accelerator.print(f"Enable memory efficient attention, mem_eff_attn:{args.mem_eff_attn}, xformers:{args.xformers}")
    chinese_sdxl_train_util.set_diffusers_xformers_flag(removal_model.diff_model, True)
    chinese_sdxl_train_util.set_diffusers_xformers_flag(vae, True)

    return removal_model, vae




def build_dataloader(
    args: argparse.Namespace,
    # train_data_path: Union[str, Path], 
    dataset_class = None,
    accelerator: Accelerator = None
) -> List[Union[torch.utils.data.DataLoader, List[str]]]:
    def cycle(dl): yield from (data for _ in iter(int, 1) for data in dl)

    if args.data_config: # YAML dataset builder
        accelerator.print(f'[info]: build dataset from {args.data_config}')
        from omegaconf import OmegaConf, DictConfig
        d_cfg = OmegaConf.load(args.data_config).data
        train_data_path = d_cfg.path
        rand_mask_config = d_cfg.rand_mask_config
        use_rand_mask = d_cfg.use_rand_mask
        use_extra_fg_mask = d_cfg.use_extra_fg_mask
        ex_masks4pure_bg = d_cfg.extra_ann_files_4_PureBackTrain_2_RandMask
        train_jsons = train_data_path if not isinstance(train_data_path, str) else [train_data_path]
    else:
        accelerator.print(f'[info]: build dataset from proto args.')
        train_data_path = args.train_data_path
        rand_mask_config = args.rand_mask_config
        use_rand_mask = args.use_rand_mask
        use_extra_fg_mask = args.use_extra_fg_mask
        ex_masks4pure_bg = args.extra_ann_files_4_PureBackTrain_2_RandMask
        train_jsons = train_data_path if not isinstance(train_data_path, str) else [train_data_path.strip()]

    for i, json_file_path in enumerate(train_jsons):
            accelerator.print(f"[info]: ==> jsonl_idx:{i}, jsonl_path:{json_file_path}")

    train_dataset = dataset_class(
        ann_files=train_jsons,
        image_size=args.image_size,
        mask_config=rand_mask_config,
        extra_ann_files_4_PureBackTrain_2_RandMask=ex_masks4pure_bg,
        num_embeddings = args.num_embeddings,
        use_rand_mask=use_rand_mask,
        use_extra_fg_mask=use_extra_fg_mask,
        quiet=True # disable print on multi devices
    )

    accelerator.print(
        f'[info]: has {len(train_dataset.data_source_bg)} background task samples.')
    accelerator.print(
        f'[info]: has {len(train_dataset.data_source_fg)} foreground task samples.')
    accelerator.print(
        f'[info]: has {len(train_dataset.data_source)} total samples.')



    accelerator.print("[info]: copying train_jsons...")
    for train_json in train_jsons:
        dst_dir = osp.join(args.output_dir,"train_jsons")
        os.makedirs(dst_dir, exist_ok=True)
        dst_json = osp.join(dst_dir,osp.basename(train_json))
        if not os.path.exists(dst_json):
            shutil.copyfile(train_json, dst_json)

    batch_size, num_workers = args.batch_size, args.num_workers
    accelerator.print(f"[info]: batch_size is {batch_size}, num_workers is {num_workers}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True, 
        drop_last=False,
        collate_fn=dataset_class.collate_fn
    )
    train_dataloader = accelerator.prepare(train_dataloader)
    return cycle(train_dataloader), train_jsons

def build_progress_bar(iterator, initial=0, 
    disable=False, desc='steps', mininterval=60, miniters=50, dynamic_ncols=False, 
    bar_format = "{l_bar}{bar:3}{r_bar}"):
    from tqdm import tqdm
    progress_bar = tqdm(
        iterator,  # range(len(train_dataloader)),
        initial=initial, 
        disable=disable, 
        mininterval=mininterval,
        miniters=miniters,
        bar_format=bar_format,
        dynamic_ncols=dynamic_ncols,
        desc=desc)
    return progress_bar



def save(model: torch.nn.Module, save_path: Union[str, Path], accelerator: Accelerator) -> None:
    accelerator.wait_for_everyone()
    removal_model_states = accelerator.unwrap_model(model).state_dict()
    if accelerator.is_main_process:
        os.makedirs(osp.dirname(save_path), exist_ok=True)
        torch.save(removal_model_states, save_path)
        accelerator.print(f"\n[info]: Model saved at: {save_path}\n")

    torch.cuda.empty_cache()


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no_half_vae", action="store_true", 
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE)")
    parser.add_argument("--reset_unet_parameters", action="store_true", 
        help="reset unet parameters")
    parser.add_argument("--lognorm_t", action="store_true", help="whether lognorm timestep")

    parser.add_argument("--global_step", type=int, default=0, help="global_step")

    # data_cfg
    parser.add_argument("--train_data_path", type=str, nargs='+', default=None, 
        help="current train json data path, support multi paths split by space")
    parser.add_argument('--use_rand_mask', type=bool, default=True)
    parser.add_argument("--rand_mask_config", type=str, help="rand mask yaml",
        default="config/rand_mask_cfg/random_medium_512.yaml")
    parser.add_argument('--use_extra_fg_mask', type=bool, default=True)
    parser.add_argument("--extra_ann_files_4_PureBackTrain_2_RandMask", 
        type=str, default=None)
    parser.add_argument("--data_config", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument('--num_embeddings', type=int, default=20)
    parser.add_argument('--image_size', type=int, default=512)
    

    parser.add_argument('--model_config_path', type=str, default="")

    parser.add_argument('--cos_loss', action='store_true', default=False, help='whether use cosine similarity loss')

    parser.add_argument('--guidance_scale', type=float, default=1.0, help='class free guidance')

    parser.add_argument("--resume_from_ckpt", type=str, default="", help="resume from ckpt")