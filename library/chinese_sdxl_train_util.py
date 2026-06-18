import os
import sys
import gc
import re
import json
import math
import time
import toml
import shutil
import argparse
from typing import Optional
from tqdm import tqdm
from PIL import Image
import importlib

import torch
from torch.utils.tensorboard import SummaryWriter

from library import train_util
from transformers import BertTokenizer, BertTokenizerFast, ChineseCLIPTextModel, PreTrainedTokenizerFast, T5Tokenizer, T5ForConditionalGeneration

from diffusers import (
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    KDPM2DiscreteScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from diffusers.models import UNet2DConditionModel, Transformer2DModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertTokenizerFast, ChineseCLIPTextModel
from library import train_util

# from mmmp_text import DebertaV2Model
# from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
# from diffusers_patch.models.vivo_llm2vec import LLM2VecWithoutPool
# from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

DEFAULT_NOISE_OFFSET = 0.0357

def load_target_model(args, accelerator, pipe_class, weight_dtype):
    # load models for each process
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            print(f"loading model for process {accelerator.process_index}/{accelerator.state.num_processes}")

            (
                text_encoder1,
                text_encoder2,
                vae,
                unet,
            ) = _load_target_model(
                args,
                args.pretrained_model_name_or_path,
                args.vae,
                pipe_class,
                weight_dtype,
                accelerator.device if args.lowram else "cpu",
            )

            gc.collect()
            torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    return text_encoder1, text_encoder2, vae, unet


def _load_target_model(args, name_or_path: str, vae_path: Optional[str], pipe_class, weight_dtype, device="cpu"):
    name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path

    model_index_path = os.path.join(name_or_path, 'model_index.json')
    model_index = read_json(model_index_path)
    TextEncoderLib1 = model_index['text_encoder'][0]
    TextEncoderLib2 = model_index['text_encoder_2'][0]
    TextEncoderClass1 = model_index['text_encoder'][-1]
    TextEncoderClass2 = model_index['text_encoder_2'][-1]
    
    library1 = importlib.import_module(TextEncoderLib1)
    library2 = importlib.import_module(TextEncoderLib2)
    TextEncoderClass1 = getattr(library1, TextEncoderClass1)
    TextEncoderClass2 = getattr(library2, TextEncoderClass2)
    
    if 'unet' in model_index:
        UNetClass = eval(model_index['unet'][-1])
        unet_dir = 'unet'
    elif 'transformer' in model_index:
        UNetClass = eval(model_index['transformer'][-1])
        unet_dir = 'transformer'
        
    print(f"TextEncoderClass1:{TextEncoderClass1}")
    print(f"TextEncoderClass2:{TextEncoderClass2}")
    print(f"UNetClass:{UNetClass}")
    
    vae = AutoencoderKL.from_pretrained(os.path.join(name_or_path, 'vae'), torch_dtype=weight_dtype, low_cpu_mem_usage=False, device_map=None)
    unet = UNetClass.from_pretrained(os.path.join(name_or_path, unet_dir), torch_dtype=weight_dtype, low_cpu_mem_usage=False, device_map=None, ignore_mismatched_sizes=True)

    text_encoder1 = TextEncoderClass1.from_pretrained(os.path.join(name_or_path, 'text_encoder'), torch_dtype=weight_dtype)
    text_encoder2 = TextEncoderClass2.from_pretrained(os.path.join(name_or_path, 'text_encoder_2'), torch_dtype=weight_dtype)
    
    vae_version = vae.config.version if 'version' in vae.config else ''
    if vae_version == 'vivo':
        vae.quant_conv = torch.nn.Identity()
        vae.post_quant_conv = torch.nn.Identity()  
    
    return text_encoder1, text_encoder2, vae, unet


def load_tokenizers(args: argparse.Namespace):
    print("prepare tokenizers")
    model_index_path = os.path.join(args.pretrained_model_name_or_path, 'model_index.json')
    model_index = read_json(model_index_path)
    ToeknierLib1 = model_index['tokenizer'][0]
    ToeknierLib2 = model_index['tokenizer_2'][0]
    TokenierClass1 = model_index['tokenizer'][-1]
    TokenierClass2 = "BertTokenizer"  # ToDo: model_index['tokenizer_2'][-1]
    
    library1 = importlib.import_module(ToeknierLib1)
    library2 = importlib.import_module(ToeknierLib2)
    TokenierClass1 = getattr(library1, TokenierClass1)
    TokenierClass2 = getattr(library2, TokenierClass2)
    

    tokenizer_1 = TokenierClass1.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer')
    tokenizer_2 = TokenierClass2.from_pretrained(args.pretrained_model_name_or_path, subfolder='tokenizer_2')
    tokeniers = [tokenizer_1, tokenizer_2]

    if hasattr(args, "max_token_length") and args.max_token_length is not None:
        print(f"update token length: {args.max_token_length}")

    return tokeniers


def get_hidden_states_sdxl(
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    tokenizer1: BertTokenizerFast,
    tokenizer2: BertTokenizerFast,
    text_encoder1: ChineseCLIPTextModel,
    text_encoder2: ChineseCLIPTextModel,
    weight_dtype: Optional[str] = None,
    attention_mask1: torch.Tensor = None,
    attention_mask2: torch.Tensor = None,
):
    # input_ids: b,n,77 -> b*n, 77
    b_size = input_ids1.size()[0]
    input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77
    if attention_mask1 is not None:
        attention_mask1 = attention_mask1.reshape((-1, tokenizer1.model_max_length))
        attention_mask2 = attention_mask2.reshape((-1, tokenizer2.model_max_length))

    hidden_states1, _ = encode_token(input_ids1, attention_mask1, text_encoder1)
    hidden_states2, pool2 = encode_token(input_ids2, attention_mask2, text_encoder2)

    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))
    if weight_dtype is not None:
        # this is required for additional network training
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2


def encode_token(input_ids, attention_mask, text_encoder):
    
    # T5
    if isinstance(text_encoder, T5ForConditionalGeneration):
        prompt_embeds = text_encoder.encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        pooled_prompt_embeds = None
        prompt_embeds = prompt_embeds.hidden_states[-1]
    
    # clip Bert
    elif isinstance(text_encoder, ChineseCLIPTextModel): 
        prompt_embeds = text_encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds['pooler_output']
        prompt_embeds = prompt_embeds.hidden_states[-2]

    # 3mp_Bert\Qwen2Model\LLM2VecWithoutPool\GLMModel
    else:
        prompt_embeds = text_encoder(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        if 'last_hidden_states' in prompt_embeds:
            prompt_embeds = prompt_embeds.last_hidden_states 
        else:
            prompt_embeds = prompt_embeds.last_hidden_state
            
        pooled_prompt_embeds = prompt_embeds.mean(dim=1)
    
    return prompt_embeds, pooled_prompt_embeds
    



def prepare_logging(args: argparse.Namespace, is_main_process):
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())
    
    log_with = args.log_with
    if log_with in ["tensorboard", "all"]:
        if logging_dir is None:
            raise ValueError("logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください")
    tensorboard_dir = os.path.join(logging_dir, 'tensorboard')
    
    writer = None
    if is_main_process:
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        if args.script_args:
            sh_basename = os.path.basename(args.script_args)
            sh_dst_path = os.path.join(logging_dir, sh_basename)
            data_basename = os.path.basename(args.dataset_config)
            data_dst_path = os.path.join(logging_dir, data_basename)
            shutil.copyfile(args.script_args, sh_dst_path)
            shutil.copyfile(args.dataset_config, data_dst_path)
    
        writer = SummaryWriter(tensorboard_dir)
    
    return writer


def verify_sdxl_training_args(args: argparse.Namespace, supportTextEncoderCaching: bool = True):
    
    if args.clip_skip is not None:
        print("clip_skip will be unexpected / SDXL学習ではclip_skipは動作しません")

    if args.multires_noise_iterations:
        print(
            f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET}, but noise_offset is disabled due to multires_noise_iterations"
        )
    else:
        if args.noise_offset is None:
            args.noise_offset = DEFAULT_NOISE_OFFSET
        elif args.noise_offset != DEFAULT_NOISE_OFFSET:
            print(
                f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET} / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されています"
            )
        print(f"noise_offset is set to {args.noise_offset}")

    assert (
        not hasattr(args, "weighted_captions") or not args.weighted_captions
    ), "weighted_captions cannot be enabled in SDXL training currently / SDXL学習では今のところweighted_captionsを有効にすることはできません"

    if supportTextEncoderCaching:
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
            print(
                "cache_text_encoder_outputs is enabled because cache_text_encoder_outputs_to_disk is enabled / "
                + "cache_text_encoder_outputs_to_diskが有効になっているためcache_text_encoder_outputsが有効になりました"
            )


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb


def get_size_embeddings(orig_size, crop_size, target_size, device):
    emb1 = get_timestep_embedding(orig_size, 256)
    emb2 = get_timestep_embedding(crop_size, 256)
    emb3 = get_timestep_embedding(target_size, 256)
    vector = torch.cat([emb1, emb2, emb3], dim=1).to(device)
    return vector


def add_sdxl_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--cache_text_encoder_outputs", action="store_true", help="cache text encoder outputs / text encoderの出力をキャッシュする"
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
    )
    

def set_unet_eff_attn(unet, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        print("Enable memory efficient attention for U-Net")
        unet.set_use_memory_efficient_attention_xformers(False, True)
    elif xformers:
        print("Enable xformers for U-Net")
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです")

        unet.set_use_memory_efficient_attention_xformers(True, False)
    elif sdpa:
        print("Enable SDPA for U-Net")
        unet.set_use_sdpa(True)

def set_diffusers_xformers_flag(model, valid):
    def fn_recursive_set_mem_eff(module: torch.nn.Module):
        if hasattr(module, "set_use_memory_efficient_attention_xformers"):
            module.set_use_memory_efficient_attention_xformers(valid)

        for child in module.children():
            fn_recursive_set_mem_eff(child)

    fn_recursive_set_mem_eff(model)


def read_json(json_path):
    return json.load(open(json_path))