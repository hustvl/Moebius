from typing import Optional, Union, Dict
import torch
from torch import nn
import os
import os.path as osp

class RemovalModel(nn.Module):
    def __init__(
            self,
            diff_model,
            num_embeddings: int,
            embedding_dim: int) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )  
        self.diff_model = diff_model
        self.num_embeddings=num_embeddings

    def forward(
            self,
            noisy_latents,
            timesteps,
            input_ids):
        encoder_hidden_states = self.embedding_layer(input_ids)  

        noise_pred = self.diff_model(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states)
        return noise_pred


def load_cfg(cfg_path: Union[str,Dict]):
    if isinstance(cfg_path, str):
        import yaml
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
    else: # from dict
        from omegaconf import OmegaConf
        config = OmegaConf.create(cfg_path)
    return config


from diffusers import UNet2DConditionModel
from model_lib import *

def build_removal_model(
    config_path=None,
    num_embeddings=20,
    ):
    config = load_cfg(config_path)

    latent_size = config['data']['image_size'] // config['vae']['downsample_ratio']

    model_cfg = config['model']
    model_cfg.update(dict(sample_size=latent_size))
    if "in_channels" not in model_cfg:
        model_cfg['in_channels'] = getattr(model_cfg, 'in_chans', None)
        model_cfg.pop("in_chans",None)
    if "out_channels" not in model_cfg:
        model_cfg['out_channels'] = getattr(model_cfg, 'out_chans', None)
        model_cfg.pop("out_channels",None)

    model_type = model_cfg.pop("model_type")

    if "Lambda" in model_type or "λ" in model_type:
        model_cfg["num_embeddings"] = num_embeddings

    diff = eval(model_type)(**model_cfg)

    embedding_dim = diff.config.encoder_hid_dim
    return RemovalModel(
        diff_model=diff,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim
    )


def load_removal_model(
        model:nn.Module,
        weight_path:str, 
        device='cpu', 
        dtype=torch.float32,
        strict=True
        ):
    model.to(device=device, dtype=dtype)
    
    state_dict = None
    if ".EMA." in osp.basename(weight_path):
        from model_lib.nets.layers.ema import load_ema_into_model, load_litema
        state_dict = load_ema_into_model(model,weight_path).state_dict()
    else:
        state_dict = torch.load(weight_path,map_location=device)

    msg = model.load_state_dict(state_dict,strict=strict)
    return msg

