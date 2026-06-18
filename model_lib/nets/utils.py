from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput


@dataclass
class CustomOutput(UNet2DConditionOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None
    block_outputs: List[torch.Tensor] = None
    cross_attention_maps: List[torch.Tensor] = None




common_args = {
    "sample_size": None,
    "in_channels": 4,
    "out_channels": 4,
    "center_input_sample": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "down_block_types": (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "up_block_types": (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    "only_cross_attention": False,
    "block_out_channels": (320, 640, 1280, 1280),
    "layers_per_block": 2,
    "downsample_padding": 1,
    "mid_block_scale_factor": 1,
    "dropout": 0.0,
    "act_fn": "silu",
    "norm_num_groups": 32,
    "norm_eps": 1e-5,
    "cross_attention_dim": 1280,
    "transformer_layers_per_block": 1,
    "reverse_transformer_layers_per_block": None,
    "encoder_hid_dim": None,
    "encoder_hid_dim_type": None,
    "attention_head_dim": 8,
    "num_attention_heads": None,
    "dual_cross_attention": False,
    "use_linear_projection": False,
    "class_embed_type": None,
    "addition_embed_type": None,
    "addition_time_embed_dim": None,
    "num_class_embeds": None,
    "upcast_attention": False,
    "resnet_time_scale_shift": "default",
    "resnet_skip_time_act": False,
    "resnet_out_scale_factor": 1.0,
    "time_embedding_type": "positional",
    "time_embedding_dim": None,
    "time_embedding_act_fn": None,
    "timestep_post_act": None,
    "time_cond_proj_dim": None,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "projection_class_embeddings_input_dim": None,
    "attention_type": "default",
    "class_embeddings_concat": False,
    "mid_block_only_cross_attention": None,
    "cross_attention_norm": None,
    "addition_embed_type_num_heads": 64,
}
