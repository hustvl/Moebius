from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import LegacyConfigMixin, register_to_config
from diffusers.utils import deprecate, logging
from diffusers.models.attention import BasicTransformerBlock
# from diffusers.models.embeddings import ImagePositionalEmbeddings, PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
# from diffusers.models.modeling_utils import LegacyModelMixin
# from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.models.transformers.transformer_2d import Transformer2DModel

from diffusers.utils.torch_utils import maybe_allow_in_graph

from ..sana.basic_modules import GLUMBConv as MixFFN


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




@maybe_allow_in_graph
class MixTransformerBlock(BasicTransformerBlock):
    r"""
    replace FeedForward in BasicTransformerBlock with MixFFN
    """

    def __init__(self, 
        mlp_ratio: float = 2.5, 
        dim: int = 2240,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        **kwargs) -> None:
        super(MixTransformerBlock, self).__init__(
            dim, num_attention_heads, attention_head_dim, **kwargs)
        # dim = kwargs["dim"]

        # 3. Feed-forward
        self.ff = MixFFN(
            in_features=dim, 
            hidden_features=int(dim*mlp_ratio), 
            out_feature=dim)


class MixTransformer2DModel(Transformer2DModel):
    """
    replace BasicTransformerBlock in Transformer2DModel with MixTransformerBlock
    used in U-Net arch, without time_modulation
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "MixTransformerBlock"]
    _skip_layerwise_casting_patterns = ["latent_image_embedding", "norm"]

    @register_to_config
    def __init__(self,       
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        interpolation_scale: float = None,
        use_additional_conditions: Optional[bool] = None,
        mlp_ratio=2.5,
    ):
        super(Transformer2DModel, self).__init__()

        # Set some common variables used across the board.
        self.use_linear_projection = use_linear_projection
        self.interpolation_scale = interpolation_scale
        self.caption_channels = caption_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        self.use_additional_conditions = False

        # 2. Initialize the right blocks.
        self.is_input_continuous=True
        self._init_continuous_input(norm_type=norm_type)


    def _init_continuous_input(self, norm_type):
        print("_init_continuous_input")
        self.norm = torch.nn.GroupNorm(
            num_groups=self.config.norm_num_groups, num_channels=self.in_channels, eps=1e-6, affine=True
        )
        if self.use_linear_projection:
            self.proj_in = torch.nn.Linear(self.in_channels, self.inner_dim)
        else:
            # a 1x1 conv, no need to change into dwconv
            self.proj_in = torch.nn.Conv2d(self.in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0) 

        self.transformer_blocks = nn.ModuleList(
            [
                MixTransformerBlock(
                    self.config.mlp_ratio,
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        if self.use_linear_projection:
            self.proj_out = torch.nn.Linear(self.inner_dim, self.out_channels)
        else:
            # a 1x1 conv, no need to change into dwconv
            self.proj_out = torch.nn.Conv2d(self.inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)
