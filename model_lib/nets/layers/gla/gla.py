import warnings
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.cache_utils import Cache, DynamicCache
from diffusers.utils import logging

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.models.gla.modeling_gla import GLAMLP, GLABlock, GatedLinearAttention, GLAConfig
from fla.modules import FusedCrossEntropyLoss, RMSNorm, ShortConvolution, FusedRMSNormSwishGate

import torch._dynamo
torch._dynamo.config.suppress_errors = True

logger = logging.get_logger(__name__)



class GatedLinearAttention_ForwardWrapper(GatedLinearAttention):
    def forward(self,*args,**kwargs):
        return super(GatedLinearAttention_ForwardWrapper, self).forward(*args,**kwargs)[0]

class GatedLinearCrossAttention(GatedLinearAttention):
    def __init__(self, **kwargs):
        cross_attention_dim = kwargs.pop('cross_attention_dim', None)
        super().__init__(**kwargs)
        self.cross_attention_dim = cross_attention_dim
        
        self.encoder_proj = nn.Sequential(
            nn.Linear(self.cross_attention_dim, self.hidden_size, bias=False),
            RMSNorm(hidden_size=self.hidden_size, eps=kwargs['norm_eps'])
        ) if cross_attention_dim else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        enc_proj = self.encoder_proj(encoder_hidden_states)

        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode



        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            position_ids = kwargs.get('position_ids', None)
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(enc_proj),
                mask=conv_mask,
                cache=conv_state_q,
                seq_idx=position_ids)
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_k,
                seq_idx=position_ids)
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_v,
                seq_idx=position_ids)
        else:
            q = self.q_proj(enc_proj)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        gk = self.gk_proj(hidden_states)

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        if self.num_kv_groups > 1:
            k, v, gk = (repeat(x, 'b t (h d) -> b t (h g) d', h=self.num_kv_heads, g=self.num_kv_groups) for x in (k, v, gk))
        else:
            k, v, gk = (rearrange(x, 'b t (h d) -> b t h d', h=self.num_kv_heads) for x in (k, v, gk))
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        cu_seqlens = kwargs.get('cu_seqlens', None)
        d_args = dict(q=q, k=k, v=v, 
            head_first=False)
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(**d_args, gk=gk)
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(**d_args, g=gk)
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(**d_args, g=gk)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if self.use_output_gate:
            g = self.g_proj(enc_proj)
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b t (h d) -> b t h d', h=self.num_heads)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, 'b t h d -> b t (h d)')
            else:
                o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, None #past_key_values


class GatedLinearCrossAttention_ForwardWrapper(GatedLinearCrossAttention):
    def forward(self,*args,**kwargs):
        return super(GatedLinearCrossAttention_ForwardWrapper, self).forward(*args,**kwargs)[0]

GLSA = GatedLinearAttention_ForwardWrapper
GLCA = GatedLinearCrossAttention_ForwardWrapper

DEFAULT_GLA_CONFIG = dict(
    mode = 'chunk',
    hidden_size = 2048,
    expand_k = 0.5,
    expand_v = 1.0,
    num_heads = 4,
    num_kv_heads = None,
    feature_map = None,
    use_short_conv = False,
    conv_size = 4,
    conv_bias = False,
    use_output_gate = True,
    gate_fn = 'swish',
    elementwise_affine = True,
    norm_eps = 1e-6,
    gate_logit_normalizer = 16,
    gate_low_rank_dim = 16,
    clamp_min = None,
    fuse_norm = True,
    layer_idx = None,
)


if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    def test_attns():
        config = GLAConfig(**DEFAULT_GLA_CONFIG, cross_attention_dim = 768)

        x = torch.randn(7,256,config.hidden_size).to('cuda:0')
        y = torch.randn(7,100,config.cross_attention_dim).to('cuda:0')

        from diffusers.models.attention import Attention
        xattn = Attention(
            query_dim = config.hidden_size, 
            cross_attention_dim = config.cross_attention_dim,
            heads = config.num_heads).to('cuda:0')
        xattn_output = xattn(x, encoder_hidden_states=y)
        print(xattn_output.shape)

        gl_xattns = nn.ModuleList([
            GatedLinearAttention_ForwardWrapper(
                **DEFAULT_GLA_CONFIG, 
                # cross_attention_dim = config.cross_attention_dim
            ).to("cuda:0"),
            RMSNorm(hidden_size=config.hidden_size,eps=config.norm_eps).to("cuda:0"),
            GatedLinearCrossAttention_ForwardWrapper(
                **DEFAULT_GLA_CONFIG, 
                cross_attention_dim = config.cross_attention_dim
            ).to("cuda:0")]
        )
        x1 = gl_xattns[0](x, encoder_hidden_states=y)
        x2 = gl_xattns[1](x1)
        glattn_output = gl_xattns[0](x2, encoder_hidden_states=y)
        print(glattn_output.shape)
        
        exit(0)

    test_attns()