import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import logging

from ..utils import (
    _default, _exists,
    _einsum, _rearrange,
    calc_rel_pos
)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

logger = logging.get_logger(__name__)



class MultiQuerySelfLambda(nn.Module):
    def __init__(self, dim, *, dim_k, n = None, r = None, heads = 4, dim_out = None, dim_u = 1):
        super().__init__()
        dim_out = _default(dim_out, dim)
        self.u = dim_u
        self.heads = heads

        assert (dim_out % heads) == 0, "'dim_out' must be divisible by number of heads for multi-head query"
        dim_v = dim_out // heads

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_k * dim_u, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_v * dim_u, 1, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = _exists(r)
        if _exists(r): # local
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))
        else: # global
            assert _exists(n), 'You must specify the window size (n=h=w)'
            rel_lengths = 2 * n - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_u))
            self.rel_pos = calc_rel_pos(n)

    def forward(self, x):
        x = _rearrange(x, 'b h w c -> b c h w')
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        Q = self.norm_q(q)
        V = self.norm_v(v)

        Q = _rearrange(Q, 'b (h k) hh ww -> b h k (hh ww)', h = h) # multi query
        k = _rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        V = _rearrange(V, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        k = k.softmax(dim=-1)
        λc = _einsum('b u k m, b u v m -> b k v', k, V)
        Yc = _einsum('b h k n, b k v -> b h v n', Q, λc)

        if self.local_contexts:
            V = _rearrange(V, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
            λp = self.pos_conv(V)
            Yp = _einsum('b h k n, b k v n -> b h v n', Q, λp.flatten(3))
        else:
            n, m = self.rel_pos.unbind(dim = -1)
            rel_pos_emb = self.rel_pos_emb[n, m]
            λp = _einsum('n m k u, b u v m -> b n k v', rel_pos_emb, V)
            Yp = _einsum('b h k n, b n k v -> b h v n', Q, λp)
 
        Y = Yc + Yp
        appearance = _rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        appearance = _rearrange(appearance, 'b c h w -> b h w c')

        return appearance


class MultiQueryCrossLambda(nn.Module):
    def __init__(self, dim, *, dim_k, dim_cross=None, n = None, m=None, r = None, heads = 4, dim_out = None, dim_u = 1):
        super().__init__()
        dim_out = _default(dim_out, dim)
        self.u = dim_u
        self.heads = heads

        assert (dim_out % heads) == 0, "'dim_out' must be divisible by number of heads for multi-head query"
        dim_v = dim_out // heads

        if dim_cross is None:
            dim_cross = dim

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias = False)
        self.to_k = nn.Linear(dim_cross, dim_k * dim_u, bias = False)
        self.to_v = nn.Linear(dim_cross, dim_v * dim_u, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm1d(dim_v * dim_u) # sequence, so 1d norm

        self.local_contexts = _exists(r)  # do not work in xattn if d_q != d_k
        if _exists(r):
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv2d(dim_u, dim_k, (1, r), padding = (0, r // 2))
        else:
            assert _exists(n), 'You must specify the window size (n=h=w)'
            assert _exists(m), 'You must specify the hidden_state length for xattn (m=len_k)'
            self.rel_pos_emb = nn.Parameter(torch.randn(n*n, m, dim_k, dim_u))
            self.rel_pos = torch.stack(torch.meshgrid(torch.arange(n*n), torch.arange(m)), dim=-1)  # self.rel_pos = calc_rel_pos(n)

    def forward(self, x, hidden_states):
        x = _rearrange(x, 'b h w c -> b c h w')
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x)

        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        k = _rearrange(k, 'b l c -> b c l')
        v = _rearrange(v, 'b l c -> b c l')

        Q = self.norm_q(q)
        V = self.norm_v(v)
        
        Q = _rearrange(Q, 'b (h k) hh ww -> b h k (hh ww)', h = h) # multi query
        k = _rearrange(k, 'b (u k) l -> b u k l', u = u) # l denotes d_k(len_k)
        V = _rearrange(V, 'b (u v) l -> b u v l', u = u) # l denotes d_k(len_k)

        k = k.softmax(dim=-1)
        λc = _einsum('b u k l, b u v l -> b k v', k, V)
        Yc = _einsum('b h k n, b k v -> b h v n', Q, λc)


        if self.local_contexts:
            V = _rearrange(V, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
            λp = self.pos_conv(V)
            Yp = _einsum('b h k n, b k v n -> b h v n', Q, λp.flatten(3))
        else:
            n, m = self.rel_pos.unbind(dim = -1)
            rel_pos_emb = self.rel_pos_emb[n, m]
            λp = _einsum('n m k u, b u v m -> b n k v', rel_pos_emb, V)
            Yp = _einsum('b h k n, b n k v -> b h v n', Q, λp)

        Y = Yc + Yp
        appearance = _rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        appearance = _rearrange(appearance, 'b c h w -> b h w c')

        return appearance




class MQSλ_FwdWrapper(MultiQuerySelfLambda):
    def __init__(self, *args, **kwargs):
        super(MQSλ_FwdWrapper, self).__init__(*args, **kwargs)
    def forward(self, x, encoder_hidden_states=None, attention_mask=None):
        h = w = int(x.shape[1] ** 0.5)    # h = w = self.sample_size
        x = _rearrange(x, "b (h w) c -> b h w c", h = h,w = w)
        out_x = super(MQSλ_FwdWrapper, self).forward(x)
        return _rearrange(out_x, "b h w c -> b (h w) c")

class MQXλ_FwdWrapper(MultiQueryCrossLambda):
    def __init__(self, *args, **kwargs):
        super(MQXλ_FwdWrapper, self).__init__(*args, **kwargs)
    def forward(self, x , encoder_hidden_states=None, attention_mask=None):
        h = w = int(x.shape[1] ** 0.5) 
        x = _rearrange(x, "b (h w) c -> b h w c", h = h,w = w)
        assert encoder_hidden_states is not None, "xattn need encoder_hidden_states."
        out_x = super(MQXλ_FwdWrapper, self).forward(x, encoder_hidden_states)
        return _rearrange(out_x, "b h w c -> b (h w) c")


MQSλ = MQSλ_FwdWrapper
MQCλ = MQXλ_FwdWrapper

DEFAULT_λ_CONFIG = dict(
    dim_k = 16,
    # n = 64, #32, # sample_size
    # r = 15, # if use local # not work with cross if different length
    heads = 4,
    dim_out = None,
    dim_u = 1
)






