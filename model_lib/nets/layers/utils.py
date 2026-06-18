import torch
from torch import einsum
from einops import rearrange#, einsum

import torch._dynamo
torch._dynamo.config.suppress_errors = True

_exists = lambda val: val is not None
_default = lambda val, d: val if _exists(val) else d

_einsum = lambda *args, **kwargs: einsum(*args, **kwargs).contiguous()
_rearrange = lambda *args, **kwargs: rearrange(*args, **kwargs).contiguous()

from torch.nn.modules.utils import _ntuple, _single, _pair, _triple, _quadruple

to_2tuple = _pair
to_3tuple = _triple
to_4tuple = _quadruple

def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = _rearrange(torch.stack(pos), 'n i j -> (i j) n')
    rel_pos = pos[None, :] - pos[:, None]
    rel_pos += n - 1
    return rel_pos


