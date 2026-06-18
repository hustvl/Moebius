# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma
'''
modified from https://github.com/NVlabs/Sana/blob/main/diffusion/model/nets/basic_modules.py#L81
'''

import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from .act import build_act, get_act_name
from .norms import build_norm, get_norm_name
from .utils import get_same_padding, val2tuple


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: int or None = None,  # type: ignore
        use_bias=False,
        dropout=0.0,
        norm="bn2d",
        act="relu",
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_dim)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):  
    # Mix-FFN in SANA: http://arxiv.org/abs/2410.10629
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding: int or None = None,  # type: ignore
        use_bias=(True, True, False), #False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
    ):
        out_feature = out_feature or in_features
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.glu_act = build_act(act[1], inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_feature,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )
        # from IPython import embed; embed(header='debug dilate conv')

    def forward(self, x: torch.Tensor, HW=None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x


class SlimGLUMBConv(GLUMBConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 移除 self.inverted_conv 层
        del self.inverted_conv
        self.out_dim = self.point_conv.out_dim

    def forward(self, x: torch.Tensor, HW=None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.reshape(B, self.out_dim, N).permute(0, 2, 1)

        return x


class MBConvPreGLU(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        mid_dim=None,
        expand=6,
        padding: int or None = None,  # type: ignore
        use_bias=False,
        norm=(None, None, "ln2d"),
        act=("silu", "silu", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        mid_dim = mid_dim or round(in_dim * expand)

        self.inverted_conv = ConvLayer(
            in_dim,
            mid_dim * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=None,
        )
        self.glu_act = build_act(act[0], inplace=False)
        self.depth_conv = ConvLayer(
            mid_dim,
            mid_dim,
            kernel_size,
            stride=stride,
            groups=mid_dim,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=act[1],
        )
        self.point_conv = ConvLayer(
            mid_dim,
            out_dim,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )

    def forward(self, x: torch.Tensor, HW=None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x = self.inverted_conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.depth_conv(x)
        x = self.point_conv(x)

        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x

    @property
    def module_str(self) -> str:
        _str = f"{self.depth_conv.kernel_size}{type(self).__name__}("
        _str += f"in={self.inverted_conv.in_dim},mid={self.depth_conv.in_dim},out={self.point_conv.out_dim},s={self.depth_conv.stride}"
        _str += (
            f",norm={get_norm_name(self.inverted_conv.norm)}"
            f"+{get_norm_name(self.depth_conv.norm)}"
            f"+{get_norm_name(self.point_conv.norm)}"
        )
        _str += (
            f",act={get_act_name(self.inverted_conv.act)}"
            f"+{get_act_name(self.depth_conv.act)}"
            f"+{get_act_name(self.point_conv.act)}"
        )
        _str += f",glu_act={get_act_name(self.glu_act)})"
        return _str


class DWMlp(Mlp):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=None,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            bias=bias,
            drop=drop,
        )
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.conv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=hidden_features,
            bias=bias,
        )

    def forward(self, x, HW=None):
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = x.reshape(B, H, W, self.hidden_features).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(B, self.hidden_features, N).permute(0, 2, 1)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Mlp(Mlp):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            bias=bias,
            drop=drop,
        )

    def forward(self, x, HW=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


if __name__ == "__main__":
    model = GLUMBConv(
        1152,
        1152 * 4,
        1152,
        use_bias=(True, True, False),
        norm=(None, None, None),
        act=("silu", "silu", None),
    ).cuda()
    input = torch.randn(4, 256, 1152).cuda()
    output = model(input)