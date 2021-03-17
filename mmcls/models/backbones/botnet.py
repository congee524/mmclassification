import copy
from einops import rearrange

import torch.einsum as einsum
import torch.nn as nn
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      constant_init, kaiming_init)
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


class BoTBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 fmap_shape,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 mhsa_cfg=dict(),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BoTBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fmap_shape = fmap_shape
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        assert stride in (1, 2)
        self.stride = stride
        self.dilation = dilation
        self.mhsa_cfg = copy.deepcopy(mhsa_cfg)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=1,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        self.mhsa = MHSA(self.mid_channels, self.fmap_shape, **self.mhsa_cfg)
        self.avg_pool = nn.AvgPool2d((2, 2)) if stride == 2 else nn.Identity()

        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.mhsa(out)
        out = self.avg_pool(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out


class MHSA(nn.Module):

    def __init__(self,
                 in_channels,
                 fmap_shape,
                 heads=4,
                 dim_qk=128,
                 dim_v=128,
                 rel_pos_emb=False):
        super().__init__()
        self.in_channels = in_channels
        self.fmap_shape = fmap_shape
        self.heads = heads
        self.scale = dim_qk**-0.5
        self.out_channels_qk = heads * dim_qk
        self.out_channels_v = heads * dim_v

        self.to_qk = nn.Conv2d(
            self.in_channels, self.out_channels_qk * 2, 1,
            bias=False)  # 1*1 conv to compute q, k
        self.to_v = nn.Conv2d(
            self.in_channels, self.out_channels_v, 1,
            bias=False)  # 1*1 conv to compute v
        self.softmax = nn.Softmax(dim=-1)

        height, width = self.fmap_shape
        if rel_pos_emb:
            self.pos_emb = RelPosEmb(height, width, dim_qk)
        else:
            self.pos_emb = AbsPosEmb(height, width, dim_qk)

    def forward(self, x):
        heads = self.heads
        n, c, h, w = x.shape
        q, k = self.to_qk(x).chunk(2, dim=1)
        v = self.to_v(x)
        q, k, v = map(
            lambda x: rearrange(x, "n (h d) h w -> n h (h w) d", h=heads),
            (q, k, v))

        q *= self.scale

        logits = einsum("n h x d, n h y d -> n h x y", q, k)
        logits += self.pos_emb(q)

        weights = self.softmax(logits)
        attn_out = einsum("n h x y, n h y d -> n h x d", weights, v)
        attn_out = rearrange(attn_out, "n h (h w) d -> n (h d) h w", h=h)

        return attn_out
