import math

from einops import rearrange, einsum
from einops.layers.torch import Rearrange

import torch
from torch import nn
import torch.nn.functional as F

from torch import Tensor

from einops.layers.torch import Rearrange

import dmme.nn as dnn


def default_norm(num_groups, num_channels):
    return nn.GroupNorm(num_groups, num_channels)


def default_act():
    return nn.SiLU()


def default_dropout(p):
    return nn.Dropout2d(p)


def conv_norm_act_drop(in_channels, out_channels, num_groups, p):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    norm = default_norm(num_groups, out_channels)
    act = default_act()
    drop = nn.Dropout2d(p)

    if p > 0:
        return nn.Sequential(conv, norm, act, drop)
    else:
        return nn.Sequential(conv, norm, act)


class Adapter(nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return self.module(x)


class Attention(nn.Module):
    r"""Self Attention layer

    Args:
        dim (int): :math:`d_\text{model}`
    """

    def __init__(self, dim):
        super().__init__()

        self.scale = dim**-0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        """Multi Head Self Attention on images with prenorm and residual connections

        Returns:
            x
        """
        h, w = x.size()[2:]

        qkv = self.to_qkv(x)

        qkv = rearrange(qkv, "b c h w -> b c (h w)")
        query, key, value = qkv.chunk(3, dim=1)

        score = einsum(query * self.scale, key, "b c qhw, b c khw -> b qhw khw")

        attention = F.softmax(score, dim=-1)

        out = einsum(attention, value, "b qhw khw, b c khw -> b c qhw")

        out = rearrange(out, "b c (h w) -> b c h w", h=h, w=w)

        return self.to_out(out)


def inout(channels):
    return zip(channels[:-1], channels[1:])


class UNet(nn.Module):
    """UNet with GroupNorm and Attention, Predicts noise from :math:`x_t` and :math:`t`

    Args:
        in_channels (int): input image channels
        pos_dim (int): sinusoidal position encoding dim
        emb_dim (int): time embedding mlp dim
        num_blocks (int): number of resblocks to use
        channels (Tuple[int...]): list of channel dimensions
        attn_depth (Tuple[int...]): depth where attention is applied
        groups (int): number of groups in `nn.GroupNorm`
        drop_rate (float): drop_rate in `ResBlock`
    """

    def __init__(
        self,
        in_channels,
        pos_dim=128,
        emb_dim=512,
        num_groups=32,
        p=0.1,
        channels=(128, 256, 256, 256, 256),
        attention_depth=(2,),
        downsample_depth=(1, 2, 3),
    ):
        super().__init__()

        depth = max(downsample_depth) + 1

        upsample_depth = tuple(depth - i + 1 for i in downsample_depth)

        self.condition = nn.Sequential(
            SinusoidalPositionEmbeddings(pos_dim),
            nn.Linear(pos_dim, emb_dim),
            default_act(),
            nn.Linear(emb_dim, emb_dim),
            default_act(),
        )
        self.input_conv = conv_norm_act_drop(3, channels[0], num_groups, p=0.0)

        def make_block(c_in, c_out):
            modules = [
                (nn.Conv2d(c_in, c_out, 3, 1, 1), "x -> h"),
                (
                    nn.Sequential(
                        nn.Linear(emb_dim, c_out),
                        Rearrange("b c -> b c 1 1"),
                    ),
                    "t -> t",
                ),
                (dnn.Add(), "h t -> h"),
                (
                    nn.Sequential(
                        default_norm(num_groups, c_out),
                        default_act(),
                        default_dropout(p),
                        conv_norm_act_drop(c_out, c_out, num_groups, p=0),
                    ),
                    "h -> h",
                ),
                (dnn.Add(), "x h -> x"),
            ]

            if c_in != c_out:
                modules.insert(
                    len(modules) - 1, (nn.Conv2d(c_in, c_out, 3, 1, 1), "x -> x")
                )

            return dnn.Sequential(*modules)

        def make_attention(dim):
            return dnn.Sequential(
                (
                    nn.Sequential(
                        default_norm(num_groups, dim),
                        Attention(dim),
                    ),
                    "x -> h",
                ),
                (dnn.Add(), "x h -> x"),
            )

        layers = []
        for current_depth, (c_in, c_out) in zip(range(1, depth + 1), inout(channels)):
            if current_depth in attention_depth:
                blocks = [
                    dnn.Sequential(
                        (make_block(c_in, c_out), "x t -> x"),
                        (make_attention(c_out), "x -> x"),
                    ),
                    dnn.Sequential(
                        (make_block(c_out, c_out), "x t -> x"),
                        (make_attention(c_out), "x -> x"),
                    ),
                ]
            else:
                blocks = [
                    make_block(c_in, c_out),
                    make_block(c_out, c_out),
                ]

            layers.extend(blocks)

            if current_depth in downsample_depth:
                layers.append(Adapter(nn.Conv2d(c_out, c_out, 3, 2, 1)))

        self.down_layers = nn.ModuleList(layers)

        c_out = channels[-1]
        self.middle_layers = nn.ModuleList(
            [
                dnn.Sequential(
                    (make_block(c_out, c_out), "x t -> x"),
                    (make_attention(c_out), "x -> x"),
                ),
                make_block(c_out, c_out),
            ]
        )

        layers = []
        for current_depth, (c_in, c_out) in zip(
            range(depth, 0, -1), inout(channels[::-1])
        ):
            if depth in attention_depth:
                blocks = [
                    dnn.Sequential(
                        (make_block(2 * c_in, c_in), "x t -> x"),
                        (make_attention(c_out), "x -> x"),
                    ),
                    dnn.Sequential(
                        (make_block(2 * c_in, c_out), "x t -> x"),
                        (make_attention(c_out), "x -> x"),
                    ),
                    dnn.Sequential(
                        (make_block(2 * c_out, c_out), "x t -> x"),
                        (make_attention(c_out)),
                        "x -> x",
                    ),
                ]
            else:
                blocks = [
                    make_block(2 * c_in, c_in),
                    make_block(2 * c_in, c_out),
                    make_block(2 * c_out, c_out),
                ]

            if current_depth in upsample_depth:
                block = blocks[-1]
                blocks[-1] = dnn.Sequential(
                    (block, "x t -> x"),
                    (
                        nn.Sequential(
                            nn.Upsample(scale_factor=2.0),
                            nn.Conv2d(c_in, c_in, 3, 1, 1),
                        ),
                        "x -> x",
                    ),
                )

            layers.extend(blocks)

        self.up_layers = nn.ModuleList(layers)

        self.output_conv = nn.Conv2d(channels[0], 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x, c):
        t = self.condition(c)

        x = self.input_conv(x)

        outputs = [x]

        for f in self.down_layers:
            x = f(x=x, t=t)
            outputs.append(x)

        for f in self.middle_layers:
            x = f(x=x, t=t)

        for f in self.up_layers:
            x = torch.cat([x, outputs.pop()], dim=1)
            x = f(x=x, t=t)

        x = self.output_conv(x)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    r"""Transformer Sinusoidal Position Encoding

    Args:
        dim (int): embedding dimension
    """

    embeddings: Tensor

    def __init__(self, dim) -> None:
        super().__init__()

        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = embeddings[None, :]

        self.register_buffer("embeddings", embeddings)

    def forward(self, t):
        embeddings = t[:, None] * self.embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
