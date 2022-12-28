import functools

import math

from einops import rearrange, parse_shape
from einops.layers.torch import Rearrange

import torch
from torch import nn
import torch.nn.functional as F

from torch import Tensor

from einops.layers.torch import Rearrange


def default_norm(num_groups, in_channels):
    return nn.GroupNorm(num_groups, in_channels)


def default_act():
    return nn.SiLU()


def norm_act_drop_conv(in_channels, out_channels, num_groups, p):
    norm = default_norm(num_groups, in_channels)
    act = default_act()
    drop = nn.Dropout2d(p)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    if p > 0:
        return nn.Sequential(norm, act, drop, conv)
    else:
        return nn.Sequential(norm, act, conv)


class Attention(nn.Module):
    def __init__(self, dim, num_groups):
        super().__init__()
        self.norm = default_norm(num_groups, dim)

        self.scale = dim**-0.5
        self.qkv_proj = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward_attention(self, x):
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, "b c h w -> b (h w) c")
        query, key, value = qkv.chunk(3, dim=2)
        key = rearrange(key, "b hw c -> b c hw") * self.scale
        score = torch.bmm(query, key)
        attention = F.softmax(score, dim=2)
        out = torch.bmm(attention, value)
        out = rearrange(out, "b (h w) c -> b c h w", **parse_shape(x, "b c h w"))
        return self.proj(out)

    def forward(self, x):
        h = self.norm(x)
        h = self.forward_attention(h)
        return h + x


def pairs(channels):
    return zip(channels[:-1], channels[1:])


class ResBlock(nn.Module):
    def __init__(
        self, c_in, c_out, with_attention=False, emb_dim=512, num_groups=32, p=0.1
    ) -> None:
        super().__init__()

        self.conv1 = norm_act_drop_conv(c_in, c_out, num_groups, p=0.0)

        self.condition = nn.Sequential(
            nn.Linear(emb_dim, c_out),
            Rearrange("b c -> b c 1 1"),
        )

        self.conv2 = norm_act_drop_conv(c_out, c_out, num_groups, p)

        if c_in != c_out:
            self.residual = nn.Conv2d(c_in, c_out, kernel_size=1)
        else:
            self.residual = nn.Identity()

        if with_attention:
            self.attention = Attention(c_out, num_groups)
        else:
            self.attention = nn.Identity()

    def forward(self, x, c):
        h = self.conv1(x)
        h += self.condition(c)
        h = self.conv2(h)
        h += self.residual(x)
        h = self.attention(h)
        return h


def DownSample(c_in, c_out):
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)


class UpSample(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2.0)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        pos_dim=128,
        emb_dim=512,
        num_groups=32,
        dropout=0.1,
        channels_per_depth=(128, 256, 256, 256),
        num_blocks=2,
        attention_depths=(2,),
    ):
        super().__init__()

        # configure channels, downsample_layers
        input_dim = channels_per_depth[0]
        channels = [input_dim]
        for c in channels_per_depth:
            channels += [c] * num_blocks

        max_depth = len(channels_per_depth)
        downsample_layers = [num_blocks * i for i in range(1, max_depth)]
        self.condition = nn.Sequential(
            SinusoidalPositionEmbeddings(pos_dim),
            nn.Linear(pos_dim, emb_dim),
            default_act(),
            nn.Linear(emb_dim, emb_dim),
            default_act(),
        )

        self.input_conv = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, stride=1, padding=1
        )

        default_resblock = functools.partial(
            ResBlock, emb_dim=emb_dim, num_groups=num_groups, p=dropout
        )

        down_layers = []

        depth = 1
        for i, (c_in, c_out) in enumerate(pairs(channels)):
            layer_num = i + 1

            down_layers += [default_resblock(c_in, c_out, depth in attention_depths)]

            if layer_num in downsample_layers:
                down_layers += [DownSample(c_out, c_out)]

                depth += 1

        depth = max_depth
        # if last down_layer is DownSample
        if down_layers[-1] == len(channels) - 1:
            up_layers = [UpSample(channels[-1], channels[-1])]
            depth -= 1
        else:
            up_layers = []

        for i, (c_in, c_out) in enumerate(pairs(channels[::-1])):
            with_attention = depth in attention_depths
            layer_num = len(channels) - 1 - i

            up_layers += [default_resblock(2 * c_in, c_out, with_attention)]

            if (layer_num - 1) in downsample_layers:
                up_layers += [
                    default_resblock(2 * c_out, c_out, with_attention),
                    UpSample(c_out, c_out),
                ]

                depth -= 1

        up_layers += [
            default_resblock(2 * channels[0], channels[0], 1 in attention_depths)
        ]

        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)

        c_out = channels[-1]
        self.middle_layers = nn.ModuleList(
            [
                default_resblock(c_out, c_out, with_attention=True),
                default_resblock(c_out, c_out, with_attention=False),
            ]
        )

        self.output_conv = norm_act_drop_conv(
            channels[0], in_channels, num_groups, p=0.0
        )

    def forward(self, x, c):
        t = self.condition(c)

        x = self.input_conv(x)

        outputs = [x]

        for f in self.down_layers:
            if isinstance(f, ResBlock):
                x = f(x, t)
            else:
                x = f(x)

            outputs.append(x)

        for f in self.middle_layers:
            x = f(x, t)

        for f in self.up_layers:
            if isinstance(f, ResBlock):
                x = torch.cat([x, outputs.pop()], dim=1)
                x = f(x, t)
            else:
                x = f(x)

        x = self.output_conv(x)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
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
