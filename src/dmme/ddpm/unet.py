import functools

import math

from einops import rearrange, parse_shape
from einops.layers.torch import Rearrange

import torch
from torch import is_deterministic_algorithms_warn_only_enabled, nn
import torch.nn.functional as F

from torch import Tensor

from einops.layers.torch import Rearrange


def default_act():
    return nn.SiLU()


def norm_act_drop_conv(in_channels, out_channels, num_groups, p):
    norm = nn.GroupNorm(num_groups, in_channels)
    act = default_act()
    drop = nn.Dropout2d(p)
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    if p > 0:
        return nn.Sequential(norm, act, drop, conv)
    else:
        return nn.Sequential(norm, act, conv)


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
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b c h w -> b c (h w)")
        query, key, value = qkv.chunk(3, dim=1)
        query = rearrange(query, "b c hw -> b hw c") * self.scale
        value = rearrange(value, "b c hw -> b hw c") * self.scale
        score = torch.bmm(query, key)
        attention = F.softmax(score, dim=2)
        out = torch.bmm(attention, value)
        out = rearrange(out, "b (h w) c -> b c h w", **parse_shape(x, "b c h w"))
        return self.to_out(out)


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
            self.residual = nn.Conv2d(c_in, c_out, 3, 1, 1)
        else:
            self.residual = nn.Identity()

        if with_attention:
            self.attention = Attention(c_out)
        else:
            self.attention = nn.Identity()

    def forward(self, x, c):
        h = self.conv1(x)
        h += self.condition(c)
        h = self.conv2(h)
        h = self.attention(h)
        return h + self.residual(x)


def DownSample(c_in, c_out):
    return nn.Conv2d(c_in, c_out, 3, 2, 1)


class UpSample(nn.Module):
    def __init__(self, c_in, c_out) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2.0)
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


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
        channels=(128, 256, 256, 256, 256, 256, 256, 256, 256),
        attention_depths=(2,),
        downsample_layers=(2, 4, 6),
    ):
        super().__init__()

        original_depth = 1
        new_depths = len(downsample_layers)
        max_depth = original_depth + new_depths

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
            ResBlock, emb_dim=emb_dim, num_groups=num_groups, p=p
        )

        down_layers = []
        up_layers = []

        up_layers += [
            default_resblock(2 * channels[0], channels[0], 1 in attention_depths)
        ]

        depth = 1
        for layer_num, (c_in, c_out) in enumerate(pairs(channels)):
            with_attention = depth in attention_depths
            downsample = layer_num in downsample_layers

            if downsample:
                down_layers += [DownSample(c_out, c_out)]
                up_layers += [UpSample(c_out, c_out)]

                up_layers += [default_resblock(2 * c_out, c_out, 1 in attention_depths)]

                depth += 1

            down_layers += [default_resblock(c_in, c_out, with_attention)]
            up_layers += [default_resblock(2 * c_out, c_in, with_attention)]

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

        for f in reversed(self.up_layers):
            if isinstance(f, ResBlock):
                x = torch.cat([x, outputs.pop()], dim=1)
                x = f(x, t)
            else:
                x = f(x)

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
