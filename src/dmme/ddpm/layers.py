import math

import torch
from torch import nn
import torch.nn.functional as F

import einops

from torch import Tensor


class Attach(nn.Module):
    def __init__(self, conditional, unconditional) -> None:
        super().__init__()

        self.cond = conditional
        self.uncond = unconditional

    def forward(self, x, t):
        x = self.cond(x, t)
        x = self.uncond(x)
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

        qkv = einops.rearrange(qkv, "b c h w -> b c (h w)")
        query, key, value = qkv.chunk(3, dim=1)

        score = einops.einsum(query * self.scale, key, "b c qhw, b c khw -> b qhw khw")

        attention = F.softmax(score, dim=-1)

        out = einops.einsum(attention, value, "b qhw khw, b c khw -> b c qhw")

        out = einops.rearrange(out, "b c (h w) -> b c h w", h=h, w=w)

        return self.to_out(out)


class PreNorm(nn.Module):
    """Pre Normalization with residual connections

    Args:
        norm_layer (nn.Module): normalization layer
        attention_layer (nn.Module): attention layer
    """

    def __init__(self, norm_layer, attention_layer) -> None:
        super().__init__()
        self.norm = norm_layer
        self.attention = attention_layer

    def forward(self, x):
        h = self.norm(x)
        h = self.attention(x)
        return h + x


class ResBlock(nn.Module):
    """BasicWideResBlock for UNet GroupNorm and optional self-attention

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        emb_dim (int): timestep embedding dim
        groups (int): num groups in `nn.GroupNorm`
        drop_rate (float): dropout applied in each conv
        attention (bool): flag for adding self-attention layer
    """

    def __init__(self, in_channels, out_channels, emb_dim, groups, drop_rate):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, groups, drop_rate)
        self.conv2 = conv3x3(out_channels, out_channels, groups, drop_rate)
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        else:
            self.conv3 = None

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x, t):
        h = self.conv1(x)
        h += einops.rearrange(self.time_mlp(t), "n c -> n c 1 1")
        h = self.conv2(h)
        if self.conv3 is not None:
            x = self.conv3(x)
        return h + x


def conv3x3(in_channels, out_channels, groups, drop_rate):
    """Build 3x3 convolution with normalization and dropout in norm act drop conv order

    Args:
        in_channels (int): passed to `nn.Conv2d`
        out_channels (int): passed to `nn.Conv2d`
        groups (int): passed to `nn.GroupNorm`
        drop_rate (float): passed to `nn.Dropout2d`
    """

    if drop_rate > 0:
        return nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )
    else:
        return nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )
