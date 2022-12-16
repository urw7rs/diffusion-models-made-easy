from typing import OrderedDict

import copy
import math

import torch
from torch import nn

import einops


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        dim=128,
        pos_dim=128,
        emb_dim=512,
        multipliers=[1, 2, 2, 2],
        attn_depth=(2,),
        groups=32,
        dropout=0.1,
    ):
        super().__init__()

        self.depth = len(multipliers)

        channels = [dim]
        for mult in multipliers:
            channels.append(dim * mult)

        output_dims = channels[1:]
        input_dims = channels[:-1]

        middle_dim = output_dims[-1]

        self.time_emb_mlp = TimeStepEmbedding(pos_dim=pos_dim, emb_dim=emb_dim)

        self.first_conv = conv2d(
            in_channels, dim, 3, 1, nn.GroupNorm(groups, dim), nn.SiLU()
        )
        self.final_conv = nn.Conv2d(dim, in_channels, 3, 1, 1)

        contract_layers = []
        expand_layers = []
        for i, (c_in, c_out) in enumerate(zip(input_dims, output_dims)):

            attention = i + 1 == attn_depth

            contract_layers.append(
                Block(
                    c_in,
                    c_out,
                    emb_dim,
                    groups,
                    dropout,
                    num_blocks=3,
                    add_attention=attention,
                ),
            )
            expand_layers.append(
                Block(
                    2 * c_out,
                    c_in,
                    emb_dim,
                    groups,
                    dropout,
                    num_blocks=3,
                    add_attention=attention,
                ),
            )
        expand_layers.reverse()

        self.contracting_path = nn.ModuleList(contract_layers)
        self.expansive_path = nn.ModuleList(expand_layers)

        self.downsamples = nn.ModuleList([DownSample(c) for c in output_dims])
        self.upsamples = nn.ModuleList([UpSample(c, 2) for c in output_dims[::-1]])

        self.middle = Block(
            middle_dim,
            middle_dim,
            emb_dim,
            groups,
            dropout,
            num_blocks=2,
            add_attention=True,
        )

    def forward(self, x, t):
        t = self.time_emb_mlp(t)

        x = self.first_conv(x)

        x_copies = []
        for i in range(self.depth):
            x = self.contracting_path[i](x, t)
            x_copies.append(x)
            x = self.downsamples[i](x)

        x = self.middle(x, t)

        for i in range(self.depth):
            x = self.upsamples[i](x)
            copied_x = x_copies.pop()
            x = torch.cat([x, copied_x], dim=1)
            x = self.expansive_path[i](x, t)

        x = self.final_conv(x)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeStepEmbedding(nn.Module):
    def __init__(self, pos_dim=64, emb_dim=256):
        super().__init__()

        self.position_embedding = SinusoidalPositionEmbeddings(pos_dim)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("linear0", nn.Linear(pos_dim, emb_dim)),
                    ("act0", nn.SiLU()),
                    ("linear1", nn.Linear(emb_dim, emb_dim)),
                ]
            )
        )

    def forward(self, t):
        h = self.position_embedding(t)
        h = self.mlp(h)
        return h


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_dim,
        groups=8,
        dropout=0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.groups = groups

        self._first = conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self._first_norm_then_act = nn.Sequential(
            OrderedDict([("norm", self.norm), ("act", self.act)])
        )

        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

        self._dropout = nn.Dropout2d(dropout)

        self._second = conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self._second_norm_then_act = nn.Sequential(
            OrderedDict([("norm", self.norm), ("act", self.act)])
        )

        if in_channels != out_channels:
            self._res_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        else:
            self._res_conv = None

        self._act = self.act

    def forward(self, x, t):
        h = self._first(x)
        h = h + einops.rearrange(self.mlp(t), "b c -> b c 1 1")
        h = self._first_norm_then_act(h)
        h = self._dropout(h)
        h = self._second(h)
        if self._res_conv is not None:
            x = self._res_conv(x)
        return self._second_norm_then_act(h + x)

    @property
    def norm(self):
        return nn.GroupNorm(self.groups, self.out_channels)

    @property
    def act(self):
        return nn.SiLU()


class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, dim, scale_factor):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        h = self.upsample(x)
        h = self.conv(h)
        return h


class Attention(nn.Module):
    def __init__(self, dim, groups=8, heads=1):
        super().__init__()

        self.norm = nn.GroupNorm(groups, dim)

        dim_head = dim // heads

        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv,
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = einops.rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out) + x


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_dim,
        groups,
        dropout,
        num_blocks=2,
        add_attention=False,
    ):
        super().__init__()

        self.num_blocks = num_blocks
        self.use_attention = add_attention

        resblocks = []
        attentions = []
        for i in range(num_blocks):
            if i == 0:
                c_in = in_channels
                c_out = out_channels
            else:
                c_in = c_out = out_channels

            resblocks.append(ResBlock(c_in, c_out, emb_dim, groups, dropout))
            if add_attention:
                attentions.append(Attention(c_out, groups))

        self.resblocks = nn.ModuleList(resblocks)
        self.attentions = nn.ModuleList(attentions)

    def forward(self, x, t):
        for i in range(self.num_blocks - 1):
            x = self.resblocks[i](x, t)
            if self.use_attention:
                x = self.attentions[i](x)
        x = self.resblocks[-1](x, t)
        return x


def conv2d(
    in_channels,
    out_channels,
    kernel_size,
    padding,
    norm=None,
    act=None,
):
    layers = OrderedDict()
    layers["conv"] = nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size, padding=padding
    )

    if norm is not None:
        layers["norm"] = copy.deepcopy(norm)

    if act is not None:
        layers["act"] = copy.deepcopy(act)

    return nn.Sequential(layers)
