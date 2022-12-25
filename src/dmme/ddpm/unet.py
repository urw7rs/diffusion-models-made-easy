import torch
from torch import nn

from .layers import (
    SinusoidalPositionEmbeddings,
    ResBlock,
    Attention,
    PreNorm,
    Attach,
    conv3x3,
)


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

    def __init__(self, in_channels=3, groups=32, drop_rate=0.1):
        super().__init__()

        emb_dim = 512

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, 128, 3, 1, 1)
        self.output_conv = conv3x3(128, in_channels, groups, drop_rate=0.0)

        def resblock(c_in, c_out):
            return ResBlock(
                c_in,
                c_out,
                emb_dim=emb_dim,
                groups=groups,
                drop_rate=drop_rate,
            )

        def attention(dim):
            return PreNorm(
                nn.GroupNorm(groups, dim),
                Attention(dim),
            )

        def downsample(dim):
            return nn.Conv2d(dim, dim, 3, 2, 1)

        def upsample(dim):
            return nn.Sequential(
                nn.Upsample(scale_factor=2.0),
                nn.Conv2d(dim, dim, 3, 1, 1),
            )

        self.down = nn.ModuleList(
            [
                # 32 x 32
                resblock(128, 256),
                resblock(256, 256),
                downsample(256),
                # 16 x 16
                Attach(resblock(256, 256), attention(256)),
                Attach(resblock(256, 256), attention(256)),
                downsample(256),
                # 8 x 8
                resblock(256, 256),
                resblock(256, 256),
                downsample(256),
                # 4 x 4
                resblock(256, 256),
                resblock(256, 256),
            ]
        )

        self.middle = nn.ModuleList(
            [
                # middle
                Attach(resblock(256, 256), attention(256)),
                resblock(256, 256),
            ]
        )

        self.up = nn.ModuleList(
            [
                # 4 x 4
                resblock(2 * 256, 256),
                resblock(2 * 256, 256),
                Attach(
                    resblock(2 * 256, 256),
                    upsample(256),
                ),
                # 8 x 8
                resblock(2 * 256, 256),
                resblock(2 * 256, 256),
                Attach(resblock(2 * 256, 256), upsample(256)),
                # 16 x 16
                Attach(resblock(2 * 256, 256), attention(256)),
                Attach(resblock(2 * 256, 256), attention(256)),
                Attach(
                    resblock(2 * 256, 256),
                    nn.Sequential(
                        attention(256),
                        upsample(256),
                    ),
                ),
                # 32 x 32
                resblock(2 * 256, 256),
                resblock(2 * 256, 128),
                resblock(2 * 128, 128),
            ]
        )

    def forward(self, x, t):
        r"""Using timestep embeddings, predict noise to denoise :math:`x_t` from :math:`x_t` and :math:`t` using a UNet

        Args:
            x (torch.Tensor): :math:`x_t`, tensor of shape :math:`(N, C, H, W)`
            t (torch.Tensor): :math:`t`, tensor of shape :math:`(N,)`

        Returns:
            (torch.Tensor): :math:`\epsilon_\theta(x_t,t)` predicted noise from image, a tensor of shape :math:`(N, C, H, W)`
        """

        t = self.time_emb(t)

        x = self.input_conv(x)
        outputs = [x]

        down = iter(self.down)
        for i in range(3):
            x = next(down)(x, t)
            outputs.append(x)
            x = next(down)(x, t)
            outputs.append(x)

            x = next(down)(x)
            outputs.append(x)

        x = next(down)(x, t)
        outputs.append(x)
        x = next(down)(x, t)
        outputs.append(x)

        x = self.middle[0](x, t)
        x = self.middle[1](x, t)

        for i in range(12):
            x = self.up[i](torch.cat([x, outputs.pop()], dim=1), t)

        x = self.output_conv(x)
        return x
