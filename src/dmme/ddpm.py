from typing import Tuple, Optional

import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import einops

import pytorch_lightning as pl

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from torch.optim import Adam
from dmme.lr_scheduler import WarmupLR
from dmme.callbacks import EMA

from dmme.common import denorm, gaussian_like, uniform_int


class LitDDPM(pl.LightningModule):
    r"""LightningModule for training DDPM

    Args:
        model (nn.Module): neural network predicting noise :math:`\epsilon_\theta`
        lr (float): learning rate, defaults to :math:`2e-4`
        warmup (int): linearly increases learning rate for
            `warmup` steps until `lr` is reached, defaults to 5000
        imgsize (Tuple[int, int, int]): image size in `(C, H, W)`
        timestpes (int): total timesteps for the
            forward and reverse process, :math:`T`
        decay (float): EMA decay value
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-4,
        warmup: int = 5000,
        imgsize: Tuple[int, int, int] = (3, 32, 32),
        timesteps: int = 1000,
        decay: float = 0.9999,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        if model is None:
            model = UNet()

        self.model = model

        self.process = DDPM(timesteps=timesteps)

        self.fid = FrechetInceptionDistance(
            normalize=True,
            reset_real_features=False,
        )

        self.inception = InceptionScore(normalize=True)

    def forward(self, x_t: Tensor, t: int, noise: Optional[Tensor] = None):
        r"""Denoise image once using `DDPM`

        Args:
            x_t (torch.Tensor): image of shape :math:`(N, C, H, W)`
            t (int): starting :math:`t` to sample from
            noise (torch.Tensor): noise to use for sampling, if `None` samples new noise

        Returns:
            (torch.Tensor): generated sample of shape :math:`(N, C, H, W)`
        """

        if noise is None:
            noise = gaussian_like(x_t)

        timestep = torch.tensor([t], device=x_t.device)

        x_t = self.process.sample(self.model, x_t, timestep, noise)

        return x_t

    def training_step(self, batch, batch_idx):
        r"""Train model using :math:`L_\text{simple}`"""

        x_0: Tensor = batch[0]

        batch_size: int = x_0.size(0)
        t: Tensor = uniform_int(
            0, self.hparams.timesteps, batch_size, device=x_0.device
        )

        noise: Tensor = gaussian_like(x_0)

        x_t: Tensor = self.process.forward_process(x_0, t, noise)

        noise_estimate: Tensor = self.model(x_t, t)

        loss: Tensor = F.mse_loss(noise, noise_estimate)
        self.log("train/loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        """Generate samples for evaluation"""

        x: Tensor = batch[0]

        self.fid.update(denorm(x), real=True)

        x_t: Tensor = gaussian_like(x)

        x_t = self.generate(x_t)

        fake_x: Tensor = denorm(x_t)

        self.fid.update(fake_x, real=False)
        self.inception.update(fake_x)

    def generate(self, x_t):
        r"""Iteratively sample from :math:`p_\theta(x_{t-1}|x_t)` to generate images

        Args:
            x_t (torch.Tensor): :math:`x_T` to start from
        """

        noise = [None]
        for _ in range(self.hparams.timesteps, 0, -1):
            noise.append(gaussian_like(x_t))

        for t in tqdm(range(self.hparams.timesteps, 0, -1), leave=False):
            x_t = self(x_t, t, noise[t])

        return x_t

    def test_epoch_end(self, outputs):
        """Compute metrics and log at the end of the epoch"""

        fid_score: Tensor = self.fid.compute()
        kl_mean, kl_std = self.inception.compute()
        inception_score = torch.exp(kl_mean)

        self.log("fid", fid_score)
        self.log("inception_score", inception_score)

    def configure_optimizers(self):
        """Configure optimizers for training Uses Adam and warmup lr"""

        optimizer = Adam(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = WarmupLR(optimizer, self.hparams.warmup)

        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def configure_callbacks(self):
        """Configure EMA callback, will override any other EMA callback"""
        ema_callback = EMA(decay=self.hparams.decay)
        return ema_callback


class DDPM(nn.Module):
    """Forward, Reverse, Sampling for DDPM

    Args:
        timesteps (int): total timesteps :math:`T`
    """

    beta: Tensor
    alpha: Tensor
    alpha_bar: Tensor
    sigma: Tensor

    def __init__(self, timesteps) -> None:
        super().__init__()

        beta = linear_schedule(timesteps)
        beta = einops.rearrange(beta, "t -> t 1 1 1")

        alpha = 1 - beta
        # alpha[0] = 1 so no problems here
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta, persistent=False)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)
        self.register_buffer("sigma", torch.sqrt(beta), persistent=False)

    def forward_process(self, x_0: Tensor, t: Tensor, noise: Tensor):
        r"""Forward Diffusion Process

        Samples :math:`x_t` from :math:`q(x_t|x_0)
        = \mathcal{N}(x_t;\sqrt{\bar\alpha_t}\bold{x}_0,(1-\bar\alpha_t)\bold{I})`

        Computes :math:`\bold{x}_t
        = \sqrt{\bar\alpha_t}\bold{x}_0 + \sqrt{1-\bar\alpha_t}\bold{I}`

        Args:
            x_0 (torch.Tensor): data to add noise to
            t (int): :math:`t` in :math:`x_t`
            noise (torch.Tensor, optional):
                :math:`\epsilon`, noise used in the forward process

        Returns:
            (torch.Tensor): :math:`\bold{x}_t \sim q(\bold{x}_t|\bold{x}_0)`
        """

        alpha_bar_t = self.alpha_bar[t]

        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        return x_t

    def reverse_process(self, model, x_t, t, noise):
        r"""Reverse Denoising Process

        Samples :math:`x_{t-1}` from
        :math:`p_\theta(\bold{x}_{t-1}|\bold{x}_t)
        = \mathcal{N}(\bold{x}_{t-1};\mu_\theta(\bold{x}_t, t), \sigma_t\bold{I})`

        .. math::
            \begin{aligned}
            \bold\mu_\theta(\bold{x}_t, t)
            &= \frac{1}{\sqrt{\alpha_t}}\bigg(\bold{x}_t
            -\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\bold{x}_t,t)\bigg) \\
            \sigma_t &= \beta_t
            \end{aligned}

        Computes :math:`\bold{x}_{t-1}
        = \frac{1}{\sqrt{\alpha_t}}\bigg(\bold{x}_t
        -\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\bold{x}_t,t)\bigg)
        +\sigma_t\epsilon`

        Args:
            model (nn.Module): model for estimating noise
            x_t (torch.Tensor): x_t
            t (int): current timestep
            noise (torch.Tensor): noise
        """

        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        sigma_t = self.sigma[t]

        noise_estimate = model(x_t, t)

        x_t_minus_one = (
            1
            / torch.sqrt(alpha_t)
            * (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_estimate)
            + sigma_t * noise
        )

        return x_t_minus_one

    def sample(self, model, x_t, t, noise):
        r"""Sample from :math:`p_\theta(x_{t-1}|x_t)`

        Args:
            model (nn.Module): model for estimating noise
            x_t (torch.Tensor): image of shape :math:`(N, C, H, W)`
            t (int): starting :math:`t` to sample from
            noise (torch.Tensor): noise to use for sampling, if `None` samples new noise

        Returns:
            (torch.Tensor): generated sample of shape :math:`(N, C, H, W)`
        """

        (idx,) = torch.where(t == 1)
        noise[idx] = 0

        x_t = self.reverse_process(model, x_t, t, noise)
        return x_t


def pad(x: Tensor, value: float = 0) -> Tensor:
    r"""pads tensor with 0 to match :math:`t` with tensor index"""

    ones = torch.ones_like(x[0:1])
    return torch.cat([ones * value, x], dim=0)


def linear_schedule(timesteps: int, start=0.0001, end=0.02) -> Tensor:
    r"""constants increasing linearly from :math:`10^{-4}` to :math:`0.02`

    Args:
        timesteps (int): total timesteps
        start (float): starting value, defaults to 0.0001
        end (float): end value, defaults to 0.02
    """

    beta = torch.linspace(start, end, timesteps)
    return pad(beta)


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
        in_channels=3,
        pos_dim=128,
        emb_dim=512,
        num_blocks=2,
        channels=(128, 256, 256, 256),
        attn_depth=(2,),
        groups=32,
        drop_rate=0.1,
    ):
        super().__init__()

        self.depth = len(channels) - 1
        self.num_blocks = num_blocks

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(pos_dim),
            nn.Linear(pos_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.pos_dim = pos_dim

        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, 1, 1)
        self.output_conv = conv3x3(channels[0], in_channels, groups, drop_rate=0.0)

        output_dims = channels[1:]
        input_dims = channels[:-1]

        down_blocks = []
        down = []

        for i, (c_in, c_out) in enumerate(zip(input_dims, output_dims)):
            attention = i + 1 == attn_depth

            layers = []
            layers.append(ResBlock(c_in, c_out, emb_dim, groups, drop_rate, attention))

            for _ in range(num_blocks - 1):
                layers.append(
                    ResBlock(c_out, c_out, emb_dim, groups, drop_rate, attention)
                )

            down_blocks.extend(layers)
            if i != self.depth - 1:
                down.append(nn.Conv2d(c_out, c_out, 3, 2, 1))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.down = nn.ModuleList(down)

        dim = dim = channels[-1]
        self.middle = nn.ModuleList(
            [
                ResBlock(dim, dim, emb_dim, groups, drop_rate, attention=True),
                ResBlock(dim, dim, emb_dim, groups, drop_rate, attention=False),
            ]
        )

        up_blocks = []
        up = []

        for i, (c_in, c_out) in enumerate(zip(input_dims[::-1], output_dims[::-1])):
            attention = i + 1 == attn_depth

            layers = []
            for _ in range(num_blocks - 1):
                layers.append(
                    ResBlock(2 * c_out, c_out, emb_dim, groups, drop_rate, attention)
                )
            layers.append(
                ResBlock(2 * c_out, c_in, emb_dim, groups, drop_rate, attention)
            )
            layers.append(
                ResBlock(2 * c_in, c_in, emb_dim, groups, drop_rate, attention)
            )

            up_blocks.extend(layers)
            if i != self.depth - 1:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2.0),
                    nn.Conv2d(c_in, c_in, 3, 1, 1),
                )
                up.append(upsample)

        self.up_blocks = nn.ModuleList(up_blocks)
        self.up = nn.ModuleList(up)

    def forward(self, x, t):
        r"""Using timestep embeddings, predict noise to denoise :math:`x_t` from :math:`x_t` and :math:`t` using a UNet

        Args:
            x (torch.Tensor): :math:`x_t`, tensor of shape :math:`(N, C, H, W)`
            t (torch.Tensor): :math:`t`, tensor of shape :math:`(N,)`

        Returns:
            (torch.Tensor): :math:`\epsilon_\theta(x_t,t)` predicted noise from image, a tensor of shape :math:`(N, C, H, W)`
        """

        t = self.time_emb(t)

        x_copies = []

        x = self.input_conv(x)
        x_copies.append(x)

        for i in range(self.depth):
            for j in range(self.num_blocks):
                x = self.down_blocks[self.num_blocks * i + j](x, t)
                x_copies.append(x)

            if i != self.depth - 1:
                x = self.down[i](x)
                x_copies.append(x)

        for i in range(self.num_blocks):
            x = self.middle[i](x, t)

        for i in range(self.depth):
            for j in range(self.num_blocks + 1):
                x = torch.cat([x, x_copies.pop()], dim=1)
                x = self.up_blocks[(self.num_blocks + 1) * i + j](x, t)

            if i != self.depth - 1:
                x = self.up[i](x)

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

    def __init__(
        self, in_channels, out_channels, emb_dim, groups, drop_rate, attention=False
    ):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, groups, drop_rate)
        self.conv2 = conv3x3(out_channels, out_channels, groups, drop_rate)
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        else:
            self.conv3 = None

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

        self.attention = (
            PreNorm(nn.GroupNorm(groups, out_channels), Attention(out_channels))
            if attention
            else nn.Identity()
        )

    def forward_resblock(self, x, t):
        h = self.conv1(x)
        h += einops.rearrange(self.time_mlp(t), "n c -> n c 1 1")
        h = self.conv2(h)
        if self.conv3 is not None:
            x = self.conv3(x)
        return h + x

    def forward(self, x, t):
        x = self.forward_resblock(x, t)
        x = self.attention(x)
        return x


def conv3x3(in_channels, out_channels, groups, drop_rate):
    """Build 3x3 convolution with normalization and dropout in norm act drop conv order

    Args:
        in_channels (int): passed to `nn.Conv2d`
        out_channels (int): passed to `nn.Conv2d`
        groups (int): passed to `nn.GroupNorm`
        drop_rate (float): passed to `nn.Dropout2d`
    """

    return nn.Sequential(
        nn.GroupNorm(groups, in_channels),
        nn.SiLU(),
        nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity(),
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
    )
