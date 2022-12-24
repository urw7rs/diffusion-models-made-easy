from typing import Tuple, Optional, Union, List

from tqdm import tqdm

import torch
from torch import Tensor, nn
from torch.optim import Adam

from einops import rearrange

import pytorch_lightning as pl

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from dmme.lr_scheduler import WarmupLR
from dmme.callbacks import EMA

from dmme.common import denorm, gaussian_like, uniform_int

from .ddpm import DDPMSampler, ForwardProcess, linear_schedule
from .loss import SimpleLoss
from .unet import UNet


class LitDDPM(pl.LightningModule):
    """LightningModule for training DDPM

    Args:
        sampler (nn.Module): an instance of `DDPMSampler`
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

        beta = linear_schedule(timesteps)
        beta = rearrange(beta, "t -> t 1 1 1")
        self.forward_process = ForwardProcess(beta)
        self.sampler = DDPMSampler(timesteps=timesteps)

        self.loss = SimpleLoss()

        self.fid = FrechetInceptionDistance(
            normalize=True,
            reset_real_features=False,
        )

        self.inception = InceptionScore(normalize=True)

    def forward(self, x_t: Tensor, t: int, noise: Optional[Tensor] = None):
        r"""Iteratively sample from :math:`p_\theta(x_{t-1}|x_t)` starting with :math:`x_t` with start, stop step specified from arguments

        Args:
            x_t (torch.Tensor): image of shape :math:`(N, C, H, W)`
            start_t (int): starting :math:`t` to sample from
            stop_t (int): stops sampling when reached
            steps_t (int): step sizes for sequence
            noise (torch.Tensor): noise to use for sampling, if `None` samples new noise

        Returns:
            (torch.Tensor): generated samples
        """

        if noise is None:
            noise = gaussian_like(x_t)

        timestep = torch.tensor([t], device=x_t.device)

        x_t = self.sampler(self.model, x_t, timestep, noise)

        return x_t

    def training_step(self, batch, batch_idx):
        """Compute loss using sampler"""
        x_0: Tensor = batch[0]

        batch_size: int = x_0.size(0)
        t: Tensor = uniform_int(
            0, self.hparams.timesteps, batch_size, device=x_0.device
        )

        noise: Tensor = gaussian_like(x_0)

        x_t: Tensor = self.forward_process(x_0, t, noise)

        noise_estimate: Tensor = self.model(x_t, t)

        loss: Tensor = self.loss(noise, noise_estimate)
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

    def setup(self, stage: str):
        """Prepare metrics for test stage"""

    def configure_callbacks(self):
        """Configure EMA callback, will override any other EMA callback"""

        ema_callback = EMA(decay=self.hparams.decay)

        return ema_callback
