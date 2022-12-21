from typing import Tuple, Optional

import torch
from torch import nn
from torch.optim import Adam

import pytorch_lightning as pl

from dmme.ddpm.ddpm_sampler import DDPMSampler

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from dmme.common import denorm, set_default
from dmme.lr_scheduler import WarmupLR

from dmme.callbacks import EMA

from .ddpm_sampler import DDPMSampler
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
        sampler: Optional[nn.Module] = None,
        lr: float = 2e-4,
        warmup: int = 5000,
        imgsize: Tuple[int, int, int] = (3, 32, 32),
        timesteps: int = 1000,
        decay: float = 0.9999,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="sampler")

        self.sampler = set_default(
            sampler, DDPMSampler(UNet(in_channels=3), timesteps=timesteps)
        )

    def forward(self, x_t, start_t, stop_t=0, step_t=-1, noise=None):
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

        if start_t is None:
            start_t = self.sampler.timesteps

        if noise is None:
            num_steps = abs(stop_t - start_t) // abs(step_t) + 1
            noise = [None] * self.sampler.timesteps

        for t in range(start_t, stop_t, step_t):
            x_t = self.sampler.sample(x_t, t, noise[t - 1])

        return x_t

    def training_step(self, batch, batch_idx):
        """Compute loss using sampler"""
        x_0, _ = batch
        loss = self.sampler.compute_loss(x_0)

        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """Generate samples for evaluation"""
        x, _ = batch

        self.fid.update(denorm(x), real=True)

        x_T = gaussian_like(x)
        x_0 = self(x_T)

        fake_x = denorm(x_0)

        self.fid.update(fake_x, real=False)
        self.inception.update(fake_x)

    def test_epoch_end(self, outputs):
        """Compute metrics and log at the end of the epoch"""

        fid_score = self.fid.compute()
        kl_mean, _ = self.inception.compute()
        inception_score = torch.exp(kl_mean)

        self.log("fid", fid_score)
        self.log("inception_score", inception_score)

    def configure_optimizers(self):
        """Configure optimizers for training Uses Adam and warmup lr"""
        optimizer = Adam(self.sampler.parameters(), lr=self.hparams.lr)
        lr_scheduler = WarmupLR(optimizer, self.hparams.warmup)

        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def setup(self, stage: str):
        """Prepare metrics for test stage"""

        if stage == "test":
            self.fid = FrechetInceptionDistance(
                normalize=True,
                reset_real_features=False,
            )

            self.inception = InceptionScore(normalize=True)

    def configure_callbacks(self):
        """Configure EMA callback, will override any other EMA callback"""

        ema_callback = EMA(decay=self.hparams.decay)

        return ema_callback
