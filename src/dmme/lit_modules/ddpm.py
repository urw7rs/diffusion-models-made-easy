from typing import Optional

import torch
from torch import Tensor

import pytorch_lightning as pl

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from torch.optim import Adam
from dmme.lr_scheduler import WarmupLR
from dmme.callbacks import EMA

import dmme
from dmme.diffusion_models import DDPM
from dmme.models import UNet


class LitDDPM(pl.LightningModule):
    r"""LightningModule for training DDPM

    Args:
        model (nn.Module): neural network predicting noise :math:`\epsilon_\theta`
        lr (float): learning rate, defaults to :math:`2e-4`
        warmup (int): linearly increases learning rate for
            `warmup` steps until `lr` is reached, defaults to 5000
        timestpes (int): total timesteps for the
            forward and reverse process, :math:`T`
        decay (float): EMA decay value
    """

    def __init__(
        self,
        diffusion_model: Optional[DDPM] = None,
        lr: float = 2e-4,
        warmup: int = 5000,
        decay: float = 0.9999,
    ):
        super().__init__()

        self.lr = lr
        self.warmup = warmup
        self.decay = decay

        if diffusion_model is None:
            model = UNet()
            diffusion_model = DDPM(model=model, timesteps=1000)

        self.diffusion_model = diffusion_model

        self.fid = FrechetInceptionDistance(
            normalize=True,
            reset_real_features=False,
        )

        self.inception = InceptionScore(normalize=True)

    def forward(self, x_t: Tensor, t: int):
        r"""Denoise image once using `DDPM`

        Args:
            x_t (torch.Tensor): image of shape :math:`(N, C, H, W)`
            t (int): starting :math:`t` to sample from
            noise (torch.Tensor): noise to use for sampling, if `None` samples new noise

        Returns:
            (torch.Tensor): generated sample of shape :math:`(N, C, H, W)`
        """

        timestep = torch.tensor([t], device=x_t.device)
        x_t = self.diffusion_model.sampling_step(x_t, timestep)
        return x_t

    def training_step(self, batch, batch_idx):
        r"""Train model using :math:`L_\text{simple}`"""

        x_0: Tensor = batch[0]

        loss: Tensor = self.diffusion_model.training_step(x_0)
        self.log("train/loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        """Generate samples for evaluation"""

        x: Tensor = batch[0]

        self.fid.update(dmme.denorm(x), real=True)

        x_t = self.generate(x.size())
        fake_x: Tensor = dmme.denorm(x_t)

        self.fid.update(fake_x, real=False)
        self.inception.update(fake_x)

    def generate(self, img_size):
        r"""Iteratively sample from :math:`p_\theta(x_{t-1}|x_t)` to generate images

        Args:
            x_t (torch.Tensor): :math:`x_T` to start from
        """

        x_t = self.diffusion_model.generate(img_size=img_size)
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

        optimizer = Adam(self.diffusion_model.parameters(), lr=self.lr)
        lr_scheduler = WarmupLR(optimizer, self.warmup)

        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def configure_callbacks(self):
        """Configure EMA callback, will override any other EMA callback"""
        ema_callback = EMA(decay=self.decay)
        return ema_callback
