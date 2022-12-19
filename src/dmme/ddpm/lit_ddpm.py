from typing import Tuple

import torch
from torch import nn
from torch.optim import Adam

import pytorch_lightning as pl

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from dmme.common import denorm, make_history
from dmme.lr_scheduler import WarmupLR

from dmme.callbacks import EMA
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


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
        sampler: nn.Module,
        lr: float = 2e-4,
        warmup: int = 5000,
        imgsize: Tuple[int, int, int] = (3, 32, 32),
        timesteps: int = 1000,
        decay: float = 0.9999,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="sampler")

        self.sampler = sampler

    def training_step(self, batch, batch_idx):
        """Compute loss using sampler"""
        x_0, _ = batch
        loss = self.sampler.compute_loss(x_0)

        self.log("train/loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        """Generate samples at the end of the epoch"""
        self.sample_and_log(num_samples=16, length=16)

    def test_step(self, batch, batch_idx):
        """Generate samples for evaluation"""
        x, _ = batch
        x = denorm(x)

        self.fid.update(x, real=True)

        batch_size = x.size(0)
        history = self.sample_and_log(num_samples=batch_size, length=1)

        final_img = history[-1]
        self.fid.update(final_img, real=False)
        self.inception.update(final_img)

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

    def sample_and_log(self, num_samples=1, length=1):
        """Sample data and log to logger

        Args:
            num_samples (int): number of samples
            length (int): length of history to save in :math:`T` timesteps
        """
        if length == 1:
            start = end = 0
            step = 1
        elif length > 2:
            start = 0
            end = self.sampler.timesteps + 1
            step = (self.sampler.timesteps - 1) // (length - 1)

        history = self.sampler.sample(
            (num_samples, *self.hparams.imgsize),
            start,
            end,
            step,
            save_last=True,
            device=self.device,
        )

        grid = make_history(history)

        self.logger.log_image("samples", [grid])

        return history
