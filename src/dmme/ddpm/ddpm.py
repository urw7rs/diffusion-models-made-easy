from typing import Tuple

import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam
from torchmetrics.image.fid import FrechetInceptionDistance

from dmme.common import denorm, make_history
from dmme.lr_scheduler import WarmupLR

from .ddpm_sampler import DDPMSampler


class LitDDPM(pl.LightningModule):
    def __init__(
        self,
        decoder: nn.Module,
        lr: float = 2e-4,
        warmup: int = 5000,
        imgsize: Tuple[int, int, int] = (3, 32, 32),
        timesteps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="decoder")

        self.sampler = DDPMSampler(decoder, timesteps=timesteps)

        self.fid = FrechetInceptionDistance(
            normalize=True,
            reset_real_features=False,
        )

    def training_step(self, batch, batch_idx):
        x_0, _ = batch
        loss = self.sampler.compute_loss(x_0)

        self.log("train/loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        self.log_sample(num_samples=16, length=16)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        self.fid.update(denorm(x), real=True)

        batch_size = x.size(0)
        history = self.log_sample(num_samples=batch_size, length=1)

        final_img = history[-1]
        self.fid.update(final_img, real=False)

    def test_epoch_end(self, outputs):
        fid_score = self.fid.compute()
        self.log("fid", fid_score)

    def configure_optimizers(self):
        optimizer = Adam(self.sampler.parameters(), lr=self.hparams.lr)
        lr_scheduler = WarmupLR(optimizer, self.hparams.warmup)

        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

    def log_sample(self, num_samples=1, length=1):
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
