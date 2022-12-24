from typing import Tuple, Optional

from tqdm import tqdm

import torch
from torch import nn, Tensor

from dmme import LitDDPM
from dmme.common import denorm, gaussian_like

from dmme.ddpm import ForwardProcess, linear_schedule
from .ddim import DDIMSampler


class LitDDIM(LitDDPM):
    """LightningModule for training DDIM

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
        tau_schedule: str = "linear",
        eta: float = 0.0,
        lr: float = 0.0002,
        warmup: int = 5000,
        imgsize: Tuple[int, int, int] = (3, 32, 32),
        sample_steps: int = 50,
        timesteps: int = 1000,
        decay: float = 0.9999,
    ):
        super().__init__(model, lr, warmup, imgsize, timesteps, decay)
        self.save_hyperparameters(ignore=["model", "eta", "sample_steps"])

        self.sample_steps = sample_steps

        beta = self.forward_process.beta
        self.sampler = DDIMSampler(beta, sample_steps, tau_schedule, eta)

    def generate(self, x_t):
        """Generate samples for evaluation"""

        noise = [None]
        for _ in range(self.sample_steps, 0, -1):
            noise.append(gaussian_like(x_t))

        for t in tqdm(range(self.sample_steps, 0, -1), leave=False):
            x_t = self(x_t, t, noise[t])

        return x_t
