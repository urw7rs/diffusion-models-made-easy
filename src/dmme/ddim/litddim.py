from typing import Tuple

from torch import nn

from dmme.ddpm import LitDDPM

from .ddim import DDIM


class LitDDIM(LitDDPM):
    r"""LightningModule for sampling with DDIM with :code:`LitDDPM`'s checkpoints

    Args:
        model (nn.Module): neural network predicting noise :math:`\epsilon_\theta`
        lr (float): learning rate, defaults to :math:`2e-4`
        warmup (int): linearly increases learning rate for
            `warmup` steps until `lr` is reached, defaults to 5000
        imgsize (Tuple[int, int, int]): image size in `(C, H, W)`
        timestpes (int): total timesteps for the
            forward and reverse process, :math:`T`
        decay (float): EMA decay value
        sample_steps (int): sample steps for generation process
        tau_schedule (str): tau schedule to use for generation, `"linear"` or `"quadratic"`
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-4,
        warmup: int = 5000,
        imgsize: Tuple[int, int, int] = (3, 32, 32),
        timesteps: int = 1000,
        decay: float = 0.9999,
        sample_steps: int = 50,
        tau_schedule: str = "quadratic",
    ):
        super().__init__(model, lr, warmup, imgsize, timesteps, decay)

        self.sample_steps = sample_steps
        self.diffusion = DDIM(model, timesteps, sample_steps, tau_schedule=tau_schedule)
