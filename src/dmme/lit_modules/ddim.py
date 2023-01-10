from typing import Optional

from torch import nn

from dmme.diffusion_models import DDIM
from dmme.models.ddpm import UNet

from .ddpm import LitDDPM


class LitDDIM(LitDDPM):
    r"""LightningModule for sampling with DDIM with :code:`LitDDPM`'s checkpoints

    Args:
        diffusion_model: diffusion_model describing the forward, reverse process and trainig step.
            If set to :code:`None` will set to default diffusion_model
        lr: learning rate, defaults to :code:`2e-4`
        warmup: linearly increases learning rate for
            `warmup` steps until `lr` is reached, defaults to 5000
        timestpes: total timesteps for the
            forward and reverse process, :math:`T`
        decay: EMA decay value
        sample_steps: sample steps for generation process
        tau_schedule: tau schedule to use for generation, `"linear"` or `"quadratic"`
    """

    def __init__(
        self,
        diffusion_model: Optional[DDIM] = None,
        lr: float = 2e-4,
        warmup: int = 5000,
        decay: float = 0.9999,
        model: Optional[nn.Module] = None,
        timesteps: int = 1000,
        sample_steps: int = 50,
        tau_schedule: str = "quadratic",
    ):

        if diffusion_model is None:
            if model is None:
                model = UNet()
            diffusion_model = DDIM(model, timesteps, sample_steps, tau_schedule)

        super().__init__(
            diffusion_model=diffusion_model,
            lr=lr,
            warmup=warmup,
            decay=decay,
        )
