from typing import Optional

from torch import nn

from dmme.diffusion_models import DDIM
from dmme.models.ddpm import UNet

from .ddpm import LitDDPM


class LitDDIM(LitDDPM):
    r"""LightningModule for accelerated sampling with DDIM using :code:`LitDDPM`'s checkpoints

    Args:
        lr: learning rate, defaults to :code:`2e-4`
        warmup: linearly increases learning rate for
            `warmup` steps until `lr` is reached, defaults to 5000
        decay: EMA decay value

        diffusion_model: overrides diffusion_model :code:`DDIM`
        model: overrides model passed to :code:`DDIM`

        timesteps: default timesteps passed to :code:`DDIM`
        sample_steps: default sample steps passed to :code:`DDIM`
        tau_schedule: default tau schedule passed to :code:`DDIM`
    """

    def __init__(
        self,
        lr: float = 2e-4,
        warmup: int = 5000,
        decay: float = 0.9999,
        diffusion_model: Optional[DDIM] = None,
        model: Optional[nn.Module] = None,
        timesteps: int = 1000,
        sample_steps: int = 50,
        tau_schedule: str = "quadratic",
    ):

        if diffusion_model is None:
            if model is None:
                model = UNet()
            diffusion_model = DDIM(model, timesteps, sample_steps, tau_schedule)

        super().__init__(lr, warmup, decay, diffusion_model)
