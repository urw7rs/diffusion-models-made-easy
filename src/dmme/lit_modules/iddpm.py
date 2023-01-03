from typing import Optional

from torch import nn

from dmme.diffusion_models import IDDPM
from dmme.models.iddpm import UNet

from .ddpm import LitDDPM


class LitIDDPM(LitDDPM):
    def __init__(
        self,
        diffusion_model: Optional[IDDPM] = None,
        lr: float = 0.0002,
        warmup: int = 5000,
        decay: float = 0.9999,
        model: Optional[nn.Module] = None,
        timesteps: int = 4000,
        offset: float = 0.008,
        loss_type: str = "hybrid",
        gamma: float = 0.001,
        schedule: str = "cosine",
    ):
        if diffusion_model is None:
            if model is None:
                model = UNet()
            diffusion_model = IDDPM(
                model, timesteps, offset, loss_type, gamma, schedule
            )

        super().__init__(diffusion_model, lr, warmup, decay)
