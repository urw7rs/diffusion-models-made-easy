from typing import Optional

from torch import nn

from dmme.diffusion_models import IDDPM
from dmme.models.iddpm import UNet

from .ddpm import LitDDPM


class LitIDDPM(LitDDPM):
    r"""Improved Denoising Diffusion Probablistic Models

    Args:
        lr: learning rate, defaults to :code:`2e-4`
        warmup: linearly increases learning rate for
            `warmup` steps until `lr` is reached, defaults to 5000
        decay: EMA decay value

        diffusion_model: overrides default diffusion_model :code:`DDPM`
        model: overrides default model passed to :code:`DDPM`

        timesteps: default timesteps passed to :code:`DDPM`
        loss_type: loss type to use either "hybrid" or "simple"
        gamma: :math:`\gamma` in hybrid loss
        shcedule: variance schedule to use either "linear" or "cosine"
        offset: default offset for :code:`IDDPM` if cosine schedule is used
        start: default start for :code:`IDDPM` if linear schedule is used
        end: default end for :code:`IDDPM` if linear schedule is used
    """

    def __init__(
        self,
        lr: float = 0.0002,
        warmup: int = 5000,
        decay: float = 0.9999,
        diffusion_model: Optional[IDDPM] = None,
        model: Optional[nn.Module] = None,
        timesteps: int = 1000,
        loss_type: str = "hybrid",
        gamma: float = 0.001,
        schedule: str = "cosine",
        offset: float = 0.008,
        start: float = 0.0001,
        end: float = 0.02,
    ):
        if diffusion_model is None:
            if model is None:
                model = UNet()
            diffusion_model = IDDPM(
                model, timesteps, loss_type, gamma, schedule, offset, start, end
            )

        super().__init__(lr, warmup, decay, diffusion_model)
