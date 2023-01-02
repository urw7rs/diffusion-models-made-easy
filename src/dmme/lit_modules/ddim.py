from typing import Optional

from .ddpm import LitDDPM

from dmme.diffusion_models import DDIM
from dmme.models import UNet


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
        diffusion_model: Optional[DDIM] = None,
        lr: float = 2e-4,
        warmup: int = 5000,
        timesteps: int = 1000,
        decay: float = 0.9999,
        sample_steps: int = 50,
        tau_schedule: str = "quadratic",
    ):

        if diffusion_model is None:
            model = UNet()
            diffusion_model = DDIM(model, timesteps, sample_steps, tau_schedule)

        super().__init__(
            diffusion_model=diffusion_model,
            lr=lr,
            warmup=warmup,
            decay=decay,
        )
