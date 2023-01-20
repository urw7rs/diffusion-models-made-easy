from typing import Tuple
from torch import Tensor

from tqdm import tqdm

import torch
from torch import nn

from dmme import gaussian

from dmme.diffusion_models import DDPM

import dmme.equations as eq


class DDIM(DDPM):
    r"""Denoising Diffusion Implicit Models

        A more efficient class of iterative implicit probablistic models with the same training
        procedure as DDPMs.

    Args:
        model: model passed to :code:`DDPM`

        timesteps: total timesteps :math:`T`
        sub_timesteps: sub-sequence length
        tau_schedule: tau schedule to use, `"linear"`or `"quadratic"`
    """

    tau: Tensor

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        sub_timesteps: int = 50,
        tau_schedule: str = "quadratic",
    ) -> None:
        super().__init__(model, timesteps)

        self.sub_timesteps = sub_timesteps

        tau_schedule = tau_schedule.lower()
        if tau_schedule == "linear":
            tau = eq.ddim.linear_tau(timesteps, sub_timesteps)

        elif tau_schedule == "quadratic":
            tau = eq.ddim.quadratic_tau(timesteps, sub_timesteps)

        else:
            raise NotImplementedError

        self.register_buffer("tau", tau, persistent=False)

    def sampling_step(self, x_tau_i: Tensor, i: Tensor) -> Tensor:
        r"""Sample from :math:`p_\theta(x_\tau_{i-1}|x_\tau_i)`

        Args:
            x_tau_i: image of shape :math:`(N, C, H, W)`
            i: :math:`i` in :math:`\tau_i`

        Returns:
            generated sample of shape :math:`(N, C, H, W)`
        """
        tau_i = self.tau[i]
        tau_i_minus_one = self.tau[i - 1]

        alpha_bar_tau_i_minus_one = self.alpha_bar[tau_i_minus_one]
        alpha_bar_tau_i = self.alpha_bar[tau_i]

        noise_in_x_tau_i = self.model(x_tau_i, tau_i)

        p = eq.ddim.reverse_process(
            x_tau_i, alpha_bar_tau_i, alpha_bar_tau_i_minus_one, noise_in_x_tau_i
        )
        # only return mean as noise term is zero
        return p.mean

    def generate(self, img_size: Tuple[int, int, int, int]) -> Tensor:
        """Generate image of shape :math:`(N, C, H, W)` faster by only sampling the sub sequence

        Args:
            img_size: image size to generate as a tuple :math:`(N, C, H, W)`

        Returns:
            generated image of shape :math:`(N, C, H, W)`
        """

        x_tau_i = gaussian(img_size, device=self.beta.device)
        all_i = torch.arange(
            0,
            self.sub_timesteps + 1,
            device=self.beta.device,
        ).unsqueeze(dim=1)

        for i in tqdm(range(self.sub_timesteps, 0, -1), leave=False):
            x_tau_i = self.sampling_step(x_tau_i, all_i[i])

        return x_tau_i
