from typing import Tuple

from tqdm import tqdm

import torch
from torch import nn

import einops

import dmme
import dmme.equations as eq


class DDPM(nn.Module):
    """Training and Sampling for DDPM

    Args:
        timesteps (int): total timesteps :math:`T`
    """

    beta: torch.Tensor
    alpha: torch.Tensor
    alpha_bar: torch.Tensor

    def __init__(
        self, model: nn.Module, timesteps: int = 1000, start=0.0001, end=0.02
    ) -> None:
        super().__init__()

        self.model = model
        self.timesteps = timesteps

        beta = eq.ddpm.linear_schedule(timesteps, start, end)
        beta = einops.rearrange(beta, "t -> t 1 1 1")

        alpha = 1 - beta

        # alpha[0] = 1 so no problems here
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta, persistent=False)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)

    def training_step(self, x_0):
        r"""Computes loss for DDPM

        Args:
            x_0 (torch.Tensor): sample image to add noise and denoise for training

        Returns:
            (torch.Tensor): loss, :math:`L_\text{simple}`
        """

        batch_size = x_0.size(0)

        time = dmme.uniform_int(
            1,
            self.timesteps,
            batch_size,
            device=x_0.device,
        )
        noise = dmme.gaussian_like(x_0)

        alpha_bar_t = self.alpha_bar[time]

        x_t = eq.ddpm.forward_process(x_0, alpha_bar_t, noise)
        noise_in_x_t = self.model(x_t, time)

        loss = eq.ddpm.simple_loss(noise, noise_in_x_t)
        return loss

    def sampling_step(self, x_t, t):
        r"""Denoise image by sampling from :math:`p_\theta(x_{t-1}|x_t)`

        Args:
            model (nn.Module): model for estimating noise
            x_t (torch.Tensor): image of shape :math:`(N, C, H, W)`
            t (torch.Tensor): starting :math:`t` to sample from, a tensor of shape :math:`(N,)`

        Returns:
            (torch.Tensor): denoised image of shape :math:`(N, C, H, W)`
        """
        noise = dmme.gaussian_like(x_t)

        (idx,) = torch.where(t == 1)
        noise[idx] = 0

        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]

        noise_in_x_t = self.model(x_t, t)
        x_t = eq.ddpm.reverse_process(
            x_t,
            beta_t,
            alpha_t,
            alpha_bar_t,
            noise_in_x_t,
            variance=beta_t,
            noise=noise,
        )
        return x_t

    def generate(self, img_size: Tuple[int, int, int, int]):
        """Generate image of shape :math:`(N, C, H, W)` by running the full denoising steps

        Args:
            img_size (Tuple[int, int, int, int]): image size to generate as a tuple :math:`(N, C, H, W)`

        Returns:
            (torch.Tensor): generated image of shape :math:`(N, C, H, W)`
        """

        x_t = dmme.gaussian(img_size, device=self.beta.device)
        all_t = torch.arange(
            0,
            self.timesteps + 1,
            device=self.beta.device,
        ).unsqueeze(dim=1)

        for t in tqdm(range(self.timesteps, 0, -1), leave=False):
            x_t = self.sampling_step(x_t, all_t[t])

        return x_t

    def forward(self, x, t):
        """Predicts noise given image and timestep"""

        noise_in_x = self.model(x, t)
        return noise_in_x
