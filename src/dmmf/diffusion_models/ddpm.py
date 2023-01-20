from typing import Tuple
from torch import Tensor

from tqdm import tqdm

import torch
from torch import nn

import einops

import dmme
import dmme.equations as eq


class DDPM(nn.Module):
    r"""Training and Sampling for DDPM

    Args:
        model: model predicting noise from data, :math:`\epsilon_\theta(x_t, t)`
        timesteps: total timesteps :math:`T`
        start: linear variance schedule start value
        end: linear variance schedule end value
    """

    beta: Tensor
    alpha: Tensor
    alpha_bar: Tensor

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        start: float = 0.0001,
        end: float = 0.02,
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

    def training_step(self, x_0: Tensor) -> Tensor:
        r"""Training step except for optimization

        Args:
            x_0: image from dataset

        Returns:
            loss, :math:`L_\text{simple}`
        """

        batch_size = x_0.size(0)

        time = dmme.uniform_int(
            1,
            self.timesteps,
            batch_size,
            device=x_0.device,
        )

        alpha_bar_t = self.alpha_bar[time]

        q = eq.ddpm.forward_process(x_0, alpha_bar_t)
        x_t = q.sample()

        noise_in_x_t = self.model(x_t, time)

        noise = (x_t - q.mean) / q.stddev
        loss = eq.ddpm.simple_loss(noise, noise_in_x_t)
        return loss

    def sampling_step(self, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Denoise image by sampling from :math:`p_\theta(x_{t-1}|x_t)`

        Args:
            x_t: image of shape :math:`(N, C, H, W)`
            t: starting :math:`t` to sample from, a tensor of shape :math:`(N,)`

        Returns:
            denoised image of shape :math:`(N, C, H, W)`
        """

        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]

        noise_in_x_t = self.model(x_t, t)
        p = eq.ddpm.reverse_process(
            x_t,
            beta_t,
            alpha_t,
            alpha_bar_t,
            noise_in_x_t,
            variance=beta_t,
        )
        x_t = p.sample()

        # set z to 0 when t = 1 by overwriting values
        x_t = torch.where(t == 1, p.mean, x_t)
        return x_t

    def generate(self, img_size: Tuple[int, int, int, int]) -> Tensor:
        """Generate image of shape :math:`(N, C, H, W)` by running the full denoising steps

        Args:
            img_size: image size to generate as a tuple :math:`(N, C, H, W)`

        Returns:
            generated image of shape :math:`(N, C, H, W)` as a tensor
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

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Applies forward to internal model

        Args:
            x: input image passed to internal model
            t: timestep passed to internal model
        """

        noise_in_x = self.model(x, t)
        return noise_in_x
