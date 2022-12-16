from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

import einops

from dmme.common import gaussian, gaussian_like, uniform_int, denorm
from dmme.noise_schedules.fixed import linear_schedule


class DDPMSampler(nn.Module):
    """Denoising Diffusion Probabilistic Model"""

    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        beta: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.model = model

        self.timesteps = timesteps

        if beta is None:
            beta = linear_schedule(timesteps)

        alpha = 1.0 - beta

        alpha_bar = torch.cumprod(alpha, dim=0)

        sqrt_alpha_bar = einops.rearrange(torch.sqrt(alpha_bar), "t -> t 1 1 1")
        sqrt_one_minus_alpha_bar = einops.rearrange(
            torch.sqrt(1 - alpha_bar), "t -> t 1 1 1"
        )

        one_over_sqrt_alpha = 1 / torch.sqrt(alpha)
        beta_over_sqrt_one_minus_alpha_bar = beta / torch.sqrt(1 - alpha_bar)

        sigma = torch.sqrt(beta)

        self.register_buffer("beta", beta)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar)
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_bar)
        self.register_buffer("one_over_sqrt_alpha", one_over_sqrt_alpha)
        self.register_buffer(
            "beta_over_sqrt_one_minus_alpha_bar", beta_over_sqrt_one_minus_alpha_bar
        )
        self.register_buffer("sigma", sigma)
        self.register_buffer("t", torch.arange(1, self.timesteps)[:, None])

    def forward(self, x, t, *args, **kwargs):
        return self.model(x, t, *args, **kwargs)

    @torch.no_grad()
    def forward_process(self, x_0, t, noise=None):
        t_index = t - 1

        if noise is None:
            noise = gaussian_like(x_0)

        x_t = (
            self.sqrt_alpha_bar[t_index] * x_0
            + self.sqrt_one_minus_alpha_bar[t_index] * noise
        )

        return x_t

    @torch.inference_mode()
    def reverse_process(self, x_t, t, noise=None):
        t_index = t - 1

        t = torch.tensor([t], device=x_t.device).float()

        if noise is None:
            noise = gaussian_like(x_t)

        x_t_minus_one = (
            self.one_over_sqrt_alpha[t_index]
            * (
                x_t
                - self.beta_over_sqrt_one_minus_alpha_bar[t_index] * self.model(x_t, t)
            )
            + self.sigma[t_index] * noise
        )

        return x_t_minus_one

    def compute_loss(self, x_0, t=None, noise=None):
        if t is None:
            batch_size = x_0.size(0)
            t = uniform_int(0, self.timesteps, batch_size, device=x_0.device)

        if noise is None:
            noise = gaussian_like(x_0)

        noisy_x = self.forward_process(x_0, t, noise)

        noise_estimate = self.model(noisy_x, t)
        loss = F.mse_loss(noise, noise_estimate)
        return loss

    @torch.inference_mode()
    def sample(self, x_shape, start=0, end=None, step=1, save_last=True, device=None):

        history = []

        def save(x):
            history.append(denorm(x))

        def format(n):
            if n < 0:
                n %= self.timesteps
                n += 1
            elif n is None:
                n = self.timesteps + 1
            return n

        start = format(start)
        end = format(end)

        x_t = gaussian(x_shape, device=device)

        if self.timesteps == start:
            save(x_t)

        for t in range(self.timesteps, 1, -1):
            x_t = self.reverse_process(x_t, t)

            if end > t - 1 >= start:
                if (t - 1 - start) % step == 0:
                    save(x_t)

        x_0 = self.reverse_process(x_t, 1, noise=0)
        if save_last or end > 0 >= start:
            save(x_0)

        return history
