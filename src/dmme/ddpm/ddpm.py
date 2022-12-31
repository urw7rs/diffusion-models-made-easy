from typing import Tuple

from tqdm import tqdm

import torch
from torch import nn

import einops

import dmme


def linear_schedule(timesteps: int, start=0.0001, end=0.02) -> torch.Tensor:
    r"""constants increasing linearly from :math:`10^{-4}` to :math:`0.02`

    Args:
        timesteps (int): total timesteps
        start (float): starting value, defaults to 0.0001
        end (float): end value, defaults to 0.02
    """

    beta = torch.linspace(start, end, timesteps)
    return dmme.pad(beta)


def alphas(beta):
    return 1 - beta


def alpha_bars(alpha):
    # alpha[0] = 1 so no problems here
    alpha_bar = torch.cumprod(alpha, dim=0)
    return alpha_bar


def sample_gaussian(mean, variance, noise):
    return mean + torch.sqrt(variance) * noise


def forward_process(image, alpha_bar_t, noise):
    mean = torch.sqrt(alpha_bar_t) * image
    variance = 1 - alpha_bar_t
    return sample_gaussian(mean, variance, noise)


def reverse_process(x_t, beta_t, alpha_t, alpha_bar_t, noise_in_x_t, variance, noise):
    mean = (
        1
        / torch.sqrt(alpha_t)
        * (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_in_x_t)
    )
    return sample_gaussian(mean, variance, noise)


def simple_loss(noise, estimated_noise):
    return nn.functional.mse_loss(noise, estimated_noise)


class DDPM(nn.Module):
    """Forward, Reverse, Sampling for DDPM

    Args:
        timesteps (int): total timesteps :math:`T`
    """

    beta: torch.Tensor
    alpha: torch.Tensor
    alpha_bar: torch.Tensor
    sigma: torch.Tensor

    def __init__(self, model, timesteps) -> None:
        super().__init__()

        self.model = model
        self.timesteps = timesteps

        beta = linear_schedule(timesteps)
        beta = einops.rearrange(beta, "t -> t 1 1 1")

        alpha = alphas(beta)
        alpha_bar = alpha_bars(alpha)

        self.register_buffer("beta", beta, persistent=False)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)
        self.register_buffer("sigma", beta, persistent=False)

    def training_step(self, x_0):
        batch_size = x_0.size(0)

        time = dmme.uniform_int(
            0,
            self.timesteps,
            batch_size,
            device=x_0.device,
        )
        noise = dmme.gaussian_like(x_0)

        alpha_bar_t = self.alpha_bar[time]

        x_t = forward_process(x_0, alpha_bar_t, noise)
        noise_in_x_t = self.model(x_t, time)

        loss = simple_loss(noise, noise_in_x_t)
        return loss

    def sampling_step(self, x_t, t):
        r"""Sample from :math:`p_\theta(x_{t-1}|x_t)`

        Args:
            model (nn.Module): model for estimating noise
            x_t (torch.Tensor): image of shape :math:`(N, C, H, W)`
            t (torch.Tensor): starting :math:`t` to sample from
            noise (torch.Tensor): noise to use for sampling, if `None` samples new noise

        Returns:
            (torch.Tensor): generated sample of shape :math:`(N, C, H, W)`
        """
        noise = dmme.gaussian_like(x_t)

        (idx,) = torch.where(t == 1)
        noise[idx] = 0

        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        sigma_t = self.sigma[t]

        noise_in_x_t = self.model(x_t, t)
        x_t = reverse_process(
            x_t,
            beta_t,
            alpha_t,
            alpha_bar_t,
            noise_in_x_t,
            variance=sigma_t,
            noise=noise,
        )
        return x_t

    def generate(self, img_size: Tuple[int, int, int, int]):
        x_t = dmme.gaussian(img_size, device=self.beta.device)
        all_t = torch.arange(
            0,
            self.timesteps + 1,
            device=self.beta.device,
        ).unsqueeze(dim=1)

        for t in tqdm(range(self.timesteps, 0, -1), leave=False):
            x_t = self.sampling_step(x_t, all_t[t])

        return x_t
