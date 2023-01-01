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


def sample_gaussian(mean, variance, noise):
    r"""Samples from a gaussian distribution using the reparameterization trick

    Args:
        mean (torch.Tensor): mean of the distribution
        variance (torch.Tensor): variance of the distribution
        noise (torch.Tensor): noise sampled from :math:`\mathcal{N}(0, I)`
    """
    return mean + torch.sqrt(variance) * noise


def forward_process(image, alpha_bar_t, noise):
    r"""Forward Process, :math:`q(x_t|x_{t-1})`

    Args:
        image (torch.Tensor): image of shape :math:`(N, C, H, W)`
        alpha_bar_t (torch.Tensor): :math:`\bar\alpha_t` of shape :math:`(N, 1, 1, *)`
        noise (torch.Tensor): noise sampled from standard normal distribution with the same shape as the image
    """

    mean = torch.sqrt(alpha_bar_t) * image
    variance = 1 - alpha_bar_t
    return sample_gaussian(mean, variance, noise)


def reverse_process(x_t, beta_t, alpha_t, alpha_bar_t, noise_in_x_t, variance, noise):
    r"""Reverse Denoising Process, :math:`p_\theta(x_{t-1}|x_t)`

    Args:
        beta_t (torch.Tensor): :math:`\beta_t` of shape :math:`(N, 1, 1, *)`
        alpha_t (torch.Tensor): :math:`\alpha_t` of shape :math:`(N, 1, 1, *)`
        alpha_bar_t (torch.Tensor): :math:`\bar\alpha_t` of shape :math:`(N, 1, 1, *)`
        noise_in_x_t (torch.Tensor): estimated noise in :math:`x_t` predicted by a neural network
        variance (torch.Tensor): variance of the reverse process, either learned or fixed
        noise (torch.Tensor): noise sampled from :math:`\mathcal{N}(0, I)`
    """

    mean = (
        1
        / torch.sqrt(alpha_t)
        * (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_in_x_t)
    )
    return sample_gaussian(mean, variance, noise)


def simple_loss(noise, estimated_noise):
    r"""Simple Loss objective :math:`L_\text{simple}`, MSE loss between noise and predicted noise

    Args:
        noise (torch.Tensor): noise used in the forward process
        estimated_noise (torch.Tensor): estimated noise with the same shape as :code:`noise`

    """
    return nn.functional.mse_loss(noise, estimated_noise)


class DDPM(nn.Module):
    """Training and Sampling for DDPM

    Args:
        timesteps (int): total timesteps :math:`T`
    """

    beta: torch.Tensor
    alpha: torch.Tensor
    alpha_bar: torch.Tensor

    def __init__(self, model, timesteps) -> None:
        super().__init__()

        self.model = model
        self.timesteps = timesteps

        beta = linear_schedule(timesteps)
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
        x_t = reverse_process(
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
