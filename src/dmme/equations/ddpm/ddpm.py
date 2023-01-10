import torch
from torch import Tensor
from torch.distributions import Normal

import dmme


def linear_schedule(timesteps: int, start: float = 0.0001, end: float = 0.02) -> Tensor:
    r"""constants increasing linearly from :math:`10^{-4}` to :math:`0.02`

    Args:
        timesteps (int): total timesteps
        start (float): starting value, defaults to 0.0001
        end (float): end value, defaults to 0.02

    Returns:
        a 1d tensor representing :math:`\beta_t` indexed by :math:`t`
    """
    beta = torch.linspace(start, end, timesteps)
    return dmme.pad(beta)


def forward_process(image: Tensor, alpha_bar_t: Tensor) -> Normal:
    r"""Forward Process, :math:`q(x_t|x_{t-1})`

    Args:
        image (Tensor): image of shape :math:`(N, C, H, W)`
        alpha_bar_t (Tensor): :math:`\bar\alpha_t` of shape :math:`(N, 1, 1, *)`
        noise (Tensor): noise sampled from standard normal distribution with the same shape as the image

    Returns:
        gaussian transition distirbution :math:`q(x_t|x_{t-1})`
    """

    mean = torch.sqrt(alpha_bar_t) * image

    variance = 1 - alpha_bar_t
    std = torch.sqrt(variance)

    return Normal(mean, std)


def reverse_process(
    x_t: Tensor,
    beta_t: Tensor,
    alpha_t: Tensor,
    alpha_bar_t: Tensor,
    noise_in_x_t: Tensor,
    variance: Tensor,
) -> Normal:
    r"""Reverse Denoising Process, :math:`p_\theta(x_{t-1}|x_t)`

    Args:
        beta_t (Tensor): :math:`\beta_t` of shape :math:`(N, 1, 1, *)`
        alpha_t (Tensor): :math:`\alpha_t` of shape :math:`(N, 1, 1, *)`
        alpha_bar_t (Tensor): :math:`\bar\alpha_t` of shape :math:`(N, 1, 1, *)`
        noise_in_x_t (Tensor): estimated noise in :math:`x_t` predicted by a neural network
        variance (Tensor): variance of the reverse process, either learned or fixed
        noise (Tensor): noise sampled from :math:`\mathcal{N}(0, I)`

    Returns:
        (torch.distributions.Normal) denoising distirbution :math:`q(x_t|x_{t-1})`
    """

    mean = (
        1
        / torch.sqrt(alpha_t)
        * (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_in_x_t)
    )
    std = torch.sqrt(variance)
    return Normal(mean, std)
