import torch

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
