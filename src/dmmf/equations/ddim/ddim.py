from torch import Tensor

import torch
from torch.distributions import Normal

import dmme.equations as eq


def linear_tau(timesteps: int, sub_timesteps: int) -> Tensor:
    r"""Linear sub-sequence :math:`\tau`

    Args:
        timesteps: total timesteps :math:`T`
        sub_timesteps: sub-sequence length less than :math:`T`
    """
    all_t = torch.arange(0, sub_timesteps + 1)
    c = timesteps / sub_timesteps

    tau = torch.round(c * all_t)
    return tau.long()


def quadratic_tau(timesteps: int, sub_timesteps: int) -> Tensor:
    r"""Quadratic sub-sequence :math:`\tau`

    Args:
        timesteps: total timesteps :math:`T`
        sub_timesteps: sub-sequence length less than :math:`T`
    """
    all_t = torch.arange(0, sub_timesteps + 1)
    c = timesteps / (sub_timesteps**2)

    tau = torch.round(c * all_t**2)
    return tau.long()


def reverse_process(
    x_t: Tensor,
    alpha_bar_t: Tensor,
    alpha_bar_t_minus_one: Tensor,
    noise_in_x_t: Tensor,
) -> Normal:
    r"""Deterministic Denoising Process where :math:`\sigma_t = 0` for all :math:`t`

    Args:
        x_t: :math:`x_t`
        alpha_bar_t: :math:`\bar\alpha_t`
        alpha_bar_t_minus_one: :math:`\bar\alpha_{t-1}` of shape :math:`(N, 1, 1, *)`
        noise_in_x_t: estimated noise in :math:`x_t` predicted by a neural network
    """

    predicted_x_0 = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_in_x_t) / torch.sqrt(
        alpha_bar_t_minus_one
    )

    p = eq.ddpm.forward_process(predicted_x_0, alpha_bar_t_minus_one)
    return p
