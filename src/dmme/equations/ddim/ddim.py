import torch

import dmme.equations as eq


def linear_tau(timesteps, sub_timesteps):
    """linear tau schedule

    Args:
        timesteps (int): total timesteps :math:`T`
        sub_timesteps (int): sub sequence length less than :math:`T`
    """
    all_t = torch.arange(0, sub_timesteps + 1)
    c = timesteps / sub_timesteps
    tau = torch.round(c * all_t).long()
    return tau


def quadratic_tau(timesteps, sub_timesteps):
    """quadratic tau schedule

    Args:
        timesteps (int): total timesteps :math:`T`
        sub_timesteps (int): sub sequence length less than :math:`T`
    """
    all_t = torch.arange(0, sub_timesteps + 1)
    c = timesteps / (timesteps**2)
    tau = torch.round(c * all_t**2).long()
    return tau


def reverse_process(
    x_tau_i, alpha_bar_tau_i, alpha_bar_tau_i_minus_one, noise_in_x_tau_i
):
    r"""DDIM Reverse Denoising Process

    Args:
        model (nn.Module): model for estimating noise
        x_t (torch.Tensor): x_t
        t (int): current timestep
        noise (torch.Tensor): noise
    """
    predicted_x_0 = (
        x_tau_i - torch.sqrt(1 - alpha_bar_tau_i) * noise_in_x_tau_i
    ) / torch.sqrt(alpha_bar_tau_i)

    x_tau_i_minus_one = eq.ddpm.forward_process(
        predicted_x_0, alpha_bar_tau_i_minus_one, noise_in_x_tau_i
    )

    return x_tau_i_minus_one
