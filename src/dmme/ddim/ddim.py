from typing import Tuple
from tqdm import tqdm

import torch

from dmme import ddpm
from dmme.common import gaussian


def linear_tau(timesteps, sub_timesteps):
    all_t = torch.arange(0, sub_timesteps + 1)
    c = timesteps / sub_timesteps
    tau = torch.round(c * all_t).long()
    return tau


def quadratic_tau(timesteps, sub_timesteps):
    all_t = torch.arange(0, sub_timesteps + 1)
    c = timesteps / (timesteps**2)
    tau = torch.round(c * all_t**2).long()
    return tau


def reverse_process(
    x_tau_i, alpha_bar_tau_i, alpha_bar_tau_i_minus_one, noise_in_x_tau_i
):
    predicted_x_0 = (
        x_tau_i - torch.sqrt(1 - alpha_bar_tau_i) * noise_in_x_tau_i
    ) / torch.sqrt(alpha_bar_tau_i)

    x_tau_i_minus_one = ddpm.forward_process(
        predicted_x_0, alpha_bar_tau_i_minus_one, noise_in_x_tau_i
    )

    return x_tau_i_minus_one


class DDIM(ddpm.DDPM):
    r"""Reverse process and Sampling for DDIM

    Args:
        timesteps (int): total timesteps :math:`T`
        tau_schedule (str): tau schedule, `"linear"`or `"quadratic"`
    """

    tau: torch.Tensor

    def __init__(
        self, model, timesteps, sub_timesteps, tau_schedule="quadratic"
    ) -> None:
        super().__init__(model, timesteps)

        self.sub_timesteps = sub_timesteps

        tau_schedule = tau_schedule.lower()
        if tau_schedule == "linear":
            tau = linear_tau(timesteps, sub_timesteps)

        elif tau_schedule == "quadratic":
            tau = quadratic_tau(timesteps, sub_timesteps)

        else:
            raise NotImplementedError

        self.register_buffer("tau", tau, persistent=False)

    def reverse_process(self, x_tau_i, i):
        r"""Reverse Denoising Process

        Samples :math:`x_{t-1}` from
        :math:`p_\theta(\bold{x}_{t-1}|\bold{x}_t)
        = \mathcal{N}(\bold{x}_{t-1};\mu_\theta(\bold{x}_t, t), \sigma_t\bold{I})`

        .. math::
            \begin{aligned}
            \bold\mu_\theta(\bold{x}_t, t)
            &= \frac{1}{\sqrt{\alpha_t}}\bigg(\bold{x}_t
            -\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\bold{x}_t,t)\bigg) \\
            \sigma_t &= \beta_t
            \end{aligned}

        Computes :math:`\bold{x}_{t-1}
        = \frac{1}{\sqrt{\alpha_t}}\bigg(\bold{x}_t
        -\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(\bold{x}_t,t)\bigg)
        +\sigma_t\epsilon`

        Args:
            model (nn.Module): model for estimating noise
            x_t (torch.Tensor): x_t
            t (int): current timestep
            noise (torch.Tensor): noise
        """

        tau_i = self.tau[i]
        tau_i_minus_one = self.tau[i - 1]

        alpha_bar_tau_i_minus_one = self.alpha_bar[tau_i_minus_one]
        alpha_bar_tau_i = self.alpha_bar[tau_i]

        noise_in_x_tau_i = self.model(x_tau_i, tau_i)

        x_tau_i_minus_one = reverse_process(
            x_tau_i, alpha_bar_tau_i, alpha_bar_tau_i_minus_one, noise_in_x_tau_i
        )

        return x_tau_i_minus_one

    def sampling_step(self, x_tau_i, i):
        r"""Sample from :math:`p_\theta(x_{t-1}|x_t)`

        Args:
            model (nn.Module): model for estimating noise
            x_t (torch.Tensor): image of shape :math:`(N, C, H, W)`
            t (int): starting :math:`t` to sample from

        Returns:
            (torch.Tensor): generated sample of shape :math:`(N, C, H, W)`
        """
        return self.reverse_process(x_tau_i, i)

    def generate(self, img_size: Tuple[int, int, int, int]):
        x_tau_i = gaussian(img_size, device=self.beta.device)
        all_i = torch.arange(
            0,
            self.sub_timesteps + 1,
            device=self.beta.device,
        ).unsqueeze(dim=1)

        for i in tqdm(range(self.sub_timesteps, 0, -1), leave=False):
            x_tau_i = self.sampling_step(x_tau_i, all_i[i])

        return x_tau_i
