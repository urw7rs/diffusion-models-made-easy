from typing import Tuple

from tqdm import tqdm

import torch
from torch import nn
from torch import Tensor

from dmme.ddpm import LitDDPM, DDPM, pad


class LitDDIM(LitDDPM):
    r"""LightningModule for sampling with DDIM with :code:`LitDDPM`'s checkpoints

    Args:
        model (nn.Module): neural network predicting noise :math:`\epsilon_\theta`
        lr (float): learning rate, defaults to :math:`2e-4`
        warmup (int): linearly increases learning rate for
            `warmup` steps until `lr` is reached, defaults to 5000
        imgsize (Tuple[int, int, int]): image size in `(C, H, W)`
        timestpes (int): total timesteps for the
            forward and reverse process, :math:`T`
        decay (float): EMA decay value
        sample_steps (int): sample steps for generation process
        tau_schedule (str): tau schedule to use for generation, `"linear"` or `"quadratic"`
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2e-4,
        warmup: int = 5000,
        imgsize: Tuple[int, int, int] = (3, 32, 32),
        timesteps: int = 1000,
        decay: float = 0.9999,
        sample_steps: int = 50,
        tau_schedule: str = "quadratic",
    ):
        super().__init__(model, lr, warmup, imgsize, timesteps, decay)

        self.sample_steps = sample_steps
        self.process = DDIM(timesteps, tau_schedule=tau_schedule)

    def forward(self, x_t: Tensor, t: int):
        r"""Denoise image once using :code:`DDIM`

        Args:
            x_t (torch.Tensor): image of shape :math:`(N, C, H, W)`
            t (int): starting :math:`t` to sample from

        Returns:
            (torch.Tensor): generated sample of shape :math:`(N, C, H, W)`
        """

        timestep = torch.tensor([t], device=x_t.device)

        x_t = self.process.sample(self.model, x_t, timestep)

        return x_t

    def generate(self, x_t):
        r"""Iteratively sample from :math:`p_\theta(x_{t-1}|x_t)` to generate images

        Args:
            x_t (torch.Tensor): :math:`x_T` to start from
        """

        for t in tqdm(range(self.sample_steps, 0, -1), leave=False):
            x_t = self(x_t, t)

        return x_t


class DDIM(DDPM):
    r"""Reverse process and Sampling for DDIM

    Args:
        timesteps (int): total timesteps :math:`T`
        tau_schedule (str): tau schedule, `"linear"`or `"quadratic"`
    """

    tau: Tensor

    def __init__(self, timesteps, tau_schedule="quadratic") -> None:
        super().__init__(timesteps)

        full_timesteps = self.beta.size(0) - 1

        tau_schedule = tau_schedule.lower()
        if tau_schedule == "linear":
            c = full_timesteps / timesteps

            tau = [round(c * i) for i in range(timesteps + 1)]

        elif tau_schedule == "quadratic":
            c = full_timesteps / (timesteps**2)

            tau = [round(c * i**2) for i in range(timesteps + 1)]
        else:
            raise NotImplementedError

        tau = torch.tensor(tau)
        tau = pad(tau)

        self.register_buffer("tau", tau, persistent=False)

    def reverse_process(self, model, x_t, t):
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

        tau_t = self.tau[t]
        tau_t_minus_one = self.tau[t - 1]

        alpha_bar_t_minus_one = self.alpha_bar[tau_t_minus_one]
        alpha_bar_t = self.alpha_bar[tau_t]

        noise_estimate = model(x_t, tau_t)

        predicted_x_0 = (
            x_t - torch.sqrt(1 - alpha_bar_t) * noise_estimate
        ) / torch.sqrt(alpha_bar_t)

        direction_pointing_to_x_t = (
            torch.sqrt(1 - alpha_bar_t_minus_one) * noise_estimate
        )

        x_t_minus_one = (
            torch.sqrt(alpha_bar_t_minus_one) * predicted_x_0
            + direction_pointing_to_x_t
        )

        return x_t_minus_one

    def sample(self, model, x_t, t):
        r"""Sample from :math:`p_\theta(x_{t-1}|x_t)`

        Args:
            model (nn.Module): model for estimating noise
            x_t (torch.Tensor): image of shape :math:`(N, C, H, W)`
            t (int): starting :math:`t` to sample from

        Returns:
            (torch.Tensor): generated sample of shape :math:`(N, C, H, W)`
        """
        return self.reverse_process(model, x_t, self.tau[t])
