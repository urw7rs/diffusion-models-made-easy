import torch
from torch import nn
from torch import Tensor

import einops


class DDPM(nn.Module):
    """Forward, Reverse, Sampling for DDPM

    Args:
        timesteps (int): total timesteps :math:`T`
    """

    beta: Tensor
    alpha: Tensor
    alpha_bar: Tensor
    sigma: Tensor

    def __init__(self, timesteps) -> None:
        super().__init__()

        beta = linear_schedule(timesteps)
        beta = einops.rearrange(beta, "t -> t 1 1 1")

        alpha = 1 - beta
        # alpha[0] = 1 so no problems here
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta, persistent=False)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)
        self.register_buffer("sigma", torch.sqrt(beta), persistent=False)

    def forward_process(self, x_0: Tensor, t: Tensor, noise: Tensor):
        r"""Forward Diffusion Process

        Samples :math:`x_t` from :math:`q(x_t|x_0)
        = \mathcal{N}(x_t;\sqrt{\bar\alpha_t}\bold{x}_0,(1-\bar\alpha_t)\bold{I})`

        Computes :math:`\bold{x}_t
        = \sqrt{\bar\alpha_t}\bold{x}_0 + \sqrt{1-\bar\alpha_t}\bold{I}`

        Args:
            x_0 (torch.Tensor): data to add noise to
            t (int): :math:`t` in :math:`x_t`
            noise (torch.Tensor, optional):
                :math:`\epsilon`, noise used in the forward process

        Returns:
            (torch.Tensor): :math:`\bold{x}_t \sim q(\bold{x}_t|\bold{x}_0)`
        """

        alpha_bar_t = self.alpha_bar[t]

        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        return x_t

    def reverse_process(self, model, x_t, t, noise):
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

        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        sigma_t = self.sigma[t]

        noise_estimate = model(x_t, t)

        x_t_minus_one = (
            1
            / torch.sqrt(alpha_t)
            * (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * noise_estimate)
            + sigma_t * noise
        )

        return x_t_minus_one

    def sample(self, model, x_t, t, noise):
        r"""Sample from :math:`p_\theta(x_{t-1}|x_t)`

        Args:
            model (nn.Module): model for estimating noise
            x_t (torch.Tensor): image of shape :math:`(N, C, H, W)`
            t (int): starting :math:`t` to sample from
            noise (torch.Tensor): noise to use for sampling, if `None` samples new noise

        Returns:
            (torch.Tensor): generated sample of shape :math:`(N, C, H, W)`
        """

        (idx,) = torch.where(t == 1)
        noise[idx] = 0

        x_t = self.reverse_process(model, x_t, t, noise)
        return x_t


def pad(x: Tensor, value: float = 0) -> Tensor:
    r"""pads tensor with 0 to match :math:`t` with tensor index"""

    ones = torch.ones_like(x[0:1])
    return torch.cat([ones * value, x], dim=0)


def linear_schedule(timesteps: int, start=0.0001, end=0.02) -> Tensor:
    r"""constants increasing linearly from :math:`10^{-4}` to :math:`0.02`

    Args:
        timesteps (int): total timesteps
        start (float): starting value, defaults to 0.0001
        end (float): end value, defaults to 0.02
    """

    beta = torch.linspace(start, end, timesteps)
    return pad(beta)
