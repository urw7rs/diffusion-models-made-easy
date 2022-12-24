import torch
import torch.nn.functional as F

import einops

from dmme.common import gaussian_like

from dmme.ddpm import DDPMSampler


class DDIMSampler(DDPMSampler):
    """Wrapper for computing forward and reverse processes,
    sampling data, and computing loss for DDIM

    > Implements sampling from an implicit model that is trained with the same procedure as Denoising Diffusion Probabilistic Model, but costs much less time and compute if you want to sample from it

    Paper: https://arxiv.org/abs/2010.02502

    Code: https://github.com/ermongroup/ddim

    Args:
        model (nn.Module): model
        timesteps (int): diffusion timesteps
        sigma (float): :math:`sigma_t` wich controls characteristics of the generative process
    """

    def __init__(
        self,
        timesteps,
        sub_timesteps,
        tau_schedule="linear",
        eta=0.0,
    ):
        self.eta = eta

        tau_schedule = tau_schedule.lower()
        if tau_schedule == "lienar":
            c = timesteps // sub_timesteps

            self.tau = [round(c * i) for i in range(sub_timesteps + 1)]

        elif tau_schedule == "quadratic":
            c = timesteps // (sub_timesteps**2)

            self.tau = [c * i**2 for i in range(sub_timesteps + 1)]

        super().__init__(model, timesteps)

    def register_alphas(self, beta):
        r"""Caches :math:`\alpha_t` used in the forward and reverse process

        :math:`\alpha_t` is constant so we register them in `nn.Module`'s buffers

        Args:
            beta (torch.Tensor):
                beta values to use to compute alphas, a tensor of shape :math:`(T,)`
        """

        alpha = 1.0 - beta

        alpha_bar = torch.cumprod(alpha, dim=0)

        alpha_bar = F.pad(alpha_bar, (1, 0), value=0)

        sqrt_alpha_bar = einops.rearrange(torch.sqrt(alpha_bar), "t -> t 1 1 1")
        sqrt_one_minus_alpha_bar = einops.rearrange(
            torch.sqrt(1 - alpha_bar), "t -> t 1 1 1"
        )

        one_over_sqrt_alpha_bar = 1 / sqrt_alpha_bar

        sigma = (
            self.eta
            * torch.sqrt((1 - alpha_bar[:-1]) / (1 - alpha_bar[1:]))
            * torch.sqrt(1 - alpha_bar[1:] / alpha_bar[:-1])
        )

        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar)
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_bar)
        self.register_buffer("one_over_sqrt_alpha_bar", one_over_sqrt_alpha_bar)
        self.register_buffer("sigma", sigma)
        self.register_buffer("sigma_squared", sigma**2)

    def reverse_process(self, model, x_t, t, noise=None):
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
            x_t (torch.Tensor): x_t
            t (int): current timestep
            noise (torch.Tensor): noise
        """

        t_tensor = torch.tensor([t], device=x_t.device).float()

        if noise is None:
            noise = gaussian_like(x_t)

        noise_estimate = model(x_t, t_tensor)

        tau_i = self.tau[t]
        tau_i_minus_one = self.tau[t - 1]

        t_index = tau_i - 1

        x_t_minus_one = (
            self.sqrt_alpha_bar[tau_i_minus_one]
            * (
                (x_t - self.sqrt_alpha_bar[tau_i] * noise_estimate)
                * self.one_over_sqrt_alpha_bar[tau_i]
            )
            + (
                torch.sqrt(
                    1 - self.alpha_bar[tau_i_minus_one] - self.sigma_squared[t_index]
                )
                * noise_estimate
            )
            + self.sigma[t_index] * noise
        )

        return x_t_minus_one

    @torch.inference_mode()
    def sample(self, x_t, t, noise=None):
        r"""Generate Samples

        Iteratively sample from :math:`p_\theta(x_{t-1}|x_t)` starting from :math:`x_T`

        Args:
            x_t (Tuple[int, int, int]): image shape
            t (int): timestep :math:`t` to sample from

        Returns:
            (torch.Tensor): sample from :math:`p_\theta(x_{t-1}|x_t)` starting from :math:`x_T`
        """

        x_t = self.reverse_process(x_t, t, noise)

        return x_t
