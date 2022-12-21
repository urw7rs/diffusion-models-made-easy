import torch
from torch import nn
import torch.nn.functional as F

import einops

from dmme.common import gaussian_like, uniform_int


class DDPMSampler(nn.Module):
    """Wrapper for computing forward and reverse processes,
    sampling data, and computing loss for DDPM

    Paper: https://arxiv.org/abs/2006.11239

    Code: https://github.com/hojonathanho/diffusion

    Args:
        model (nn.Module): model
        timesteps (int): diffusion timesteps
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
    ):
        super().__init__()

        self.model = model

        self.timesteps = timesteps

        beta = self.noise_schedule()

        if beta is not None:
            self.register_alphas(beta)

    def forward(self, x_t, t):
        r"""Predicts the noise given :math:`x_t` and :math:`t`

        Applies forward to the internal model

        Expects :math:`x_t` to have shape :math:`(N, C, H, W)`

        Args:
            x_t (torch.Tensor): image
            t (int): :math:`t` in :math:`\bold{x}_t`
        """

        return self.model(x_t, t)

    def forward_process(self, x_0, t, noise=None):
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

        t_index = t - 1

        if noise is None:
            noise = gaussian_like(x_0)

        x_t = (
            self.sqrt_alpha_bar[t_index] * x_0
            + self.sqrt_one_minus_alpha_bar[t_index] * noise
        )

        return x_t

    def noise_schedule(self):
        r"""Noise Schedule for DDPM

        DDPM sets :math:`T = 1000` and linearly increases
        :math:`\beta_t` from :math:`10^{-4}` to :math:`0.02`

        Returns:
            (torch.Tensor):
                :math:`\beta_1, \, ... \, ,\beta_T` as a tensor of shape :math:`(T,)`
        """

        return linear_schedule(timesteps=self.timesteps)

    def register_alphas(self, beta):
        r"""Caches :math:`\alpha_t` used in the forward and reverse process

        :math:`\alpha_t` is constant so we register them in `nn.Module`'s buffers

        Args:
            beta (torch.Tensor):
                beta values to use to compute alphas, a tensor of shape :math:`(T,)`
        """

        alpha = 1.0 - beta

        alpha_bar = torch.cumprod(alpha, dim=0)

        sqrt_alpha_bar = einops.rearrange(torch.sqrt(alpha_bar), "t -> t 1 1 1")
        sqrt_one_minus_alpha_bar = einops.rearrange(
            torch.sqrt(1 - alpha_bar), "t -> t 1 1 1"
        )

        one_over_sqrt_alpha = 1 / torch.sqrt(alpha)
        beta_over_sqrt_one_minus_alpha_bar = beta / torch.sqrt(1 - alpha_bar)

        sigma = torch.sqrt(beta)

        self.register_buffer("beta", beta)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_bar)
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_bar)
        self.register_buffer("one_over_sqrt_alpha", one_over_sqrt_alpha)
        self.register_buffer(
            "beta_over_sqrt_one_minus_alpha_bar", beta_over_sqrt_one_minus_alpha_bar
        )
        self.register_buffer("sigma", sigma)
        self.register_buffer("t", torch.arange(1, self.timesteps)[:, None])

    def reverse_process(self, x_t, t, noise=None):
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

        t_index = t - 1

        t = torch.tensor([t], device=x_t.device).float()

        if noise is None:
            noise = gaussian_like(x_t)

        x_t_minus_one = (
            self.one_over_sqrt_alpha[t_index]
            * (
                x_t
                - (
                    self.beta_over_sqrt_one_minus_alpha_bar[t_index]
                    * self.model(x_t, t)
                )
            )
            + self.sigma[t_index] * noise
        )

        return x_t_minus_one

    def compute_loss(self, x_0, t=None, noise=None):
        r"""Computes the loss

        :math:`L_\text{simple} = \mathbb{E}_{\bold{x}_0\sim q(\bold{x}_0),
        \epsilon\sim\mathcal{N}(\bold{0},\bold{I}),
        t\sim\mathcal{U}(1,T)}
        \left[\|\epsilon-\epsilon_\theta(\bold{x}_t, t) \|^2\right]`

        Args:
            x_0 (torch.Tensor): :math:`x_0`
            t (int, optional): sampled :math:`t`
            noise (torch.Tensor, optional): sampled :math:`\epsilon`
        """

        if t is None:
            batch_size = x_0.size(0)
            t = uniform_int(0, self.timesteps, batch_size, device=x_0.device)

        if noise is None:
            noise = gaussian_like(x_0)

        noisy_x = self.forward_process(x_0, t, noise)

        noise_estimate = self.model(noisy_x, t)
        loss = F.mse_loss(noise, noise_estimate)
        return loss

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

        if t == 1:
            x_t = self.reverse_process(x_t, 1, noise=0)
        else:
            x_t = self.reverse_process(x_t, t, noise)

        return x_t


def linear_schedule(timesteps, start=0.0001, end=0.02):
    r"""constants increasing linearly from :math:`10^{-4}` to :math:`0.02`

    Args:
        timesteps (int): total timesteps
        start (float): starting value, defaults to 0.0001
        end (float): end value, defaults to 0.02
    """
    return torch.linspace(start, end, timesteps)
