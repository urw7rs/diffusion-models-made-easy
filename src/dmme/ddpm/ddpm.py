import torch
from torch import nn, Tensor

from einops import rearrange


class Process(nn.Module):
    """Base class for `ForwardProcess` and `ReverseProcess`, Use `__init__()` for fine grained control, `build()` for simplicity"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):
        """forward for `nn.Module`"""
        raise NotImplementedError

    @staticmethod
    def build(*args, **kwargs):
        """Easy to use constructor"""
        raise NotImplementedError


class ForwardProcess(Process):
    alpha_bar: Tensor

    def __init__(self, beta: Tensor) -> None:
        super().__init__()

        beta = rearrange(beta, "t -> t 1 1 1")

        alpha = alpha_from_beta(beta)
        alpha_bar = alpha_bar_from_alpha(alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, x_0: Tensor, t: Tensor, noise: Tensor):
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

    @staticmethod
    def build(timesteps):
        beta = linear_schedule(timesteps)
        return ForwardProcess(beta)


class ReverseProcess(Process):
    beta: Tensor
    alpha: Tensor
    alpha_bar: Tensor
    sigma: Tensor

    def __init__(self, beta: Tensor, sigma: Tensor) -> None:
        super().__init__()

        beta = rearrange(beta, "t -> t 1 1 1")
        sigma = rearrange(sigma, "t -> t 1 1 1")

        alpha = alpha_from_beta(beta)
        alpha_bar = alpha_bar_from_alpha(alpha)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sigma", sigma)

    def forward(self, x_t, t, noise_estimate, noise):
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

        x_t_minus_one = (
            x_t / torch.sqrt(alpha_t)
            - (beta_t / torch.sqrt(alpha_t * (1 - alpha_bar_t)) * noise_estimate)
            + sigma_t * noise
        )

        return x_t_minus_one

    @staticmethod
    def build(timesteps):
        beta = linear_schedule(timesteps)
        return ReverseProcess(beta=beta, sigma=beta)


class DDPMSampler(nn.Module):
    """Wrapper for computing forward and reverse processes,
    sampling data, and computing loss for DDPM

    Paper: https://arxiv.org/abs/2006.11239

    Code: https://github.com/hojonathanho/diffusion

    Args:
        model (nn.Module): model for estimating noise
        timesteps (int): diffusion timesteps
    """

    def __init__(self, reverse_process: ReverseProcess):
        super().__init__()

        self.reverse_process = reverse_process

        t = torch.arange(0, self.reverse_process.beta.size(0) + 1)
        self.register_buffer("t", t)

    def forward(self, x_t, t, noise_estimate, noise):
        r"""Generate Samples

        Iteratively sample from :math:`p_\theta(x_{t-1}|x_t)` starting from :math:`x_T`

        Args:
            model (nn.Module): model for estimating noise
            x_t (Tuple[int, int, int]): image shape
            t (Tensor): timestep :math:`t` to sample from

        Returns:
            (torch.Tensor): sample from :math:`p_\theta(x_{t-1}|x_t)` starting from :math:`x_T`
        """

        if t == 1:
            x_t = self.reverse_process(x_t, t, noise_estimate, 0)
        else:
            x_t = self.reverse_process(x_t, t, noise_estimate, noise)

        return x_t


def alpha_from_beta(beta: Tensor) -> Tensor:
    r"""returns alpha from beta

    :math:`\alpha_t = 1 - \beta_t`
    """
    return 1 - beta


def alpha_bar_from_alpha(alpha: Tensor) -> Tensor:
    r"""returns alpha_bar from alpha

    :math:`\bar\alpha_t = \prod_{s=1}^t\alpha_s`
    """

    alpha_bar = pad(torch.cumprod(alpha[1:], dim=0))
    return alpha_bar


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
