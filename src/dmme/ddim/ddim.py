import torch
from torch import nn, Tensor

from dmme.ddpm import ForwardProcess, pad, linear_schedule


class DDIMReverseProcess(ForwardProcess):
    sigma: Tensor
    tau: Tensor

    def __init__(self, beta: Tensor, tau: Tensor, eta: float) -> None:
        beta = beta[tau]
        super().__init__(beta)

        sigma = (
            eta
            * torch.sqrt((1 - self.alpha_bar[:-1]) / (1 - self.alpha_bar[1:]))
            * torch.sqrt(1 - self.alpha_bar[1:] / self.alpha_bar[:-1])
        )
        sigma = pad(sigma)

        self.register_buffer("sigma", sigma, persistent=False)
        self.register_buffer("tau", tau, persistent=False)

    def forward(self, model, x_t, t, noise):
        alpha_bar_t_minus_one = self.alpha_bar[t - 1]
        alpha_bar_t = self.alpha_bar[t]
        sigma_t = self.sigma[t]

        noise_estimate = model(x_t, self.tau[t])

        x_t_minus_one = (
            torch.sqrt(alpha_bar_t_minus_one)
            * (x_t - torch.sqrt(1 - alpha_bar_t) * noise_estimate)
            / torch.sqrt(alpha_bar_t)
            + torch.sqrt(1 - alpha_bar_t_minus_one - sigma_t**2) * noise_estimate
            + sigma_t * noise
        )

        return x_t_minus_one


class DDIMSampler(nn.Module):
    def __init__(
        self,
        beta,
        timesteps: int = 50,
        tau_schedule: str = "quadratic",
        eta: float = 0.0,
    ):
        super().__init__()
        full_timesteps = beta.size(0) - 1

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

        self.reverse_process = DDIMReverseProcess(beta, tau, eta)

    def forward(self, model, x_t, t, noise):
        return self.reverse_process(model, x_t, t, noise)
