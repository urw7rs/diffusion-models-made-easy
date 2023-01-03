from collections import namedtuple

import torch
from torch import nn

from einops import rearrange

import dmme
import dmme.equations as eq

from .ddpm import DDPM

NoiseVariance = namedtuple("NoiseVariance", ["noise", "variance"])


class IDDPM(DDPM):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        offset=0.008,
        loss_type="hybrid",
        gamma=0.001,
        schedule="cosine",
    ) -> None:
        super().__init__(model, timesteps)

        self.loss_type = loss_type
        self.gamma = gamma

        if schedule == "cosine":
            alpha_bar = eq.iddpm.cosine_schedule(timesteps, offset)
            alpha_bar = rearrange(alpha_bar, "t -> t 1 1 1")

            # clip values to prevent singularities at the end of the diffusion near t = T
            beta = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0, 0.999)
            beta = dmme.pad(beta)

            alpha = 1 - beta

            self.register_buffer("beta", beta, persistent=False)
            self.register_buffer("alpha", alpha, persistent=False)
            self.register_buffer("alpha_bar", alpha_bar, persistent=False)
        elif schedule != "linear":
            raise NotImplementedError

    def training_step(self, x_0):
        r"""Computes hybrid loss for improved DDPM

        Args:
            x_0 (torch.Tensor): sample image to add noise and denoise for training

        Returns:
            (torch.Tensor): loss, :math:`L_\text{simple}`
        """

        batch_size = x_0.size(0)

        t = dmme.uniform_int(
            1,
            self.timesteps,
            batch_size,
            device=x_0.device,
        )

        noise = dmme.gaussian_like(x_0)

        alpha_bar_t = self.alpha_bar[t]

        x_t = eq.ddpm.forward_process(x_0, alpha_bar_t, noise)

        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t_minus_one = self.alpha_bar[t - 1]

        model_output = self.forward_model(
            x_t, t, beta_t, alpha_bar_t, alpha_bar_t_minus_one
        )

        vlb_loss = 0
        if self.loss_type == "hybrid" or self.loss_type == "vlb":

            vlb_loss = eq.iddpm.loss_vlb(
                model_output.noise,
                model_output.variance,
                x_t,
                t,
                x_0,
                beta_t,
                alpha_t,
                alpha_bar_t,
                alpha_bar_t_minus_one,
            )

            if self.loss_type == "vlb":
                return vlb_loss

        if self.loss_type == "hybrid":
            simple_loss = eq.ddpm.simple_loss(noise, model_output.noise)

            loss = simple_loss + self.gamma * vlb_loss
            return loss

    def sampling_step(self, x_t, t):
        r"""Denoise image by sampling from :math:`p_\theta(x_{t-1}|x_t)`

        Args:
            model (nn.Module): model for estimating noise
            x_t (torch.Tensor): image of shape :math:`(N, C, H, W)`
            t (torch.Tensor): starting :math:`t` to sample from, a tensor of shape :math:`(N,)`

        Returns:
            (torch.Tensor): denoised image of shape :math:`(N, C, H, W)`
        """
        noise = dmme.gaussian_like(x_t)

        (idx,) = torch.where(t == 1)
        noise[idx] = 0

        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]

        model_output = self.forward_model(
            x_t, t, beta_t, alpha_bar_t, self.alpha_bar[t - 1]
        )

        x_t = eq.ddpm.reverse_process(
            x_t,
            beta_t,
            alpha_t,
            alpha_bar_t,
            model_output.noise,
            variance=model_output.variance,
            noise=model_output.noise,
        )
        return x_t

    def forward_model(self, x_t, t, beta_t, alpha_bar_t, alpha_bar_t_minus_one):
        model_output = self.model(x_t, t)
        noise_in_x_t, v = model_output.chunk(2, dim=1)

        beta_tilde_t = (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t) * beta_t
        variance = eq.iddpm.interpolate_variance(v, beta_t, beta_tilde_t)
        return NoiseVariance(noise_in_x_t, variance)
