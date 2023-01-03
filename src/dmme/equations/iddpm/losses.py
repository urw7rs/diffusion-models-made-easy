import time

import torch

from torch.distributions import Normal


def discrete_nll_loss(x_0, p: Normal):
    delta_minus = torch.where(x_0 > -0.999, x_0 - 1 / 255, x_0)
    delta_plus = torch.where(x_0 < 0.999, x_0 + 1 / 255, x_0)

    prob = p.cdf(delta_plus) - p.cdf(delta_minus)
    return -torch.log(prob)


def reverse_dist_mean(x_t, noise_in_x_t, alpha_bar_t):
    return (
        1 / torch.sqrt(alpha_bar_t) * (x_t - torch.sqrt(1 - alpha_bar_t) * noise_in_x_t)
    )


def forward_posterior(x_t, x_0, beta_t, alpha_t, alpha_bar_t, alpha_bar_t_minus_one):
    mean = (
        torch.sqrt(alpha_bar_t_minus_one) * beta_t / (1 - alpha_bar_t) * x_0
        + torch.sqrt(alpha_t) * (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t) * x_t
    )

    variance = (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t) * beta_t

    return Normal(mean, torch.sqrt(variance))


def interpolate_variance(v, beta_t, beta_tilde_t):
    return torch.exp(v * torch.log(beta_t) + (1 - v) * torch.log(beta_tilde_t + 1e-7))


def loss_t_minus_one(
    mean, variance, x_t, x_0, beta_t, alpha_t, alpha_bar_t, alpha_bar_t_minus_one
):
    q = forward_posterior(x_t, x_0, beta_t, alpha_t, alpha_bar_t, alpha_bar_t_minus_one)
    p = Normal(mean, torch.sqrt(variance))

    return torch.distributions.kl_divergence(q, p)


def loss_vlb(
    noise_in_x_t,
    variance,
    x_t,
    t,
    x_0,
    beta_t,
    alpha_t,
    alpha_bar_t,
    alpha_bar_t_minus_one,
):

    mean = reverse_dist_mean(x_t, noise_in_x_t, alpha_bar_t)

    vlb_loss = []

    if (t != 1).any():
        mask = t != 1

        q = forward_posterior(
            x_t[mask],
            x_0[mask],
            beta_t[mask],
            alpha_t[mask],
            alpha_bar_t[mask],
            alpha_bar_t_minus_one[mask],
        )
        p = Normal(mean[mask], torch.sqrt(variance)[mask])

        loss = torch.distributions.kl_divergence(q, p)
        vlb_loss.append(loss)

    if (t == 1).any():
        mask = t == 1

        p = Normal(mean[mask], torch.sqrt(variance)[mask])

        loss = discrete_nll_loss(x_0[mask], p)
        vlb_loss.append(loss)

    if len(vlb_loss) > 1:
        vlb_loss = torch.cat(vlb_loss, dim=0)
    else:
        vlb_loss = vlb_loss[0]

    return vlb_loss.mean()
