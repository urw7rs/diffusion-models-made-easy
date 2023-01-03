import torch

from torch.distributions import Normal


from collections import namedtuple

Gaussian = namedtuple("Gaussian", ["mean", "variance"])


def gaussian_kl_divergence(q_mean, q_variance, p_mean, p_variance):
    return (
        torch.log(p_variance / q_variance) / 2
        + (q_variance + (q_mean - p_mean) ** 2) / (2 * p_variance)
        - 1 / 2
    )


def discrete_nll_loss(x_0, mean, variance):
    m = Normal(mean, torch.sqrt(variance))

    delta_minus = torch.where(x_0 > -0.999, x_0 - 1 / 255, x_0)
    delta_plus = torch.where(x_0 < 0.999, x_0 + 1 / 255, x_0)

    prob = m.cdf(delta_plus) - m.cdf(delta_minus)
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

    return Gaussian(mean, variance)


def interpolate_variance(v, beta_t, beta_tilde_t):
    return torch.exp(v * torch.log(beta_t) + (1 - v) * torch.log(beta_tilde_t + 1e-7))


def loss_t_minus_one(
    mean, variance, x_t, x_0, beta_t, alpha_t, alpha_bar_t, alpha_bar_t_minus_one
):
    q = forward_posterior(x_t, x_0, beta_t, alpha_t, alpha_bar_t, alpha_bar_t_minus_one)
    loss_t_minus_one = gaussian_kl_divergence(q.mean, q.variance, mean, variance)
    return loss_t_minus_one


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

    loss = []
    if (t != 1).any():
        mask = t != 1
        loss.append(
            loss_t_minus_one(
                mean[mask],
                variance[mask],
                x_t[mask],
                x_0[mask],
                beta_t[mask],
                alpha_t[mask],
                alpha_bar_t[mask],
                alpha_bar_t_minus_one[mask],
            )
        )

    if (t == 1).any():
        mask = t == 1
        loss.append(discrete_nll_loss(x_0[mask], mean[mask], variance[mask]))

    vlb_loss = torch.cat(loss, dim=0).mean()
    return vlb_loss
