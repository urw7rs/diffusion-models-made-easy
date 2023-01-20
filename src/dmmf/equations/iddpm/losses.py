import torch

from torch.distributions import Normal

from .. import ddpm


def discrete_nll_loss(x_0, p: Normal):
    r"""Calculate the discrete log likelihood of :math:`p_\theta(x_0|x_1)`

    Args:
        x_0 (torch.Tensor): image from dataset
        p (torch.distributions.Normal): Gaussian distribution for marginalizing
    """
    F_delta_plus = torch.where(x_0 < 1, p.cdf(x_0 + 1 / 255), torch.ones_like(x_0))
    F_delta_minus = torch.where(x_0 > -1, p.cdf(x_0 - 1 / 255), torch.zeros_like(x_0))

    prob = F_delta_plus - F_delta_minus
    return -torch.log(prob.clamp(1e-12))


def true_reverse_process(x_t, x_0, beta_t, alpha_t, alpha_bar_t, alpha_bar_t_minus_one):
    mean = (
        torch.sqrt(alpha_bar_t_minus_one) * beta_t / (1 - alpha_bar_t) * x_0
        + torch.sqrt(alpha_t) * (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t) * x_t
    )

    variance = (1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t) * beta_t
    std = torch.sqrt(variance)

    return Normal(mean, std)


def interpolate_variance(v, beta_t, beta_tilde_t):
    return torch.exp(
        v * torch.log(beta_t) + (1 - v) * torch.log(beta_tilde_t.clamp(1e-12))
    )


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
    # apply stop-gradient
    all_p = ddpm.reverse_process(
        x_t, beta_t, alpha_t, alpha_bar_t, noise_in_x_t.clone().detach(), variance
    )

    vlb_loss = []

    if (t == 1).any():
        mask = t == 1

        p = Normal(all_p.mean[mask], all_p.stddev[mask])

        loss = discrete_nll_loss(x_0[mask], p)

        vlb_loss.append(loss)

    if (t != 1).any():
        mask = t != 1

        q = true_reverse_process(
            x_t[mask],
            x_0[mask],
            beta_t[mask],
            alpha_t[mask],
            alpha_bar_t[mask],
            alpha_bar_t_minus_one[mask],
        )
        p = Normal(all_p.mean[mask], all_p.stddev[mask])

        # vlb uses the reverse kl divergence
        loss = torch.distributions.kl_divergence(q, p)

        vlb_loss.append(loss)

    if len(vlb_loss) > 1:
        vlb_loss = torch.cat(vlb_loss, dim=0)
    else:
        vlb_loss = vlb_loss[0]

    return vlb_loss.mean()
