import jax.numpy as jnp


def reverse_process(alpha_bar_t, beta_t, x_t, noise, noise_in_x_t):
    r"""Reverse Denoising Process, :math:`p_\theta(x_{t-1}|x_t)`

    Args:
        x_t: :math:`\x_t` of shape :math:`(N, H, W, C)`
        beta_t: :math:`\beta_t` of shape :math:`(N, 1, 1, *)`
        alpha_t: :math:`\alpha_t` of shape :math:`(N, 1, 1, *)`
        alpha_bar_t: :math:`\bar\alpha_t` of shape :math:`(N, 1, 1, *)`
        noise_in_x_t: estimated noise in :math:`x_t` predicted by a neural network
        variance: variance of the reverse process, either learned or fixed
        noise: noise sampled from :math:`\mathcal{N}(0, I)`

    Returns:
        denoising distirbution :math:`q(x_t|x_{t-1})`
    """
    mean = (
        1
        / jnp.sqrt(alpha_bar_t)
        * (x_t - beta_t / jnp.sqrt(1 - alpha_bar_t) * noise_in_x_t)
    )
    stddev = jnp.sqrt(beta_t)
    return mean + stddev * noise


def step(state):
    pass
