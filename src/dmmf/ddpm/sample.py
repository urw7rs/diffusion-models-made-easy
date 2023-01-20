from typing import Any, Callable

import jax
from jax import random
import jax.numpy as jnp

from flax import struct

from .train import HyperParams
from .schedule import Linear


@struct.dataclass
class SampleState(struct.PyTreeNode):
    step: int
    apply_fn: Callable
    params: Any
    hparams: HyperParams
    schedule: Any
    key: random.KeyArray

    @classmethod
    def create(cls, apply_fn, params, hparams, key):
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            hparams=hparams,
            schedule=Linear.create(hparams.timesteps),
            key=key,
        )


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

    def sample():
        stddev = jnp.sqrt(beta_t)
        return mean + stddev * noise

    return jax.lax.cond(noise == 0, sample, lambda x: x)


def step(state: SampleState, x_t, timestep):
    schedule = state.schedule

    noise_in_x_t = state.apply_fn(
        {"params": state.params}, x_t, timestep, training=False
    )

    alpha_bar_t = schedule.alpha_bar[timestep]
    beta_t = schedule.beta[timestep]

    def sample_noise():
        return random.normal(state.key, shape=x_t.shape)

    def zero_noise():
        return jnp.zeros_like(x_t)

    noise = jax.lax.cond(timestep > 1, sample_noise, zero_noise)

    x_t_minus_one = reverse_process(alpha_bar_t, beta_t, x_t, noise, noise_in_x_t)
    return x_t_minus_one
