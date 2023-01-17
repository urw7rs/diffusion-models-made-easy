import functools

import jax
import jax.numpy as jnp

from flax import struct
from flax.training import train_state
from flax.training.dynamic_scale import DynamicScale

import optax

from einops import rearrange

Array = jax.Array


class TrainState(train_state.TrainState):
    dropout_key: jax.random.KeyArray
    timestep_key: jax.random.KeyArray
    noise_key: jax.random.KeyArray
    dynamic_scale: DynamicScale


@struct.dataclass
class LinearSchedule:
    r"""constants increasing linearly from :math:`10^{-4}` to :math:`0.02`

    Args:
        timesteps: total timesteps
        start: starting value, defaults to 0.0001
        end: end value, defaults to 0.02

    Returns:
        a 1d tensor representing :math:`\beta_t` indexed by :math:`t`
    """
    beta: Array
    alpha: Array
    alpha_bar: Array
    timesteps: int

    @classmethod
    def create(
        cls, timesteps: int, start: float = 0.0001, end: float = 0.02, dtype=None
    ):
        beta = jnp.linspace(start, end, num=timesteps, dtype=dtype)
        beta = jnp.pad(beta, pad_width=(1, 0))
        beta = rearrange(beta, "t -> t 1 1 1")
        alpha = 1 - beta
        alpha_bar = jnp.cumprod(alpha, axis=0)
        timesteps = beta.shape[0] - 1
        return cls(beta, alpha, alpha_bar, timesteps)


def forward_process(alpha_bar_t, x, noise):
    r"""Forward Process, :math:`q(x_t|x_{t-1})`

    Args:
        x: image of shape :math:`(N, C, H, W)`
        noise: noise sampled from standard normal distribution with the same shape as the image
        alpha_bar_t: :math:`\bar\alpha_t` of shape :math:`(N, 1, 1, *)`

    Returns:
        gaussian transition distirbution :math:`q(x_t|x_{t-1})`
    """
    mean = jnp.sqrt(alpha_bar_t) * x
    stddev = jnp.sqrt(1 - alpha_bar_t)
    return mean + stddev * noise


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


def simple_loss(params, state, dropout_key, alpha_bar_t, image, timestep, noise):
    x_t = forward_process(alpha_bar_t, image, noise)

    noise_in_x_t = state.apply_fn(
        {"params": params},
        x_t,
        timestep,
        training=True,
        rngs={"dropout": dropout_key},
    )

    loss = optax.l2_loss(predictions=noise_in_x_t, targets=noise)
    return jnp.mean(loss)


def train_step(state: TrainState, dropout_key, alpha_bar_t, image, timestep, noise):
    def loss_fn(params):
        return jnp.mean(
            simple_loss(params, state, dropout_key, alpha_bar_t, image, timestep, noise)
        )

    if state.dynamic_scale:
        loss_grad_fn = state.dynamic_scale.value_and_grad(loss_fn)
        dynamic_scale, is_fin, loss, grads = loss_grad_fn(state.params)

        new_state = state.apply_gradients(grads=grads)

        select_fn = functools.partial(jnp.where, is_fin)

        new_state = new_state.replace(
            opt_state=jax.tree_util.tree_map(
                select_fn,
                new_state.opt_state,
                state.opt_state,
            ),
            params=jax.tree_util.tree_map(select_fn, new_state.params, state.params),
            dynamic_scale=dynamic_scale,
        )
    else:
        loss_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_grad_fn(state.params)

        new_state = state.apply_gradients(grads=grads)
    return loss, new_state
