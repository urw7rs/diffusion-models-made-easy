import functools

from dataclasses import dataclass

import jax
from jax import random
import jax.numpy as jnp

from flax.training import train_state
from flax.training.dynamic_scale import DynamicScale
from flax import struct

import optax

from .models import UNet

from .schedule import Linear

Array = jax.Array


class TrainState(train_state.TrainState):
    key: jax.random.KeyArray
    dynamic_scale: DynamicScale
    schedule: Linear


@struct.dataclass
class HyperParams:
    batch_size: int = 128
    height: int = 32
    width: int = 32
    channels: int = 3
    timesteps: int = 1000

    learning_rate: float = 2e-4
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    warmup_steps: int = 5000
    train_iterations: int = 800_000


def create_state(key, hparams: HyperParams):
    dropout_key, params_key = random.split(key)

    model = UNet(
        in_channels=3,
        pos_dim=128,
        emb_dim=512,
        drop_rate=0.1,
        channels_per_depth=(128, 256, 256, 256),
        num_blocks=2,
        attention_depths=(2,),
    )

    dummy_x = jnp.empty(
        (hparams.batch_size, hparams.height, hparams.width, hparams.channels)
    )
    dummy_t = jnp.empty((hparams.batch_size,))

    variables = model.init(params_key, dummy_x, dummy_t, training=False)
    state, params = variables.pop("params")
    del state

    learning_rate_schedule = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=hparams.learning_rate / hparams.warmup_steps,
                end_value=hparams.learning_rate,
                transition_steps=hparams.warmup_steps,
            ),
        ],
        [hparams.warmup_steps],
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        key=dropout_key,
        tx=optax.chain(
            optax.clip_by_global_norm(hparams.grad_clip_norm),
            optax.adam(learning_rate=learning_rate_schedule),
            optax.ema(decay=hparams.ema_decay),
        ),
        dynamic_scale=DynamicScale(growth_factor=10, growth_interval=1),
        schedule=Linear.create(hparams.timesteps),
    )


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


def step(state: TrainState, image):
    schedule = state.schedule

    key = random.fold_in(state.key, state.step)

    timestep_key, noise_key, dropout_key = random.split(key, num=3)

    leading_dims = image.shape[:-3]
    timestep = random.randint(
        timestep_key,
        shape=leading_dims,
        minval=1,
        maxval=schedule.timesteps,
    )

    noise = random.normal(noise_key, shape=image.shape)

    alpha_bar_t = schedule.alpha_bar[timestep]

    def loss_fn(params):
        return jnp.mean(
            simple_loss(params, state, dropout_key, alpha_bar_t, image, timestep, noise)
        )

    def mixed_precision():
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
        return loss, new_state

    def normal_precision():
        loss_grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = loss_grad_fn(state.params)

        new_state = state.apply_gradients(grads=grads)
        return loss, new_state

    loss, new_state = jax.lax.cond(
        state.dynamic_scale is None, mixed_precision, normal_precision
    )
    return loss, new_state
