import functools

from dataclasses import dataclass

import jax
from jax import random
import jax.numpy as jnp

from flax.training import train_state
from flax.training.dynamic_scale import DynamicScale

import optax

from .models import UNet

from .schedule import Linear

Array = jax.Array


class TrainState(train_state.TrainState):
    dropout_key: jax.random.KeyArray
    timestep_key: jax.random.KeyArray
    noise_key: jax.random.KeyArray
    dynamic_scale: DynamicScale
    schedule: Linear


@dataclass
class HyperParams:
    batch_size = 128
    height = width = 32
    channels = 3
    timesteps = 1000

    learning_rate = 2e-4
    grad_clip_norm = 1.0
    ema_decay = 0.999
    warmup_steps = 5000
    train_iterations = 800_000


def create_state(key, hparams: HyperParams):
    params_key, timestep_key, noise_key, dropout_key = random.split(key, num=4)

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
        dropout_key=dropout_key,
        timestep_key=timestep_key,
        noise_key=noise_key,
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

    leading_dims = image.shape[:-3]
    timestep = random.randint(
        random.fold_in(state.timestep_key, state.step),
        shape=leading_dims,
        minval=1,
        maxval=schedule.timesteps,
    )

    noise = random.normal(
        random.fold_in(state.noise_key, state.step), shape=image.shape
    )

    dropout_key = random.fold_in(state.dropout_key, state.step)

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
