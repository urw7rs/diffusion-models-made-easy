import jax.numpy as jnp
from jax import random

from jax import jit

from dmme import ddpm


def test_state(key):
    hparams = ddpm.train.HyperParams()
    state = ddpm.train.create_state(key, hparams)


def test_ddpm_train_step(key):
    hparams = ddpm.train.HyperParams()
    state = ddpm.train.create_state(key, hparams)
    # TODO: set dynamic_scale to None in create_state arguments
    state.replace(dynamic_scale=None)

    train_step_jitted = jit(ddpm.train.step)

    key, subkey = random.split(key)
    x = random.normal(
        subkey,
        shape=(hparams.batch_size, hparams.height, hparams.width, hparams.channels),
    )
    loss, state = train_step_jitted(state, x)


def test_ddpm_train_step_mixed_precision(key):
    hparams = ddpm.train.HyperParams()
    state = ddpm.train.create_state(key, hparams)

    train_step_jitted = jit(ddpm.train.step)

    key, subkey = random.split(key)
    x = random.normal(
        subkey,
        shape=(hparams.batch_size, hparams.height, hparams.width, hparams.channels),
    )
    loss, state = train_step_jitted(state, x)


def test_ddpm_sampling(key):
    hparams = ddpm.train.HyperParams(batch_size=4)
    state = ddpm.train.create_state(key, hparams)

    sample_step_jitted = jit(ddpm.sample.step)

    key, subkey = random.split(key)

    x_t = random.normal(
        subkey,
        shape=(hparams.batch_size, hparams.height, hparams.width, hparams.channels),
    )

    key, subkey = random.split(key)

    leading_dims = x_t.shape[:-3]
    timestep = random.randint(
        subkey,
        shape=leading_dims,
        minval=1,
        maxval=hparams.timesteps,
    )
    denoised_x = sample_step_jitted(state, x_t, timestep, subkey)

    timesteps = jnp.ones_like(timestep)
    denoised_x = sample_step_jitted(state, x_t, timestep, subkey)
