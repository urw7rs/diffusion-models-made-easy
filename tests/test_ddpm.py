from jax import jit
from jax import random

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


def test_ddpm_sampling():
    pass


def test_full_step():
    pass
