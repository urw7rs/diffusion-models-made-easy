import time

from jax import jit
from jax import random
import jax.numpy as jnp

import optax

from dmme import ddpm

batch_size = 128
height = width = 32
channels = 3

learning_rate = 2e-4
timesteps = 1000
train_iterations = 800_000


def main(seed):
    key = random.PRNGKey(seed)
    params_key, timestep_key, noise_key, dropout_key = random.split(key, num=4)

    model = ddpm.UNet(
        in_channels=3,
        pos_dim=128,
        emb_dim=512,
        drop_rate=0.1,
        channels_per_depth=(128, 256, 256, 256),
        num_blocks=2,
        attention_depths=(2,),
    )

    dummy_x = jnp.empty((batch_size, height, width, channels))
    dummy_t = jnp.empty((batch_size,))

    variables = model.init(params_key, dummy_x, dummy_t, training=False)
    state, params = variables.pop("params")
    del variables

    state = ddpm.TrainState.create(
        apply_fn=model.apply,
        params=params,
        dropout_key=dropout_key,
        timestep_key=timestep_key,
        noise_key=noise_key,
        tx=optax.adam(learning_rate),
    )

    schedule = ddpm.LinearSchedule.create(timesteps)

    x = random.normal(key, shape=(batch_size, height, width, channels))

    train_step_jitted = jit(ddpm.train_step)

    t0 = time.perf_counter()
    for i in range(train_iterations):
        loss, state = train_step_jitted(
            state, schedule.alpha_bar, schedule.timesteps, x
        )

        if i % 100 == 0:
            t = time.perf_counter() - t0
            print(f"loss: {loss} {100 / t} it/s")
            t0 = time.perf_counter()


if __name__ == "__main__":
    main(seed=0)
