import time

import numpy as np

from jax import jit
from jax import random
import jax.numpy as jnp

from flax.training.dynamic_scale import DynamicScale

import optax

from torch.utils import data

from torchvision import datasets
import torchvision.transforms as TF

from dmme import ddpm

batch_size = 128
height = width = 32
channels = 3
timesteps = 1000

learning_rate = 2e-4
grad_clip_norm = 1.0
ema_decay = 0.999
warmup_steps = 5000
train_iterations = 800_000


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class ToNumpy(object):
    def __init__(self, dtype) -> None:
        self.dtype = dtype

    def __call__(self, pic):
        return np.array(pic, dtype=self.dtype)


def norm(x):
    return (x - 0.5) * 2


def create_train_state(key):
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
    del state

    learning_rate_schedule = optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=learning_rate / warmup_steps,
                end_value=learning_rate,
                transition_steps=warmup_steps,
            ),
        ],
        [warmup_steps],
    )

    return ddpm.TrainState.create(
        apply_fn=model.apply,
        params=params,
        dropout_key=dropout_key,
        timestep_key=timestep_key,
        noise_key=noise_key,
        tx=optax.chain(
            optax.clip_by_global_norm(grad_clip_norm),
            optax.adam(learning_rate=learning_rate_schedule),
            optax.ema(decay=ema_decay),
        ),
        dynamic_scale=DynamicScale(growth_factor=10, growth_interval=1),
    )


def main(seed):
    key = random.PRNGKey(seed)
    state = create_train_state(key)

    schedule = ddpm.LinearSchedule.create(timesteps)

    train_step_jitted = jit(ddpm.train_step)

    train_set = datasets.CIFAR10(
        root=".",
        train=True,
        transform=TF.Compose([TF.RandomHorizontalFlip(), ToNumpy(jnp.float16), norm]),
        download=True,
    )

    dataloader = NumpyLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    t0 = time.perf_counter()
    step = 1
    while step < train_iterations:
        for x, _ in dataloader:
            if step > train_iterations:
                break

            timestep = random.randint(
                random.fold_in(state.timestep_key, state.step),
                shape=(batch_size,),
                minval=1,
                maxval=timesteps,
            )

            noise = random.normal(
                random.fold_in(state.noise_key, state.step), shape=x.shape
            )

            loss, state = train_step_jitted(
                state,
                random.fold_in(state.dropout_key, state.step),
                schedule.alpha_bar[timestep],
                x,
                timestep,
                noise,
            )

            if step % 100 == 0:
                t = time.perf_counter() - t0
                print(f"step: {step} {t} seconds, loss: {loss} {100 / t} it/s")
                t0 = time.perf_counter()

            step += 1


if __name__ == "__main__":
    main(seed=0)
