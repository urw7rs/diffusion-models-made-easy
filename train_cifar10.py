from tqdm import tqdm

import numpy as np

from jax import jit
from jax import random
import jax.numpy as jnp

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


def main(seed):
    key = random.PRNGKey(seed)
    hparams = ddpm.train.HyperParams()
    state = ddpm.train.create_state(key, hparams)

    train_step_jitted = jit(ddpm.train.step)

    train_set = datasets.CIFAR10(
        root=".",
        train=True,
        transform=TF.Compose(
            [
                TF.RandomHorizontalFlip(),
                ToNumpy(jnp.float16),
                ddpm.data.norm,
            ]
        ),
        download=True,
    )

    dataloader = NumpyLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    step = 0
    while step < train_iterations:
        for x, _ in tqdm(dataloader):
            if step == train_iterations:
                break
            step += 1

            loss, state = train_step_jitted(state, x)

            if step % 50 == 0:
                print(f"loss: {loss} iter: {step}")


if __name__ == "__main__":
    main(seed=0)
