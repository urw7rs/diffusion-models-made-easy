from typing import Tuple
import functools
import time

import jax
from jax import random
import jax.numpy as jnp

from flax import linen as nn

import optax

import einops
from einops.layers.flax import Rearrange

import requests

from flax.training import train_state


half = jnp.float16
full = jnp.float32


class TrainState(train_state.TrainState):
    key: jax.random.KeyArray


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


class CIFAR10:
    pass


class ResBlock(nn.Module):
    features: int
    emb_dim: int
    drop_rate: float
    with_attention: bool

    @nn.compact
    def __call__(self, x, t, training: bool):
        residual = x
        Conv3x3 = functools.partial(
            nn.Conv,
            features=self.features,
            kernel_size=(3, 3),
            padding="same",
            dtype=half,
        )

        x = nn.GroupNorm(dtype=full)(x)
        x = nn.swish(x)
        x = Conv3x3()(x)

        num_channels = x.shape[-1]
        condition = nn.Sequential(
            [nn.Dense(features=num_channels, dtype=half), Rearrange("b c -> b 1 1 c")]
        )(t)

        x = x + condition

        x = nn.GroupNorm(dtype=full)(x)
        x = nn.swish(x)

        if self.drop_rate > 0:
            x = nn.Dropout(
                rate=self.drop_rate,
                broadcast_dims=(-2, -1),
                deterministic=not training,
            )(x)

        x = Conv3x3()(x)

        if residual.shape != x.shape:
            num_channels = x.shape[-1]
            residual = nn.DenseGeneral(num_channels, axis=-1, batch_dims=(0,))(x)

        x = x + residual

        if self.with_attention:
            x = nn.GroupNorm(dtype=full)(x)
            residual = x

            x = nn.MultiHeadDotProductAttention(
                num_heads=1, qkv_features=self.features, dtype=half
            )(inputs_q=x, inputs_kv=x)

            x = x + residual

        return x


def embed(t, dim, dtype=None):
    half_dim = dim // 2
    embeddings = jnp.log(10000) / (half_dim - 1)
    embeddings = jnp.exp(jnp.arange(0, half_dim, dtype=dtype) * -embeddings)
    embeddings = embeddings[None, :]
    embeddings = t[:, None] * embeddings
    embeddings = jnp.concatenate(
        (jnp.sin(embeddings), jnp.cos(embeddings)),
        axis=-1,
    )
    return embeddings


class UNet(nn.Module):
    in_channels: int
    pos_dim: int
    emb_dim: int
    drop_rate: float
    channels_per_depth: Tuple[int, ...]
    num_blocks: int
    attention_depths: Tuple[int, ...]

    @nn.compact
    def __call__(self, x, t, training: bool):
        t = embed(t, self.pos_dim, dtype=full)
        t = nn.Dense(self.emb_dim)(t)
        t = nn.silu(t)
        t = nn.Dense(self.emb_dim)(t)
        t = nn.silu(t)

        # conv3x3 builder
        Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3), padding="same")

        # input conv
        x = Conv3x3(features=self.channels_per_depth[0])(x)
        feature_maps = [x]

        default_resblock = functools.partial(
            ResBlock, emb_dim=self.emb_dim, drop_rate=self.drop_rate
        )

        max_depth = len(self.channels_per_depth)
        # contract path
        for i, channels in enumerate(self.channels_per_depth):
            depth = i + 1
            with_attention = depth in self.attention_depths

            for _ in range(self.num_blocks):
                x = default_resblock(features=channels, with_attention=with_attention)(
                    x, t, training
                )
                feature_maps.append(x)

            if depth < max_depth:
                # downsample
                x = nn.Conv(
                    features=channels, kernel_size=(3, 3), strides=(2, 2), padding=1
                )(x)
                feature_maps.append(x)

        # middle
        x = default_resblock(features=self.channels_per_depth[-1], with_attention=True)(
            x, t, training
        )
        x = default_resblock(
            features=self.channels_per_depth[-1], with_attention=False
        )(x, t, training)

        # expand path
        for i, channels in enumerate(self.channels_per_depth[::-1]):
            depth = max_depth - i
            with_attention = depth in self.attention_depths

            for _ in range(self.num_blocks + 1):
                x = default_resblock(features=channels, with_attention=with_attention)(
                    jnp.concatenate((x, feature_maps.pop()), axis=-1), t, training
                )

            if depth > 1:
                # upsample
                B, H, W, C = x.shape

                x = jax.image.resize(x, shape=(B, H * 2, W * 2, C), method="nearest")
                x = Conv3x3(features=channels)(x)

        # output conv
        x = nn.GroupNorm(dtype=full)(x)
        x = nn.silu(x)
        x = Conv3x3(features=self.in_channels)(x)

        return x


batch_size = 128
height = 32
width = 32
channels = 3

learning_rate = 2e-4
num_steps = 800_000

timesteps = 1000


def mse_loss():
    pass


def forward_process(x, noise, alpha_bar_t):
    mean = jnp.sqrt(alpha_bar_t) * x
    stddev = jnp.sqrt(1 - alpha_bar_t)
    return mean + stddev * noise


def reverse_process(x_t, noise, beta_t, alpha_t, alpha_bar_t, noise_in_x_t):
    mean = (
        1
        / jnp.sqrt(alpha_t)
        * (x_t - beta_t / jnp.sqrt(1 - alpha_bar_t) * noise_in_x_t)
    )
    stddev = jnp.sqrt(beta_t)
    return mean + stddev * noise


def linear_schedule(timesteps: int, start: float = 0.0001, end: float = 0.02):
    beta = jnp.linspace(start, end, num=timesteps, dtype=full)
    beta = jnp.pad(beta, pad_width=(1, 0))
    return beta


@jax.jit
def train_step(state: TrainState, batch, dropout_key):
    def loss_fn(params, x, t, noise, alpha_bar_t):
        x_t = forward_process(x, noise, alpha_bar_t)

        noise_in_x_t = state.apply_fn(
            {"params": params}, x_t, t, training=True, rngs={"dropout": dropout_key}
        )

        loss = jnp.mean(optax.l2_loss(predictions=noise_in_x_t, targets=noise))

        return loss

    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss_val, grads = loss_grad_fn(
        state.params, batch["x"], batch["t"], batch["noise"], batch["alpha_bar_t"]
    )
    state = state.apply_gradients(grads=grads)
    return loss_val, state


def train(key):
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

    dummy_x = jnp.empty((batch_size, height, width, channels))
    dummy_t = jnp.empty((batch_size,))

    variables = model.init(params_key, dummy_x, dummy_t, training=False)
    state, params = variables.pop("params")
    del variables

    jax.tree_util.tree_map(lambda x: x.shape, params)  # Checking output shapes

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        key=dropout_key,
        tx=optax.adam(learning_rate),
    )

    beta = linear_schedule(timesteps)
    beta = einops.rearrange(beta, "t -> t 1 1 1")

    alpha = 1 - beta

    alpha_bar = jnp.cumprod(alpha, axis=0)

    x = random.normal(key, shape=(batch_size, height, width, channels))

    t0 = time.time()
    for i in range(num_steps):
        timestep_train_key = random.fold_in(timestep_key, state.step)
        noise_train_key = random.fold_in(noise_key, state.step)

        t = random.randint(
            timestep_train_key, shape=(batch_size,), minval=1, maxval=timesteps
        )
        noise = random.normal(
            noise_train_key, shape=(batch_size, height, width, channels)
        )

        alpha_bar_t = alpha_bar[t]
        t = t.astype(noise.dtype)

        batch = {"x": x, "t": t, "noise": noise, "alpha_bar_t": alpha_bar_t}

        loss, state = train_step(state, batch, dropout_key)

        if i % 100 == 0:
            t = time.time() - t0
            print(t / 100)
            t0 = time.time()


if __name__ == "__main__":
    key = random.PRNGKey(0)

    train(key)
