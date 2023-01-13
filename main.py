from typing import Tuple
import functools

import jax
from jax import random
import jax.numpy as jnp

from flax import linen as nn

from einops.layers.flax import Rearrange


class ResBlock(nn.Module):
    features: int
    emb_dim: int
    drop_rate: float
    with_attention: bool

    @nn.compact
    def __call__(self, x, t, training: bool):
        residual = x
        Conv3x3 = functools.partial(
            nn.Conv, features=self.features, kernel_size=(3, 3), padding="same"
        )

        x = nn.GroupNorm()(x)
        x = nn.swish(x)
        x = Conv3x3()(x)

        num_channels = x.shape[-1]
        condition = nn.Sequential(
            [nn.Dense(features=num_channels), Rearrange("b c -> b 1 1 c")]
        )(t)

        x = x + condition

        x = nn.GroupNorm()(x)
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
            x = nn.GroupNorm()(x)
            residual = x

            x = nn.MultiHeadDotProductAttention(
                num_heads=1, qkv_features=self.features
            )(inputs_q=x, inputs_kv=x)

            x = x + residual

        return x


def embed(t, dim):
    half_dim = dim // 2
    embeddings = jnp.log(10000) / (half_dim - 1)
    embeddings = jnp.exp(jnp.arange(0, half_dim) * -embeddings)
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
        t = embed(t, self.pos_dim)
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
        x = nn.GroupNorm()(x)
        x = nn.silu(x)
        x = Conv3x3(features=self.in_channels)(x)

        return x


def train(key, batch_size, height, width, channels):
    key, subkey = random.split(key)

    unet = UNet(
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
    variables = unet.init(subkey, dummy_x, dummy_t, training=False)


if __name__ == "__main__":
    key = random.PRNGKey(0)

    train(key, batch_size=128, height=32, width=32, channels=3)