import jax
import jax.numpy as jnp

from flax import struct

from einops import rearrange

Array = jax.Array


@struct.dataclass
class Linear:
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
