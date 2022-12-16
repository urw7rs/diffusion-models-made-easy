from torch import randn, randint
from torch import randn_like


def gaussian(shape, dtype=None, device=None):
    return randn(shape, dtype=dtype, device=device)


def gaussian_like(x):
    return randn_like(x)


def uniform_int(min, max, count=1, device=None):
    return randint(min, max, size=(count,), device=device)
