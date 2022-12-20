from torch import randn, randint
from torch import randn_like


def gaussian(shape, dtype=None, device=None):
    """Samples from gaussian with specified shape, dtype, device using `torch.randn`"""
    return randn(shape, dtype=dtype, device=device)


def gaussian_like(x):
    """Samples from gaussian like the tensor x using `torch.randn_like`"""
    return randn_like(x)


def uniform_int(min, max, count=1, device=None):
    """Samples ints from uniform distribution using `torch.randint`"""
    return randint(min, max, size=(count,), device=device)
