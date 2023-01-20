import torch


def gaussian(shape, dtype=None, device=None):
    """Samples from gaussian with specified shape, dtype, device using `torch.randn`"""
    return torch.randn(shape, dtype=dtype, device=device)


def gaussian_like(x):
    """Samples from gaussian like the tensor x using `torch.randn_like`"""
    return torch.randn_like(x)


def uniform_int(min, max, count=1, device=None):
    """Samples ints from uniform distribution using `torch.randint`"""
    return torch.randint(min, max, size=(count,), device=device)


def pad(x: torch.Tensor, value: float = 0) -> torch.Tensor:
    r"""pads tensor with 0 to match :math:`t` with tensor index"""

    ones = torch.ones_like(x[0:1])
    return torch.cat([ones * value, x], dim=0)
