from torch import clip


def norm(x):
    r"""Normalize input to :math:`[-1, 1]` linearly"""
    return (x - 0.5) * 2


def denorm(x):
    r"""Denormalize input normalized to :math:`[-1, 1]` linearly back to :math:`[0, 1]`"""
    return clip((x + 1) / 2, 0, 1)
