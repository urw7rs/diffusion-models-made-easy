from torch import clip


def norm(x):
    return (x - 0.5) * 2


def denorm(x):
    return clip((x + 1) / 2, 0, 1)
