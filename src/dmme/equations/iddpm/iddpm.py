import math

import torch


def cosine_schedule(timesteps, offset):
    def f(t):
        return torch.cos((t / timesteps + offset) / (1 + offset) * math.pi / 2) ** 2

    t = torch.arange(0, timesteps + 1)
    zero = torch.tensor([0], dtype=torch.float32)
    alpha_bar = f(t) / f(zero)
    return alpha_bar
