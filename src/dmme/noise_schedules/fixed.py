import torch


def cosine_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def quadratic_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start**0.5, end**0.5, timesteps) ** 2


def sigmoid_schedule(timesteps, start=0.0001, end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (end - start) + start
