import math

import torch
from torchvision.utils import make_grid


def make_history(history):
    r"""Visualize diffusion process given an array of histories

    Args:
        history (List[torch.Tensor]): list of diffusion history with each item as a tensor of shape :math:`(N, C, H, W)`

    """
    if len(history) == 1:
        img = history[-1]

        nrow = 1

        batch_size = img.size(0)
        for i in range(int(math.sqrt(batch_size)), 2, -1):
            if batch_size % i == 0:
                nrow = batch_size // i
                break

        grid = make_grid(img, nrow=nrow)
    else:
        history = torch.stack(history, dim=1)
        grid = make_grid(history.flatten(0, 1), nrow=history.size(1))

    return grid
