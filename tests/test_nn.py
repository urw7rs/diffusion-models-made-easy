import torch
from torch import nn

from einops.layers.torch import Rearrange

import dmme.nn as dnn


def test_sequential():
    f = dnn.Sequential(
        (nn.Conv2d(1, 4, 3, 1, 1), "x -> x"),
        (
            nn.Sequential(
                nn.Linear(1, 4),
                Rearrange("b c ->  b c 1 1"),
            ),
            "t -> t",
        ),
        (dnn.Add(), "x t -> x"),
        (nn.GroupNorm(2, 4), "x -> x"),
        (nn.SiLU(), "x -> x"),
    )

    x = torch.randn(2, 1, 32, 32)
    t = torch.randn(2, 1)

    x = f(x=x, t=t)

    assert isinstance(x, torch.Tensor)
    # x = f(x, t)
