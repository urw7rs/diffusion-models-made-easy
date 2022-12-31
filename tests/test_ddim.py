import torch
from torch import nn

from dmme.ddim import DDIM


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, *args):
        return x


def test_ddim_sampling():
    model = DummyModel()
    ddim = DDIM(model, timesteps=100, sub_timesteps=5)

    x_t = torch.randn(3, 3, 32, 32)
    t = torch.tensor([1])

    output = ddim.sampling_step(x_t, t)

    assert output.size() == x_t.size()
