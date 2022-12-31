import torch
from torch import nn

from dmme.ddpm import DDPM


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, *args):
        return x


def test_ddpm_training():
    model = DummyModel()
    ddpm = DDPM(model, timesteps=100)

    x_0 = torch.randn(3, 3, 32, 32)

    loss: torch.Tensor = ddpm.training_step(x_0)

    assert torch.isnan(loss).any().item() == False
    assert loss.ndim == 0


def test_ddpm_sampling():
    model = DummyModel()
    ddpm = DDPM(model, timesteps=100)

    x_t = torch.randn(3, 3, 32, 32)
    t = torch.tensor([1])

    output = ddpm.sampling_step(x_t, t)

    assert output.size() == x_t.size()


def test_ddpm_generate():
    model = DummyModel()
    ddpm = DDPM(model, timesteps=100)

    x_t = torch.randn(2, 3, 32, 32)

    output = ddpm.generate((2, 3, 32, 32))

    assert output.size() == x_t.size()
