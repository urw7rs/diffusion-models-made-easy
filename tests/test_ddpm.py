import torch
from torch import nn

from dmme.ddpm import DDPM, UNet


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, *args):
        return x


def test_ddpm_training():
    model = UNet(
        in_channels=3,
        pos_dim=4,
        emb_dim=8,
        num_groups=2,
        channels_per_depth=(4, 8, 16, 32),
        num_blocks=3,
    )
    ddpm = DDPM(model, timesteps=100)

    x_0 = torch.randn(3, 3, 32, 32)

    loss: torch.Tensor = ddpm.training_step(x_0)

    assert torch.isnan(loss).any().item() == False
    assert loss.ndim == 0


def test_ddpm_sampling():
    model = UNet(
        in_channels=3,
        pos_dim=4,
        emb_dim=8,
        num_groups=2,
        channels_per_depth=(4, 8, 16, 32),
        num_blocks=3,
    )
    ddpm = DDPM(model, timesteps=100)

    x_t = torch.randn(3, 3, 32, 32)
    t = torch.randint(0, 100, size=(3,))

    output = ddpm.sampling_step(x_t, t)

    assert output.size() == x_t.size()


def test_ddpm_generate():
    model = UNet(
        in_channels=3,
        pos_dim=4,
        emb_dim=8,
        num_groups=2,
        channels_per_depth=(4, 8, 16, 32),
        num_blocks=3,
    )
    ddpm = DDPM(model, timesteps=100)

    x_t = torch.randn(2, 3, 32, 32)

    output = ddpm.generate(x_t.size())

    assert output.size() == x_t.size()
