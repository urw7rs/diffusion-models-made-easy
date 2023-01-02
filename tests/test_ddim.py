import torch

from dmme.diffusion_models import DDIM
from dmme.models import UNet


def test_ddim_sampling():
    model = UNet(
        in_channels=3,
        pos_dim=4,
        emb_dim=8,
        num_groups=2,
        channels_per_depth=(4, 8, 16, 32),
        num_blocks=3,
    )
    ddim = DDIM(model, timesteps=100, sub_timesteps=5)

    x_t = torch.randn(3, 3, 32, 32)
    i = torch.randint(0, 5, size=(3,))

    output = ddim.sampling_step(x_t, i)

    assert output.size() == x_t.size()


def test_ddim_generate():
    model = UNet(
        in_channels=3,
        pos_dim=4,
        emb_dim=8,
        num_groups=2,
        channels_per_depth=(4, 8, 16, 32),
        num_blocks=3,
    )
    ddim = DDIM(model, timesteps=100, sub_timesteps=5)

    x_t = torch.randn(3, 3, 32, 32)

    output = ddim.generate(x_t.size())

    assert output.size() == x_t.size()
