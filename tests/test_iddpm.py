import torch
from torch import nn

import dmme.equations as eq

from dmme.diffusion_models import IDDPM
from dmme.models import UNet


def test_cosine_schedule():
    alpha_bar = eq.iddpm.cosine_schedule(100, 0.008)

    assert torch.isnan(alpha_bar).any().item() == False
    assert alpha_bar.size(0) == 101


def test_vlb_loss():
    model = UNet(
        in_channels=3,
        pos_dim=4,
        emb_dim=8,
        num_groups=2,
        channels_per_depth=(4, 8, 16, 32),
        num_blocks=3,
    )
    model.output_conv = nn.Conv2d(4, 6, 3, 1, 1)

    for loss_type in ["hybrid", "vlb"]:
        iddpm = IDDPM(model, timesteps=2, loss_type=loss_type)

        x_0 = torch.randn(4, 3, 64, 64)
        loss = iddpm.training_step(x_0)

        assert torch.isnan(loss).any().item() == False
        loss.backward()


def test_improved_ddpm_sampling():
    model = UNet(
        in_channels=3,
        pos_dim=4,
        emb_dim=8,
        num_groups=2,
        channels_per_depth=(4, 8, 16, 32),
        num_blocks=3,
    )
    model.output_conv = nn.Conv2d(4, 6, 3, 1, 1)

    iddpm = IDDPM(model, timesteps=100)

    x_t = torch.randn(3, 3, 32, 32)
    t = torch.randint(0, 100, size=(3,))

    output = iddpm.sampling_step(x_t, t)

    assert output.size() == x_t.size()
