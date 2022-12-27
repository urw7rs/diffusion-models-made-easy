import torch

from dmme.ddpm import UNet

import pytest


def test_unet():
    model = UNet(in_channels=3)

    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(1, 8, size=(2,))

    output = model(x, t)

    assert x.size() == output.size()

    model = UNet(
        in_channels=3,
        channels=(128, 256, 256, 256, 256),
        downsample_layers=(1, 3),
        attention_depths=(1,),
    )

    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(1, 8, size=(2,))

    output = model(x, t)

    assert x.size() == output.size()

    model = UNet(in_channels=3, channels=(128, 256, 256, 256))

    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(1, 8, size=(2,))

    output = model(x, t)

    assert x.size() == output.size()

    model = UNet(
        in_channels=3,
        channels=(128, 256, 256, 256),
        downsample_layers=(1, 2),
    )

    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(1, 8, size=(2,))

    output = model(x, t)

    assert x.size() == output.size()
