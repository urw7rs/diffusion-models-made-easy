import torch

from dmme.models import UNet


def test_unet():
    model = UNet(in_channels=3)

    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(1, 8, size=(2,))

    output = model(x, t)

    assert output.size() == x.size()
