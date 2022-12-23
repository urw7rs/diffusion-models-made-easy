import torch
from torch import nn

from einops import rearrange


from dmme.ddpm import ForwardProcess, linear_schedule
from dmme.ddpm import alpha_from_beta, alpha_bar_from_alpha, pad

from dmme.common import gaussian_like

from pytorch_lightning import Trainer

from dmme.ddpm.lit_ddpm import LitDDPM
from dmme.ddpm.unet import UNet

from dmme import CIFAR10


def test_pad():
    x = torch.randn(4, 4, 4)
    x = pad(x)
    assert list(x.size()) == [5, 4, 4]
    assert x[0, 0, 0].item() == 0


def test_linear_schedule():
    beta = linear_schedule(8)

    assert list(beta.size()) == [9]


def test_alpha_from_beta():
    beta = linear_schedule(8)
    alpha = alpha_from_beta(beta)
    assert list(alpha.size()) == [9]


def test_alpha_bar_from_beta():
    beta = rearrange(linear_schedule(8), "t -> t 1 1 1")
    alpha = alpha_from_beta(beta)
    alpha_bar = alpha_bar_from_alpha(alpha)
    assert list(alpha_bar.size()) == [9, 1, 1, 1]


def test_forward_process():
    forward_process = ForwardProcess.build(8)

    x_0 = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 8, size=(4,))
    noise = gaussian_like(x_0)

    x_t = forward_process(x_0, t, noise)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x, *args, **kwargs):
        return self.conv(x)


def test_litddpm_fit():
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(LitDDPM(DummyModel()), CIFAR10(batch_size=4))


def test_litddpm_test():
    trainer = Trainer(fast_dev_run=True)
    trainer.test(LitDDPM(DummyModel()), CIFAR10(batch_size=4))
