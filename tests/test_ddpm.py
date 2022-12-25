import torch
from torch import nn

from einops import rearrange

from dmme.ddpm import (
    DDPMSampler,
    ForwardProcess,
    ReverseProcess,
    linear_schedule,
)
from dmme.ddpm import pad

from dmme.common import gaussian_like

from pytorch_lightning import Trainer
from dmme import LitDDPM, CIFAR10


def test_pad():
    x = torch.randn(4, 4, 4)
    x = pad(x)
    assert list(x.size()) == [5, 4, 4]
    assert x[0, 0, 0].item() == 0


def test_linear_schedule():
    beta = linear_schedule(8)

    assert list(beta.size()) == [9]


def test_forward_process():
    beta = linear_schedule(8)
    beta = rearrange(beta, "t -> t 1 1 1")
    forward_process = ForwardProcess(beta)

    x_0 = torch.randn(4, 3, 32, 32)
    t = torch.randint(1, 8, size=(4,))
    noise = gaussian_like(x_0)

    x_t = forward_process(x_0, t, noise)


def test_reverse_process():
    beta = linear_schedule(8)
    beta = rearrange(beta, "t -> t 1 1 1")
    reverse_process = ReverseProcess(beta, sigma=beta)

    x_0 = torch.randn(4, 3, 32, 32)
    t = torch.randint(1, 8, size=(4,))
    noise = gaussian_like(x_0)

    x_t = reverse_process(x_0, t, torch.randn_like(noise), noise)


def test_ddpmsampler():
    sampler = DDPMSampler(timesteps=8)

    x_t = torch.randn(4, 3, 32, 32)
    noise = gaussian_like(x_t)

    timestep = torch.tensor([1, 2, 3, 8], device=x_t.device)
    x_t = sampler(x_t, timestep, torch.randn_like(noise), noise)


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
