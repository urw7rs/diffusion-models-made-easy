import pytest

import torch
from torch import nn

from einops import rearrange

from dmme.ddim import DDIMSampler, DDIMReverseProcess

from dmme.common import gaussian_like

from pytorch_lightning import Trainer
from dmme import LitDDIM, CIFAR10

from dmme.ddpm import linear_schedule


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("tau_schedule", ["linear", "quadratic", "Linear", "Quadratic"])
def test_reverse_process(tau_schedule, eta):
    beta = linear_schedule(timesteps=100)
    beta = rearrange(beta, "t -> t 1 1 1")

    tau = torch.arange(0, 5)
    reverse_process = DDIMReverseProcess(beta, tau, eta)

    x_0 = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 5, size=(4,))
    noise = gaussian_like(x_0)

    x_t = reverse_process(x_0, t, torch.randn_like(noise), noise)


@pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("tau_schedule", ["linear", "quadratic", "Linear", "Quadratic"])
def test_ddpmsampler(tau_schedule, eta):
    beta = linear_schedule(timesteps=100)
    beta = rearrange(beta, "t -> t 1 1 1")
    sampler = DDIMSampler(beta, 10, tau_schedule, eta)

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


def test_litddim_fit():
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(LitDDIM(DummyModel()), CIFAR10(batch_size=4))


def test_litddim_test():
    trainer = Trainer(fast_dev_run=True)
    trainer.test(LitDDIM(DummyModel()), CIFAR10(batch_size=4))
