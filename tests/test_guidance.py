import pytest

import torch
from torch import nn

from einops.layers.torch import Rearrange

from dmme.guidance import ClassifierGuidedDDPM, ClassifierGuidedDDIM
from dmme.common import gaussian_like

num_classes = 10
batch_size = 8
timesteps = 10


@pytest.fixture()
def y():
    return torch.randint(0, num_classes, size=(batch_size,))


@pytest.fixture()
def x_t():
    return torch.randn(batch_size, 3, 32, 32)


@pytest.fixture()
def t():
    return torch.randint(1, timesteps, size=(batch_size,))


@pytest.fixture()
def model():
    return Model()


@pytest.fixture()
def classifier():
    return Classifier()


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, 1, 1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(4, 3, 3, 1, 1)

        self.linear = nn.Sequential(
            Rearrange("b -> b 1"),
            nn.Linear(1, 4),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x, t):
        x = self.conv1(x)
        x += self.linear(t.float())
        x = self.conv2(x)
        return x


class Classifier(Model):
    def __init__(self) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, num_classes),
        )

    def forward(self, x, t):
        x = super().forward(x, t)
        return self.fc(x)


def test_classifier_guided_diffusion_sampling(model, classifier, y, x_t, t):
    guidance = ClassifierGuidedDDPM(timesteps=timesteps)

    noise = gaussian_like(x_t)
    output = guidance.sample(model, classifier, y, x_t, t, noise)

    assert output.size() == x_t.size()


def test_classifier_guided_ddim_sampling(model, classifier, y, x_t, t):
    guidance = ClassifierGuidedDDIM(timesteps=timesteps)

    output = guidance.sample(model, classifier, y, x_t, t)

    assert output.size() == x_t.size()
