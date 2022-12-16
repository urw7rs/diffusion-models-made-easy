# Diffusion Models Made Easy

Diffusion Models Made Easy(`dmme`) is a collection of easy to understand diffusion model implementations in Pytorch.

# Getting Started

## Installation

```bash
pip install dmme
```

installing `dmme` in edit mode requires `pip>=22.3`, update pip by running `pip install -U pip`

## Training

Train DDPM Using `LightningCLI` and `wandb` logger with mixed precision

```bash
python scripts/trainer.py fit --config configs/ddpm/cifar10.yaml
```

Train DDPM from python using pytorch-lightning

```python
from pytorch_lightning import Trainer

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from dmme import LitDDPM, CIFAR10
from dmme.ddpm import UNet


def main():
    trainer = Trainer(
        logger=WandbLogger(project="CIFAR10 Image Generation", name="DDPM"),
        callbacks=[
            ModelCheckpoint(save_last=True),
            LearningRateMonitor(),
        ],
        gradient_clip_val=1.0,
        auto_select_gpus=True,
        accelerator="gpu",
        precision=16,
        max_steps=800_000,
    )

    ddpm = LitDDPM(
        decoder=UNet(in_channels=3),
        lr=2e-4,
        warmup=5000,
        imgsize=(3, 32, 32),
        timesteps=1000,
    )
    cifar10 = CIFAR10()

    trainer.fit(ddpm, cifar10)


if __name__ == "__main__":
    main()
```

or use the `DDPMSampler` class to train using pytorch

note: does not include gradient clipping, logging and checkpointing

```python
from tqdm import tqdm

import torch
from torch.optim import Adam

from dmme import CIFAR10

from dmme.ddpm import UNet, DDPMSampler
from dmme.lr_scheduler import WarmupLR
from dmme.noise_schedules import linear_schedule


def train(timesteps=1000, lr=2e-4, warmup=5000, max_steps=800_000):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model = UNet()
    beta = linear_schedule(timesteps=timesteps)
    sampler = DDPMSampler(model, timesteps=timesteps, beta=beta)
    sampler = sampler.to(device)

    cifar10 = CIFAR10()
    cifar10.prepare_data()
    cifar10.setup("fit")

    train_dataloader = cifar10.train_dataloader()

    optimizer = Adam(sampler.parameters(), lr=lr)
    lr_scheduler = WarmupLR(optimizer, warmup=warmup)

    steps = 0
    while steps < max_steps:
        prog_bar = tqdm(train_dataloader)
        for x_0, _ in prog_bar:
            x_0 = x_0.to(device)
            with torch.autocast("cuda" if device != "cpu" else "cpu"):
                loss = sampler.compute_loss(x_0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            steps += 1

            prog_bar.set_postfix({"loss": loss.cpu().item(), "steps": steps})

            if steps == max_steps:
                break


if __name__ == "__main__":
    train()
```

## Supported Diffusion Models
- [DDPM](https://arxiv.org/abs/2006.11239)
- Score Based Models comming soon...
