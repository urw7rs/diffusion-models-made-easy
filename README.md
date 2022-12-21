# Diffusion Models Made Easy

Diffusion Models Made Easy(`dmme`) is a collection of easy to understand diffusion model implementations in Pytorch.

# Getting Started

Documentation is available at https://diffusion-models-made-easy.readthedocs.io/en/latest/

## Installation

Install from pip

```bash
pip install dmme
```

installing `dmme` in edit mode requires `pip>=22.3`, update pip by running `pip install -U pip`

Install for customization or development

```bash
pip install -e ".[dev]"
```

Install dependencies for testing

```bash
pip install dmme[tests]
```

Install dependencies for docs

```bash
pip install dmme[docs]
```

## Train Diffusion Models

Train DDPM Using `LightningCLI` and `wandb` logger with mixed precision

```bash
python scripts/trainer.py fit --config configs/ddpm/cifar10.yaml
```

Train DDPM from python using pytorch-lightning

```python
from pytorch_lightning import Trainer

from pytorch_lightning.loggers import WandbLogger

from dmme import LitDDPM, DDPMSampler, CIFAR10
from dmme.ddpm import UNet

from dmme.callbacks import GenerateImage


def main():
    trainer = Trainer(
        logger=WandbLogger(project="CIFAR10 Image Generation", name="DDPM"),
        callbacks=GenerateImage((3, 32, 32)),
        gradient_clip_val=1.0,
        auto_select_gpus=True,
        accelerator="gpu",
        precision=16,
        max_steps=800_000,
    )

    ddpm = LitDDPM(
        DDPMSampler(UNet(in_channels=3), timesteps=1000),
        lr=2e-4,
        warmup=5000,
        imgsize=(3, 32, 32),
        timesteps=1000,
        decay=0.9999,
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


def train(timesteps=1000, lr=2e-4, clip_val=1.0, warmup=5000, max_steps=800_000):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model = UNet()

    sampler = DDPMSampler(model, timesteps=timesteps)
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

            torch.nn.utils.clip_grad_norm_(sampler.parameters(), clip_val)

            optimizer.step()
            lr_scheduler.step()

            steps += 1

            prog_bar.set_postfix({"loss": loss, "steps": steps})

            if steps == max_steps:
                break


if __name__ == "__main__":
    train()

```

## Supported Diffusion Models
- [DDPM](https://arxiv.org/abs/2006.11239)
