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
dmme.trainer fit --config configs/ddpm/cifar10.yaml
```

Train DDPM from python using pytorch-lightning

```python
from pytorch_lightning import Trainer

from pytorch_lightning.loggers import WandbLogger

from dmme.ddpm import LitDDPM, UNet
from dmme.data import CIFAR10

from dmme.callbacks import GenerateImage


def main():
    trainer = Trainer(
        logger=WandbLogger(project="CIFAR10_Image_Generation", name="DDPM"),
        callbacks=GenerateImage((3, 32, 32), timesteps=1000),
        gradient_clip_val=1.0,
        auto_select_gpus=True,
        accelerator="gpu",
        precision=16,
        max_steps=800_000,
    )

    ddpm = LitDDPM(
        UNet(in_channels=3),
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
import torch.nn.functional as F

from dmme.data import CIFAR10

from dmme.ddpm import DDPM, UNet
from dmme.lr_scheduler import WarmupLR

from dmme.common import gaussian_like, uniform_int


def train(timesteps=1000, lr=2e-4, clip_val=1.0, warmup=5000, max_steps=800_000):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model = UNet()
    model = model.to(device)

    ddpm = DDPM(timesteps=timesteps)
    ddpm = ddpm.to(device)

    cifar10 = CIFAR10()
    cifar10.prepare_data()
    cifar10.setup("fit")

    train_dataloader = cifar10.train_dataloader()

    optimizer = Adam(model.parameters(), lr=lr)
    lr_scheduler = WarmupLR(optimizer, warmup=warmup)

    steps = 0
    while steps < max_steps:
        prog_bar = tqdm(train_dataloader)
        for x_0, _ in prog_bar:
            x_0 = x_0.to(device)

            batch_size: int = x_0.size(0)
            t = uniform_int(0, timesteps, batch_size, device=x_0.device)

            noise = gaussian_like(x_0)

            with torch.autocast("cuda" if device != "cpu" else "cpu"):
                x_t = ddpm.forward_process(x_0, t, noise)

                noise_estimate = model(x_t, t)

                loss = F.mse_loss(noise, noise_estimate)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

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
