from tqdm import tqdm

import torch
from torch.optim import Adam

from dmme import CIFAR10

from dmme.ddpm import UNet, DDPMSampler
from dmme.lr_scheduler import WarmupLR
from dmme.noise_schedules import linear_schedule


def train(timesteps=1000, lr=2e-4, clip_val=1.0, warmup=5000, max_steps=800_000):
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

            torch.nn.utils.clip_grad_norm(sampler.parameters(), clip_val)

            optimizer.step()
            lr_scheduler.step()

            steps += 1

            prog_bar.set_postfix({"loss": loss, "steps": steps})

            if steps == max_steps:
                break


if __name__ == "__main__":
    train()
