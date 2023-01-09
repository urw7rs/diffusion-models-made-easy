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
