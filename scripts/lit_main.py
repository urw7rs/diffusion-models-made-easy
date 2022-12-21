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
