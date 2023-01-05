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
