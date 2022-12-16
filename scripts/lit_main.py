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
