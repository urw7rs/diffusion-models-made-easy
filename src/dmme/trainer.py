from pytorch_lightning.cli import LightningCLI

import dmme


def main():
    cli = LightningCLI(seed_everything_default=1337)


if __name__ == "__main__":
    main()
