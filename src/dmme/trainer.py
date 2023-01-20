from pytorch_lightning.cli import LightningCLI


def main():
    """CLI entrypoint

    refer to `LightningCLI <https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html#lightning-cli>`_ docs for more information
    """
    cli = LightningCLI(seed_everything_default=1337)


if __name__ == "__main__":
    main()
