# Diffusion Models Made Easy

Diffusion Models Made Easy(`dmme`) is a collection of easy to understand diffusion model implementations in Pytorch.

Documentation is available at https://diffusion-models-made-easy.readthedocs.io/en/latest/

## Installation

Install from pip

```bash
pip install dmme
```

Install for customization or development

```bash
pip install -e ".[dev]"
```

Install dependencies for testing

```bash
pip install dmme[tests]
```

Install dependencies for editing documentation

```bash
pip install dmme[docs]
```

## Train Diffusion Models

`dmme` uses [LightningCLI](https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli.html#lightning-cli) as a cli interface for training and evaluation.

You can find sample configuration file in the `configs` directory

Using config files you can train DDPM by running

```bash
dmme.trainer fit --config configs/ddpm/cifar10.yaml
```

Or you can manually specify configurations for training

```bash
dmme.trainer fit --seed_everything 1337 \
    --trainer.accelerator gpu --trainer.precision 16 --trainer.benchmark true \
    --trainer.logger=pytorch_lightning.loggers.WandbLogger \
    --trainer.logger.project="CIFAR10_Image_Generation" \
    --trainer.logger.name="DDPM" \
    --trainer.gradient_clip_val=1.0 \
    --trainer.max_steps 800_000 \
    --model LitDDPM --data CIFAR10
```

## Supported Diffusion Models
- [DDPM](https://arxiv.org/abs/2006.11239)
- [DDIM](https://arxiv.org/abs/2010.02502)
- [IDDPM](https://arxiv.org/abs/2102.09672)
- (WIP) [Classifier Guidance](https://arxiv.org/abs/2105.05233)
