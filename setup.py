from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="dmme",
    version="0.5.2",
    description="Diffusion Models Made Easy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/urw7rs/diffusion-models-made-easy",
    author="Chanhyuk Jung",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "torchvision",
        "einops",
        "pytorch-lightning>=1.8",
        "jsonargparse[signatures]>=4.12.0",
        "torchmetrics[image]",
        "wandb",
        "lmdb",
        "pillow==9.4",
    ],
    extras_require={
        "dev": ["black", "flake8", "sphinx-autobuild", "bumpver"],
        "docs": ["sphinx", "myst_parser", "furo", "sphinxcontrib-katex"],
    },
    entry_points={
        "console_scripts": [
            "dmme.trainer=dmme.trainer:main",
        ]
    },
)
