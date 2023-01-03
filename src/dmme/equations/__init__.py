from dmme.equations import ddpm
from dmme.equations import ddim
from dmme.equations import iddpm

from collections import namedtuple

Gaussian = namedtuple("Gaussian", ["mean", "variance"])

__all__ = ["ddpm", "ddim", "iddpm", "Gaussian"]
