from typing import Callable, List, Optional

from torchvision import datasets
import torchvision.transforms as TF

from dmme.common import norm

from .data_module import DataModule


class CIFAR10(DataModule):
    r"""CIFAR10 Dataset scaled to :math:`[-1, 1]`

    Download CIFAR10 using torchvision and apply augmentations from argument and finally scale images to :math:`[-1, 1]`

    Args:
        data_dir (str): path to cifar10
        batch_size (int): batch size
        augs (List[Transform]): augmentations on PIL images only from torchvision
    """

    def __init__(
        self,
        data_dir: str = ".",
        batch_size: int = 128,
        augs: Optional[List[Callable]] = None,
    ):
        super().__init__(batch_size)

        self.data_dir = data_dir

        if augs is None:
            augs = [TF.RandomHorizontalFlip()]
        self.augs = augs

    def prepare_data(self):
        """Download dataset"""
        datasets.CIFAR10(root=self.data_dir, download=True)

    def dataset(self, augs=[]):
        """Dataset builder"""
        return datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            transform=TF.Compose([*augs, TF.ToTensor(), norm]),
        )

    def setup_train(self):
        return self.dataset(self.augs)

    def setup_test(self):
        return self.dataset()
