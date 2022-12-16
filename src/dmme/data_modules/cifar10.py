from typing import Callable, List

from torchvision import datasets
import torchvision.transforms as TF

from dmme.common import norm

from .dataset import Dataset


class CIFAR10(Dataset):
    def __init__(
        self,
        data_dir: str = ".",
        batch_size: int = 128,
        augs: List[Callable] = [TF.RandomHorizontalFlip()],
    ):
        super().__init__(batch_size)

        self.data_dir = data_dir
        self.augs = augs

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir, download=True)

    def dataset(self, augs=[]):
        return datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            transform=TF.Compose([*augs, TF.ToTensor(), norm]),
        )

    def setup_train(self):
        return self.dataset(self.augs)

    def setup_test(self):
        return self.dataset()
