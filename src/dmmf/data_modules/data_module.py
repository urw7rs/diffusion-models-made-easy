import multiprocessing as mp

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    """LightningDataModule with defaults for generative modeling

    > Defaults are set from DDPM.

    `setup_train` and `setup_test` is used for preparing training and test sets.
    In practice, they both use training sets but augmentations are only applied on `setup_train`

    Prepares `DataLoader`s with good defaults with batch size set from `__init__`.

    Args:
        batch_size (int): batch size for `DataLoader`
    """

    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size

    def setup_train(self):
        """Prepare training set"""
        raise NotImplementedError

    def setup_test(self):
        """Prepare test set"""
        raise NotImplementedError

    def setup(self, stage: str):
        """Prepare dataset for training or testing"""
        if stage == "fit":
            self.train_set = self.setup_train()
        elif stage == "test":
            self.test_set = self.setup_test()

    def train_dataloader(self):
        """DataLoader with good defaults

        automatically sets num_workers based on cpu count.
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cpu_count(),
        )

    def test_dataloader(self):
        """DataLoader with good defaults

        automatically sets num_workers based on cpu count.
        """
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=cpu_count(),
        )


def cpu_count(*args, **kwargs):
    """returns cpu count from multiprocessing package"""
    return mp.cpu_count(*args, **kwargs)
