import multiprocessing as mp

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class Dataset(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size

    def setup_train(self):
        raise NotImplementedError

    def setup_test(self):
        raise NotImplementedError

    def setup(self, stage: str):
        if stage == "fit":
            self.train_set = self.setup_train()
        elif stage == "test":
            self.test_set = self.setup_test()

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cpu_count(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=cpu_count(),
        )


def cpu_count(*args, **kwargs):
    return mp.cpu_count(*args, **kwargs)
