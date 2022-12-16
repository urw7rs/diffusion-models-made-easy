from typing import Callable, List

import subprocess
import os.path as osp

from torchvision import datasets
import torchvision.transforms as TF

import zipfile

from dmme.common import norm

from .dataset import Dataset


class LSUN(Dataset):
    def __init__(
        self,
        data_dir: str = ".",
        batch_size: int = 128,
        class_name: str = "train",
        augs: List[Callable] = [TF.RandomHorizontalFlip()],
    ):
        super().__init__(batch_size)

        self.data_dir = data_dir
        self.class_name = class_name
        self.augs = augs

    def prepare_data(self):
        self.download(self.data_dir, category=self.class_name, set_name="train")

    def download(self, out_dir, category, set_name):
        """code from https://github.com/fyu/lsun/blob/master/download.py"""

        url = f"http://dl.yf.io/lsun/scenes/{category}_{set_name}_lmdb.zip"
        if set_name == "test":
            out_name = "test_lmdb.zip"
            url = f"http://dl.yf.io/lsun/scenes/{set_name}_lmdb.zip"
        else:
            out_name = f"{category}_{set_name}_lmdb.zip"
        out_path = osp.join(out_dir, out_name)
        if osp.exists(osp.join(out_dir, out_name.split(".")[0])):
            print("File exists skipping download")
        else:
            cmd = ["aria2c", "-x", "16", "-s", "16", url, "-o", out_path]
            print("Downloading", category, set_name, "set")
            subprocess.call(cmd)

            with zipfile.ZipFile(out_path) as f:
                f.extractall(out_dir)

    def dataset(self, augs=[]):
        return datasets.LSUN(
            root=self.data_dir,
            classes="train",
            transform=TF.Compose([*augs, TF.ToTensor(), norm]),
        )

    def setup_train(self):
        return self.dataset(self.augs)

    def setup_test(self):
        return self.dataset()
