from typing import Callable, List, Optional

import subprocess
import os.path as osp

import torchvision.transforms as TF
from torch.utils.data import Dataset

import zipfile

from dmme.common import norm

from .data_module import DataModule
from .. import datasets


class LSUN(DataModule):
    """LSUN datamodule

    Args:
        data_dir: path to cifar10
        batch_size: batch size
        class_name: lsun zipfile name to load, specify as list to load multiple classes.
            Set to :code:`"train"` to load all train scenes in lsun. Unsupported for objects.
        imgsize: dataset image size
        augs (List[Transform]): augmentations on PIL images only from torchvision.
            Set to :code:`None` to disable augmentations
    """

    scenes = set(
        [
            "bedroom",
            "bridge",
            "church_outdoor",
            "classroom",
            "conference_room",
            "dining_room",
            "kitchen",
            "living_room",
            "restaurant",
            "test",
            "tower",
        ]
    )

    def __init__(
        self,
        data_dir: str = ".",
        batch_size: int = 128,
        class_name: str = "train",
        imgsize: int = 256,
        augs: Optional[List[Callable]] = [TF.RandomHorizontalFlip()],
    ):
        super().__init__(batch_size)

        self.data_dir = data_dir
        self.class_name = class_name
        self.imgsize = imgsize
        if augs is None:
            augs = []
        self.augs = augs

    def prepare_data(self):
        r"""Download and extract `LSUN <https://www.yf.io/p/lsun>`_ dataset

        LSUN is downloaded using `aria2 <https://aria2.github.io>`_ to speedup downloads.
        """

        if self.class_name in ["train", "val", "test"]:
            for category in self.scenes:
                self.download_scenes(self.data_dir, category, set_name=self.class_name)

            self.classes = self.class_name
        else:
            category = "_".join(self.class_name.split("_")[:-1])
            if category in self.scenes:
                set_name = self.class_name.split("_")[-1]

                self.download_scenes(self.data_dir, category, set_name)

            else:
                self.download_objects(self.data_dir, category=self.class_name)

            self.classes = self.class_name
            if self.class_name not in ["train", "val", "test"]:
                self.classes = [self.classes]

    def download_scenes(self, out_dir, category: str, set_name: str):
        """Download lsun scenes data

        Args:
            out_dir: output directory
            category: category name to download. should match the category
                names in `scenes <http://dl.yf.io/lsun/scenes/>`_
            set_name: either "train", "val", or "test"
        """

        if set_name == "test":
            out_name = "test_lmdb.zip"
        else:
            out_name = f"{category}_{set_name}_lmdb.zip"

        url = f"http://dl.yf.io/lsun/scenes/{out_name}"

        self.download_url(url, out_dir, out_name)

    def download_objects(self, out_dir, category: str):
        """Download lsun objects data

        Args:
            out_dir: output directory
            category: category name to download. should match the category
                names in `scenes <http://dl.yf.io/lsun/scenes/>`_
            set_name: either "train", "val", or "test"
        """

        out_name = f"{category}.zip"
        url = f"http://dl.yf.io/lsun/objects/{out_name}"

        self.download_url(url, out_dir, out_name)

    def download_url(self, url: str, out_dir, out_name: str):
        """Download url and extract zip, will skip if file exists

        Args:
            url: url to download
            out_dir: output directory
            out_name: file name to download as
        """

        lmdb_path = osp.join(out_dir, out_name.split(".")[0])

        if osp.exists(lmdb_path):
            print("File exists skipping download")
        else:
            out_path = osp.join(out_dir, out_name)

            if not osp.exists(out_path):
                cmd = ["aria2c", "-x", "16", "-s", "16", url, "-o", out_path]
                print(f"Downloading {out_name}...")
                subprocess.call(cmd)

            print(f"Extracting {out_name}...")
            with zipfile.ZipFile(out_path) as f:
                f.extractall(out_dir)

    def dataset(self, augs: List[Callable] = []) -> Dataset:
        classes = self.class_name

        if self.class_name not in ["train", "val", "test"]:
            if isinstance(self.class_name, str):
                classes = [self.class_name]

        return datasets.LSUN(
            root=self.data_dir,
            classes=classes,
            transform=TF.Compose(
                [
                    *augs,
                    TF.Resize(size=self.imgsize),
                    TF.CenterCrop(size=self.imgsize),
                    TF.ToTensor(),
                    norm,
                ]
            ),
        )

    def setup_train(self):
        return self.dataset(self.augs)

    def setup_test(self):
        return self.dataset()
