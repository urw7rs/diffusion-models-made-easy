from typing import Callable, List, Optional

import subprocess
import os.path as osp

from .. import datasets
import torchvision.transforms as TF

import zipfile

from dmme.common import norm

from .data_module import DataModule


class LSUN(DataModule):
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
        imgsize=256,
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

    def download_scenes(self, out_dir, category, set_name):
        if set_name == "test":
            out_name = "test_lmdb.zip"
        else:
            out_name = f"{category}_{set_name}_lmdb.zip"

        url = f"http://dl.yf.io/lsun/scenes/{out_name}"

        self.download_url(url, out_dir, out_name)

    def download_objects(self, out_dir, category):
        out_name = f"{category}.zip"
        url = f"http://dl.yf.io/lsun/objects/{out_name}"

        self.download_url(url, out_dir, out_name)

    def download_url(self, url, out_dir, out_name):
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

    def dataset(self, augs=[]):
        return datasets.LSUN(
            root=self.data_dir,
            classes=self.classes,
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
