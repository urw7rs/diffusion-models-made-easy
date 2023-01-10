from typing import Any, cast, Callable, List, Optional, Tuple, Union

import os.path as osp
import io
import pickle
import string

import PIL
from PIL import Image

from torchvision.datasets import VisionDataset


class LSUNClass(VisionDataset):
    """LSUNClass from torchvision

    Loads lmdb dataset with empty value checks.

    Args:
        root (str): directory containing mdb files
        transform (Transform): transforms to apply on data, Optional
        target_transform (Transform): transforms to apply on labels, Optional
        ignore_keys (List): list of keys to ignore in lmdb database, some keys have empty values
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        ignore_keys: Optional[List] = None,
    ) -> None:
        import lmdb

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        cache_file = "_cache_" + "".join(c for c in root if c in string.ascii_letters)
        if osp.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            print("creating cache, this may take a while...")

            with self.env.begin(write=False) as txn:
                if ignore_keys is not None:
                    self.keys = [
                        key for key in txn.cursor().iternext(keys=True, values=False)
                    ]
                    for key in ignore_keys:
                        self.keys.remove(key)
                else:
                    self.keys = []
                    for key, value in txn.cursor():
                        self.keys.append(key)

                        try:
                            buf = io.BytesIO()
                            buf.write(value)
                            Image.open(buf)
                        except PIL.UnidentifiedImageError:
                            # skip invalid values
                            print(f"skipped {key}")
                            self.keys.pop()

            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.keys)


class LSUN(VisionDataset):
    """`LSUN <https://www.yf.io/p/lsun>`_ dataset.

    You will need to install the ``lmdb`` package to use this dataset: run
    ``pip install lmdb``

    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    scenes = [
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

    objects = [
        "airplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining_table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted_plant",
        "sheep",
        "sofa",
        "train",
        "tv-monitor",
    ]

    ignore_keys = {
        "cat": [
            b"05c509a12295c0725be85566680c58c81965ea63",
            b"0ec91d487375c2663a43d463f9e5b4e34b8527aa",
        ]
    }

    def __init__(
        self,
        root: str,
        classes: Union[str, List[str]] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        if classes in ["train", "val", "test"]:
            classes = cast(str, classes)
            if classes == "test":
                classes = [classes]
            else:
                classes = [c + "_" + classes for c in self.scenes]
        elif isinstance(classes, str):
            classes = [classes]

        self.classes = classes

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            ignore_keys = None

            if c in self.objects:
                root = osp.join(root, c)
                ignore_keys = self.ignore_keys[c]
            else:
                root = osp.join(root, f"{c}_lmdb")

            self.dbs.append(
                LSUNClass(root=root, transform=transform, ignore_keys=ignore_keys)
            )

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target

    def __len__(self) -> int:
        return self.length

    def extra_repr(self) -> str:
        return "Classes: {classes}".format(**self.__dict__)
