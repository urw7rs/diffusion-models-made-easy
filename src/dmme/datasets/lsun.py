from typing import Any, cast, Callable, List, Optional, Tuple, Union

import os.path as osp

from torchvision.datasets import VisionDataset, LSUNClass


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
            if c in self.objects:
                root = osp.join(root, c)
            else:
                root = osp.join(root, f"{c}_lmdb")

            self.dbs.append(LSUNClass(root=root, transform=transform))

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
