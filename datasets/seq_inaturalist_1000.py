import json
from pathlib import Path
from typing import List, Tuple

import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from datasets.seq_inaturalist import DEFAULT_IMAGE_SIZE, METADATA_FILENAME, MyINaturalist
from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import (
    ContinualDataset,
    fix_class_names_order,
    store_masked_loaders,
)
from utils.conf import base_path


CLASS_LIST_FILENAME = "inaturalist1000_classes.txt"


class MyINaturalist1000(MyINaturalist):
    """
    Thin wrapper around ``MyINaturalist`` to emphasize separate storage for the 1000-class setup.
    """

    def __init__(self, root: str, split: str = "train", transform=None, target_transform=None) -> None:
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform)


class SequentialINaturalist1000(ContinualDataset):
    """Sequential iNaturalist dataset configured for 1000 classes."""

    NAME = "seq-inaturalist-1000"
    SETTING = "class-il"
    N_TASKS = 100
    N_CLASSES_PER_TASK = 10
    N_CLASSES = N_TASKS * N_CLASSES_PER_TASK
    SIZE = DEFAULT_IMAGE_SIZE
    MEAN, STD = (0.5071, 0.4867, 0.4408), (1.0, 1.0, 1.0)
    TRANSFORM = transforms.Compose(
        [
            transforms.Resize(SIZE, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(SIZE, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    def _root(self) -> Path:
        return Path(base_path() + "inaturalist1000")

    def get_data_loaders(self):
        root = self._root()
        try:
            train_dataset = MyINaturalist1000(str(root), split="train", transform=self.TRANSFORM)
            test_dataset = MyINaturalist1000(str(root), split="test", transform=self.TEST_TRANSFORM)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{exc}. Please run `python SACK/scripts/download_inaturalist_1000.py` before launching training."
            ) from exc

        if train_dataset.num_classes != self.N_CLASSES:
            raise ValueError(
                "Metadata reports "
                f"{train_dataset.num_classes} classes but dataset is configured for {self.N_CLASSES}. "
                "Regenerate metadata with the desired class count using `SACK/scripts/download_inaturalist_1000.py --force`."
            )

        train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)
        return train_loader, test_loader

    def _load_class_names_from_file(self) -> List[str]:
        class_list_path = self._root() / CLASS_LIST_FILENAME
        if not class_list_path.exists():
            return []
        return [name.strip() for name in class_list_path.read_text().splitlines() if name.strip()]

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        names = self._load_class_names_from_file()
        if not names:
            metadata_path = self._root() / METADATA_FILENAME
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata not found at {metadata_path}. Run `python SACK/scripts/download_inaturalist_1000.py` first."
                )
            metadata = json.loads(metadata_path.read_text())
            names = metadata.get("class_names", [])

        names = fix_class_names_order(names, self.args)
        self.class_names = names
        return self.class_names

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialINaturalist1000.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet50"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialINaturalist1000.MEAN, SequentialINaturalist1000.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialINaturalist1000.MEAN, SequentialINaturalist1000.STD)

    @set_default_from_args("batch_size")
    def get_batch_size(self):
        return 128

    @set_default_from_args("n_epochs")
    def get_epochs(self):
        return 70
