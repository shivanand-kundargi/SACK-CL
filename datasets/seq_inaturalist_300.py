import json
from pathlib import Path
from typing import List, Optional, Sequence

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
CLASS_LIMIT = 300
CLASSES_PER_TASK = 30


class MyINaturalist300(MyINaturalist):
    """Subset of iNaturalist limited to the first 300 classes defined in the 1000-class list."""

    CLASS_LIMIT = CLASS_LIMIT

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        allowed_classes: Optional[Sequence[int]] = None,
    ) -> None:
        root_path = Path(root)
        resolved_allowed = (
            tuple(allowed_classes) if allowed_classes is not None else tuple(self._resolve_allowed_classes(root_path))
        )
        self.allowed_classes = resolved_allowed
        super().__init__(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform,
            class_filter=self.allowed_classes,
        )

    @classmethod
    def _resolve_allowed_classes(cls, root: Path) -> List[int]:
        class_list_path = root / CLASS_LIST_FILENAME
        if not class_list_path.exists():
            raise FileNotFoundError(
                f"Class list not found at {class_list_path}. Ensure the iNaturalist 1000 class list is downloaded."
            )

        metadata_path = root / METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. Run `python SACK/scripts/download_inaturalist_1000.py` first."
            )

        metadata = json.loads(metadata_path.read_text())
        class_names = metadata.get("class_names", [])
        if not class_names:
            raise ValueError(
                "Class names missing from metadata; cannot select the first 300 classes without class name information."
            )

        name_to_index = {name: idx for idx, name in enumerate(class_names)}
        ordered_names = [line.strip() for line in class_list_path.read_text().splitlines() if line.strip()]
        limited_names = ordered_names[: cls.CLASS_LIMIT]
        if len(limited_names) < cls.CLASS_LIMIT:
            raise ValueError(
                f"Expected at least {cls.CLASS_LIMIT} classes in {class_list_path}, found only {len(limited_names)}."
            )

        missing = [name for name in limited_names if name not in name_to_index]
        if missing:
            sample = ", ".join(missing[:3])
            raise ValueError(
                f"The following classes are missing from metadata and cannot be selected: {sample}"
                + ("..." if len(missing) > 3 else "")
            )

        return [name_to_index[name] for name in limited_names]


class SequentialINaturalist300(ContinualDataset):
    """Sequential iNaturalist dataset restricted to the first 300 classes from the 1000-class configuration."""

    NAME = "seq-inaturalist-300"
    SETTING = "class-il"
    N_TASKS = CLASS_LIMIT // CLASSES_PER_TASK
    N_CLASSES_PER_TASK = CLASSES_PER_TASK
    N_CLASSES = CLASS_LIMIT
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

    def _allowed_classes(self) -> Sequence[int]:
        if not hasattr(self, "_cached_allowed_classes"):
            self._cached_allowed_classes = tuple(MyINaturalist300._resolve_allowed_classes(self._root()))
        return self._cached_allowed_classes

    def get_data_loaders(self):
        root = self._root()
        allowed = self._allowed_classes()
        try:
            train_dataset = MyINaturalist300(str(root), split="train", transform=self.TRANSFORM, allowed_classes=allowed)
            test_dataset = MyINaturalist300(str(root), split="test", transform=self.TEST_TRANSFORM, allowed_classes=allowed)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{exc}. Please run `python SACK/scripts/download_inaturalist_1000.py` before launching training."
            ) from exc

        if train_dataset.num_classes != self.N_CLASSES:
            raise ValueError(
                "Metadata reports "
                f"{train_dataset.num_classes} classes but dataset is configured for {self.N_CLASSES}. "
                "Verify that the class list contains at least 300 entries and regenerate metadata if necessary."
            )

        train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)
        return train_loader, test_loader

    def _load_class_names_from_file(self) -> List[str]:
        class_list_path = self._root() / CLASS_LIST_FILENAME
        if not class_list_path.exists():
            return []
        names = [line.strip() for line in class_list_path.read_text().splitlines() if line.strip()]
        return names[: self.N_CLASSES]

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        names = self._load_class_names_from_file()
        if not names:
            names = MyINaturalist.get_class_names(str(self._root()), class_filter=self._allowed_classes())

        names = fix_class_names_order(names, self.args)
        self.class_names = names
        return self.class_names

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialINaturalist300.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet50"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialINaturalist300.MEAN, SequentialINaturalist300.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialINaturalist300.MEAN, SequentialINaturalist300.STD)

    @set_default_from_args("batch_size")
    def get_batch_size(self):
        return 128

    @set_default_from_args("n_epochs")
    def get_epochs(self):
        return 70
