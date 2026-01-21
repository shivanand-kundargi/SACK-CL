import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
try:
    RESIZE_INTERPOLATION = Image.Resampling.BILINEAR  # Pillow >= 9.1.0
except AttributeError:  # pragma: no cover - fallback for older Pillow versions
    RESIZE_INTERPOLATION = Image.BILINEAR
from torch.utils.data import Dataset

from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import (
    ContinualDataset,
    fix_class_names_order,
    store_masked_loaders,
)
from utils.conf import base_path


METADATA_FILENAME = "seq_inaturalist_metadata.json"
DEFAULT_IMAGE_SIZE = (32, 32)


def _load_metadata(root: Path) -> dict:
    metadata_path = root / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. Run scripts/download_inaturalist.py first."
        )
    return json.loads(metadata_path.read_text())


class MyINaturalist(Dataset):
    """Thin wrapper around the metadata produced by download_inaturalist.py."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        target_transform=None,
        class_filter: Optional[Sequence[int]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split.lower()
        if self.split not in {"train", "test", "val"}:
            raise ValueError("Split must be 'train' or 'test'.")

        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = transforms.ToTensor()
        self.class_filter = tuple(class_filter) if class_filter is not None else None

        metadata = _load_metadata(self.root)
        self.metadata = metadata
        path_key = "train_paths" if self.split == "train" else "val_paths"
        label_key = "train_labels" if self.split == "train" else "val_labels"

        if path_key not in metadata or label_key not in metadata:
            raise KeyError(f"Missing {path_key}/{label_key} in metadata.")

        self.data = np.array(metadata[path_key], dtype=object)
        self.targets = np.array(metadata[label_key], dtype=np.int64)
        if len(self.data) != len(self.targets):
            raise RuntimeError("Mismatch between image paths and labels in metadata.")

        if self.class_filter is not None:
            allowed = np.array(self.class_filter, dtype=np.int64)
            mask = np.isin(self.targets, allowed)
            self.data = self.data[mask]
            self.targets = self.targets[mask]

            remap = {orig: new for new, orig in enumerate(allowed)}
            try:
                self.targets = np.array([remap[int(target)] for target in self.targets], dtype=np.int64)
            except KeyError as exc:  # pragma: no cover - defensive, should not happen
                raise KeyError(f"Target {exc} not present in provided class filter.") from exc

            class_names = metadata.get("class_names", [])
            if class_names:
                self.class_names = [class_names[idx] for idx in allowed]
            else:
                self.class_names = []
            self.num_classes = len(allowed)
        else:
            self.class_names = metadata.get("class_names", [])
            self.num_classes = metadata.get("num_classes", len(set(self.targets.tolist())))

    @staticmethod
    def get_class_names(root: str, class_filter: Optional[Sequence[int]] = None):
        metadata = _load_metadata(Path(root))
        names = metadata.get("class_names", [])
        if class_filter is None or not names:
            return names

        filtered_names = []
        max_idx = len(names)
        for cls_idx in class_filter:
            if cls_idx >= max_idx or cls_idx < 0:
                raise IndexError(f"Class index {cls_idx} out of range for class list of length {max_idx}.")
            filtered_names.append(names[cls_idx])
        return filtered_names

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple:
        data_entry = self.data[index]
        if isinstance(data_entry, (str, Path)):
            img_path = self.root / data_entry
            if not img_path.exists():
                raise FileNotFoundError(f"Image {img_path} not found.")
            img = Image.open(img_path).convert("RGB")
        elif isinstance(data_entry, np.ndarray):
            arr = data_entry
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                img = Image.fromarray(arr, mode="L").convert("RGB")
            else:
                img = Image.fromarray(arr, mode="RGB")
        elif torch.is_tensor(data_entry):
            img = transforms.ToPILImage()(data_entry.cpu())
        else:
            raise TypeError(f"Unsupported data type for entry {type(data_entry)} at index {index}.")
        img = img.resize(DEFAULT_IMAGE_SIZE, RESIZE_INTERPOLATION)
        target = int(self.targets[index])

        not_aug_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self.not_aug_transform(img.copy())

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.long)

        if self.split != "train":
            return img, target

        if hasattr(self, "logits"):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialINaturalist(ContinualDataset):
    """Sequential iNaturalist dataset."""

    NAME = "seq-inaturalist"
    SETTING = "class-il"
    N_TASKS = 10
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

    def get_data_loaders(self):
        root = base_path() + "inaturalist"
        try:
            train_dataset = MyINaturalist(root, split="train", transform=self.TRANSFORM)
            test_dataset = MyINaturalist(root, split="test", transform=self.TEST_TRANSFORM)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{exc}. Please run `python scripts/download_inaturalist.py` before launching training."
            ) from exc

        if train_dataset.num_classes != self.N_CLASSES:
            raise ValueError(
                f"Metadata reports {train_dataset.num_classes} classes but dataset is configured for {self.N_CLASSES}. "
                "Regenerate metadata with the desired class count (see scripts/download_inaturalist.py)."
            )

        train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)
        return train_loader, test_loader

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names

        root = base_path() + "inaturalist"
        classes = MyINaturalist.get_class_names(root)

        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialINaturalist.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet50"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialINaturalist.MEAN, SequentialINaturalist.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialINaturalist.MEAN, SequentialINaturalist.STD)

    @set_default_from_args("batch_size")
    def get_batch_size(self):
        return 128

    @set_default_from_args("n_epochs")
    def get_epochs(self):
        return 70
