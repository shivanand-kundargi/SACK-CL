"""Sequential CIFAR-100 evaluation on CIFAR-100-C corruptions.

This dataset mirrors :mod:`datasets.seq_cifar100` for training but swaps the
test loader with a corruption-specific subset from CIFAR-100-C. It allows
reusing checkpoints trained on the standard sequential CIFAR-100 stream while
probing robustness under distribution shift.
"""

import logging
import os
from typing import Tuple

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR100
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.functional import InterpolationMode

from datasets.seq_cifar100 import MyCIFAR100
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              fix_class_names_order,
                                              store_masked_loaders)
from datasets.utils import set_default_from_args
from utils.conf import base_path
from utils.prompt_templates import templates


class CIFAR100C(VisionDataset):
    """Thin wrapper around the CIFAR-100-C corruption benchmark."""

    CORRUPTIONS = (
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
        "gaussian_blur",
        "saturate",
        "spatter",
        "speckle_noise",
    )

    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        root = os.path.expanduser(root)
        labels_path = os.path.join(root, "labels.npy")
        corruption_path = os.path.join(root, f"{corruption}.npy")

        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"CIFAR-100-C labels not found at '{labels_path}'. "
                "Please download the corruption benchmark (see scripts/download_cifar100c.py)."
            )

        if corruption not in self.CORRUPTIONS:
            raise ValueError(f"Unknown corruption '{corruption}'. "
                             f"Available options: {', '.join(self.CORRUPTIONS)}")
        if not os.path.exists(corruption_path):
            raise FileNotFoundError(
                f"Corruption file '{corruption_path}' not found. "
                "Ensure CIFAR-100-C is downloaded and extracted correctly."
            )

        if not 1 <= severity <= 5:
            raise ValueError("Severity must be an integer in [1, 5].")

        logging.info("Loading CIFAR-100-C corruption '%s' (severity %d).", corruption, severity)
        labels = np.load(labels_path)
        data = np.load(corruption_path)

        if data.shape[0] != labels.shape[0]:
            raise ValueError(f"Mismatch between images ({data.shape[0]}) and labels ({labels.shape[0]}).")

        chunk = labels.shape[0] // 5
        start = (severity - 1) * chunk
        end = severity * chunk

        self.data = data[start:end]
        self.targets = labels[start:end]

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.targets)

    def __getitem__(self, index: int):  # type: ignore[override]
        image = Image.fromarray(self.data[index])
        target = int(self.targets[index])

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


class SequentialCIFAR100C(ContinualDataset):
    """Sequential CIFAR-100 with CIFAR-100-C test distribution."""

    NAME = "seq-cifar100-c"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    SIZE = (32, 32)
    MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    TRANSFORM = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    TEST_TRANSFORM = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    def __init__(
        self,
        args,
        cifar_c_corruption: str = "gaussian_noise",
        cifar_c_severity: int = 5,
    ) -> None:
        self.cifar_c_corruption = cifar_c_corruption
        self.cifar_c_severity = cifar_c_severity

        if cifar_c_corruption not in CIFAR100C.CORRUPTIONS:
            raise ValueError(
                f"Unsupported corruption '{cifar_c_corruption}'. "
                f"Valid options: {', '.join(CIFAR100C.CORRUPTIONS)}"
            )
        if not 1 <= cifar_c_severity <= 5:
            raise ValueError("The CIFAR-100-C severity must lie in [1, 5].")

        super().__init__(args)

    def get_data_loaders(self) -> Tuple:
        transform = self.TRANSFORM
        test_transform = self.TEST_TRANSFORM

        train_dataset = MyCIFAR100(
            os.path.join(base_path(), "CIFAR100"),
            train=True,
            download=True,
            transform=transform,
        )

        cifar_c_root = os.path.join(base_path(), "CIFAR-100-C")
        test_dataset = CIFAR100C(
            cifar_c_root,
            corruption=self.cifar_c_corruption,
            severity=self.cifar_c_severity,
            transform=test_transform,
        )

        train_loader, test_loader = store_masked_loaders(train_dataset, test_dataset, self)
        return train_loader, test_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR100C.TRANSFORM]
        )
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCIFAR100C.MEAN, SequentialCIFAR100C.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialCIFAR100C.MEAN, SequentialCIFAR100C.STD)

    @set_default_from_args("n_epochs")
    def get_epochs(self):
        return 50

    @set_default_from_args("batch_size")
    def get_batch_size(self):
        return 32

    @set_default_from_args("lr_scheduler")
    def get_scheduler_name(self):
        return "multisteplr"

    @set_default_from_args("lr_milestones")
    def get_lr_milestones(self):
        return [35, 45]

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = CIFAR100(os.path.join(base_path(), "CIFAR100"), train=True, download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names


class SequentialCIFAR100C224(ContinualDataset):
    """Sequential CIFAR-100-C evaluation with 224x224 resolution."""

    NAME = "seq-cifar100-c-224"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 10
    N_TASKS = 10
    N_CLASSES = 100
    SIZE = (224, 224)
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    def __init__(
        self,
        args,
        transform_type: str = "weak",
        cifar_c_corruption: str = "gaussian_noise",
        cifar_c_severity: int = 5,
    ) -> None:
        assert transform_type in {"weak", "strong"}, "Transform type must be either 'weak' or 'strong'."
        if cifar_c_corruption not in CIFAR100C.CORRUPTIONS:
            raise ValueError(
                f"Unsupported corruption '{cifar_c_corruption}'. "
                f"Valid options: {', '.join(CIFAR100C.CORRUPTIONS)}"
            )
        if not 1 <= cifar_c_severity <= 5:
            raise ValueError("The CIFAR-100-C severity must lie in [1, 5].")

        self.cifar_c_corruption = cifar_c_corruption
        self.cifar_c_severity = cifar_c_severity

        if transform_type == "strong":
            logging.info("Using strong augmentation for CIFAR-100-C 224x224")
            self.TRANSFORM = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])

        super().__init__(args)

    def get_data_loaders(self) -> Tuple:
        train_dataset = MyCIFAR100(
            os.path.join(base_path(), "CIFAR100"),
            train=True,
            download=True,
            transform=self.TRANSFORM,
        )

        cifar_c_root = os.path.join(base_path(), "CIFAR-100-C")
        test_dataset = CIFAR100C(
            cifar_c_root,
            corruption=self.cifar_c_corruption,
            severity=self.cifar_c_severity,
            transform=self.TEST_TRANSFORM,
        )

        return store_masked_loaders(train_dataset, test_dataset, self)

    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.ToPILImage(),
            SequentialCIFAR100C224.TRANSFORM,
        ])

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCIFAR100C224.MEAN, SequentialCIFAR100C224.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialCIFAR100C224.MEAN, SequentialCIFAR100C224.STD)

    @set_default_from_args("n_epochs")
    def get_epochs(self):
        return 20

    @set_default_from_args("batch_size")
    def get_batch_size(self):
        return 128

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = CIFAR100(os.path.join(base_path(), "CIFAR100"), train=True, download=True).classes
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates["cifar100"]
