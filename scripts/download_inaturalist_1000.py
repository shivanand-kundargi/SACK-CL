#!/usr/bin/env python3
"""
Utility script to download a 1000-class subset of the iNaturalist dataset and
generate the metadata required by the SequentialINaturalist1000 loader.

The script mirrors ``download_inaturalist.py`` but uses different defaults:
- stores data under ``./data/inaturalist1000`` (configurable via ``--root``)
- retains 1000 classes by default
- writes a companion ``inaturalist1000_classes.txt`` file listing class names
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from torchvision.datasets import INaturalist


DEFAULT_TRAIN_VERSION = "2021_train"
DEFAULT_VAL_VERSION = "2021_valid"
DEFAULT_NUM_CLASSES = 1000
METADATA_FILENAME = "seq_inaturalist_metadata.json"
CLASS_LIST_FILENAME = "inaturalist1000_classes.txt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download iNaturalist and prepare metadata for the 1000-class sequential loader.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./data/inaturalist1000"),
        help="Directory where the dataset and metadata will be stored.",
    )
    parser.add_argument(
        "--train-version",
        default=DEFAULT_TRAIN_VERSION,
        help=f"Training split version to download (default: {DEFAULT_TRAIN_VERSION}).",
    )
    parser.add_argument(
        "--val-version",
        default=DEFAULT_VAL_VERSION,
        help=f"Validation split version to download (default: {DEFAULT_VAL_VERSION}).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help=f"Number of classes to retain (default: {DEFAULT_NUM_CLASSES}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild metadata even if it already exists.",
    )
    return parser.parse_args()


def _format_class_name(raw: str) -> str:
    """Return a readable class name from the raw directory string."""
    parts = raw.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    if not parts:
        return raw
    return " ".join(parts)


def _summarize_classes(
    indices: Iterable[Tuple[int, str]], num_to_keep: int
) -> Tuple[List[int], Dict[int, int]]:
    """Return a deterministic subset of class ids and its mapping to [0, ..]."""
    counts = Counter(cat_id for cat_id, _ in indices)
    if len(counts) < num_to_keep:
        logging.warning(
            "Requested %d classes but only %d found; reducing.",
            num_to_keep,
            len(counts),
        )
    selected = sorted(counts.keys())[:num_to_keep]
    mapping = {orig: idx for idx, orig in enumerate(selected)}
    return selected, mapping


def _collect_split(
    dataset: INaturalist,
    class_mapping: Dict[int, int],
    version: str,
) -> Tuple[List[str], List[int]]:
    """Collect relative paths and remapped labels for a dataset split."""
    paths: List[str] = []
    labels: List[int] = []
    version_path = Path(version)

    for cat_id, file_name in dataset.index:
        new_label = class_mapping.get(cat_id)
        if new_label is None:
            continue
        category_dir = dataset.all_categories[cat_id]
        rel_path = version_path / category_dir / file_name
        paths.append(str(rel_path))
        labels.append(new_label)

    return paths, labels


def _write_class_names(root: Path, class_names: List[str]) -> None:
    """Persist the class names to plain-text files (one per line)."""
    serialized = "\n".join(class_names) + "\n"
    root.mkdir(parents=True, exist_ok=True)

    class_path = root / CLASS_LIST_FILENAME
    class_path.write_text(serialized)
    logging.info("Class-name list written to %s.", class_path)

    # Mirror the file to the parent directory for quick inspection.
    mirror_path = root.parent / CLASS_LIST_FILENAME
    mirror_path.write_text(serialized)
    logging.info("Class-name list mirrored to %s.", mirror_path)


def _build_metadata(
    root: Path,
    train_version: str,
    val_version: str,
    num_classes: int,
    force: bool,
) -> None:
    """Download the dataset splits (if missing) and generate metadata."""
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / METADATA_FILENAME
    if metadata_path.exists() and not force:
        logging.info(
            "Metadata already exists at %s; skipping (use --force to rebuild).",
            metadata_path,
        )
        return

    logging.info("Loading training split %s...", train_version)
    train_ds = INaturalist(
        root=root,
        version=train_version,
        download=True,
        target_type="full",
    )
    logging.info("Loading validation split %s...", val_version)
    val_ds = INaturalist(
        root=root,
        version=val_version,
        download=True,
        target_type="full",
    )

    train_selected, class_mapping = _summarize_classes(train_ds.index, num_classes)
    val_classes = {cat_id for cat_id, _ in val_ds.index}
    missing_in_val = [cls for cls in train_selected if cls not in val_classes]
    if missing_in_val:
        logging.warning(
            "Validation split is missing %d of the selected classes; removing them.",
            len(missing_in_val),
        )
        for cls in missing_in_val:
            class_mapping.pop(cls, None)
        remapped = sorted(class_mapping.keys())
        class_mapping = {orig: idx for idx, orig in enumerate(remapped)}

    train_paths, train_labels = _collect_split(train_ds, class_mapping, train_version)
    val_paths, val_labels = _collect_split(val_ds, class_mapping, val_version)

    if not train_paths:
        raise RuntimeError(
            "No training samples retained; please check the requested configuration."
        )

    class_names = [
        _format_class_name(train_ds.category_name("full", orig))
        for orig, _ in sorted(class_mapping.items(), key=lambda item: item[1])
    ]

    mapping_as_list = [
        {"original": int(orig), "mapped": int(new)}
        for orig, new in sorted(class_mapping.items(), key=lambda item: item[1])
    ]
    metadata = {
        "train_version": train_version,
        "val_version": val_version,
        "num_classes": len(class_mapping),
        "class_mapping": mapping_as_list,
        "class_names": class_names,
        "train_paths": train_paths,
        "train_labels": train_labels,
        "val_paths": val_paths,
        "val_labels": val_labels,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))
    logging.info(
        "Metadata written to %s (train samples: %d, val samples: %d, classes: %d).",
        metadata_path,
        len(train_paths),
        len(val_paths),
        len(class_mapping),
    )

    _write_class_names(root, class_names)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _build_metadata(
        root=args.root,
        train_version=args.train_version,
        val_version=args.val_version,
        num_classes=args.num_classes,
        force=args.force,
    )


if __name__ == "__main__":
    main()
