#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

DEFAULT_URL = "https://data.caltech.edu/api/records/65de6-vp158/files/CUB_200_2011.tgz/content"
DEFAULT_MD5 = "97eceeb196236b17998738112f37df78"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CUB-200-2011 NPZ files expected by seq-cub200.")
    parser.add_argument("--root", type=Path, default=Path("data/CUB200"),
                        help="Directory where train_data.npz/test_data.npz should live.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL,
                        help="CUB_200_2011.tgz download URL.")
    parser.add_argument("--archive", type=Path, default=None,
                        help="Use an existing CUB_200_2011.tgz archive instead of downloading.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild NPZ files even if they already exist.")
    parser.add_argument("--keep-extracted", action="store_true",
                        help="Keep the extracted CUB_200_2011 directory after writing NPZ files.")
    parser.add_argument("--skip-md5", action="store_true",
                        help="Skip archive md5 validation.")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Square image size stored in the NPZ files.")
    return parser.parse_args()


def md5sum(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as output:
            total_size = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if total_size > 0:
                    percent = 100.0 * downloaded / total_size
                    print(f"\rDownloading CUB-200-2011: {percent:5.1f}%", end="", flush=True)
            print()
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    tmp_path.replace(destination)


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if destination not in member_path.parents and member_path != destination:
                raise RuntimeError(f"Unsafe path in tar archive: {member.name}")
        tar.extractall(destination)


def read_space_mapping(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with path.open() as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            idx, value = line.split(" ", 1)
            mapping[int(idx)] = value
    return mapping


def object_array(values: Iterable[np.ndarray]) -> np.ndarray:
    values = list(values)
    array = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        array[idx] = value
    return array


def load_split(cub_dir: Path, train: bool, image_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_paths = read_space_mapping(cub_dir / "images.txt")
    class_labels = {idx: int(label) - 1 for idx, label in read_space_mapping(cub_dir / "image_class_labels.txt").items()}
    split_flags = {idx: int(flag) == 1 for idx, flag in read_space_mapping(cub_dir / "train_test_split.txt").items()}

    data: List[np.ndarray] = []
    targets: List[int] = []
    segs: List[np.ndarray] = []
    selected_ids = [idx for idx in sorted(image_paths) if split_flags[idx] == train]
    split_name = "train" if train else "test"

    for count, image_id in enumerate(selected_ids, start=1):
        rel_path = image_paths[image_id]
        image_path = cub_dir / "images" / rel_path
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
            image_array = np.asarray(image, dtype=np.uint8)
        data.append(image_array)
        targets.append(class_labels[image_id])
        segs.append(np.zeros((1, 1), dtype=np.uint8))
        if count % 500 == 0 or count == len(selected_ids):
            print(f"Prepared {count:5d}/{len(selected_ids):5d} {split_name} images")

    return np.stack(data, axis=0), np.asarray(targets, dtype=np.int64), np.stack(segs, axis=0)


def load_classes(cub_dir: Path) -> np.ndarray:
    classes = []
    for idx, raw_name in sorted(read_space_mapping(cub_dir / "classes.txt").items()):
        display_name = raw_name.split(".", 1)[-1].replace("_", " ").lower()
        classes.append((idx - 1, display_name))
    return np.asarray(classes, dtype=object)


def npz_ready(root: Path) -> bool:
    return (root / "train_data.npz").is_file() and (root / "test_data.npz").is_file()


def prepare(root: Path, url: str = DEFAULT_URL, archive: Path = None,
            force: bool = False, keep_extracted: bool = False, skip_md5: bool = False,
            image_size: int = 256) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if npz_ready(root) and not force:
        print(f"CUB-200 NPZ files already exist in {root}")
        return

    archive_path = archive or (root / "CUB_200_2011.tgz")
    if not archive_path.is_file():
        print(f"Downloading official CUB-200-2011 archive to {archive_path}")
        download(url, archive_path)

    if not skip_md5:
        observed_md5 = md5sum(archive_path)
        if observed_md5 != DEFAULT_MD5:
            raise RuntimeError(
                f"Archive md5 mismatch for {archive_path}: expected {DEFAULT_MD5}, got {observed_md5}."
            )

    cub_dir = root / "CUB_200_2011"
    extracted_here = False
    if not cub_dir.is_dir():
        print(f"Extracting {archive_path}")
        safe_extract_tar(archive_path, root)
        extracted_here = True
    if not cub_dir.is_dir():
        raise RuntimeError(f"Expected extracted directory was not found: {cub_dir}")

    classes = load_classes(cub_dir)
    for train, filename in ((True, "train_data.npz"), (False, "test_data.npz")):
        data, targets, segs = load_split(cub_dir, train=train, image_size=image_size)
        out_path = root / filename
        print(f"Writing {out_path}")
        np.savez_compressed(out_path, data=data, targets=targets, classes=classes, segs=segs)

    if extracted_here and not keep_extracted:
        print(f"Removing extracted directory {cub_dir}")
        shutil.rmtree(cub_dir)


def main() -> None:
    args = parse_args()
    prepare(
        root=args.root,
        url=args.url,
        archive=args.archive,
        force=args.force,
        keep_extracted=args.keep_extracted,
        skip_md5=args.skip_md5,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
