#!/usr/bin/env python3
"""Download and extract the CIFAR-100-C corruption benchmark.

The dataset is roughly 1.9 GB. By default it is stored under the Mammoth
``base_path`` so that the new CIFAR-100-C continual datasets pick it up
automatically.
"""

import argparse
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
import sys
sys.path.insert(0,"/p/lustre1/kundargi1/SACK")
from utils.conf import base_path

CIFAR100C_URL = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"


def _download(url: str, destination: Path) -> None:
    def _progress(blocks_transferred: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = blocks_transferred * block_size
        percent = min(downloaded / total_size, 1.0) * 100
        print(f"\rDownloading CIFAR-100-C: {percent:5.1f}%", end="", file=sys.stderr)

    urllib.request.urlretrieve(url, destination, _progress)
    print("\rDownload complete.          ", file=sys.stderr)


def _extract(archive_path: Path, target_dir: Path) -> None:
    print(f"Extracting archive to '{target_dir}'", file=sys.stderr)
    with tarfile.open(archive_path) as archive:
        archive.extractall(target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR-100-C into the Mammoth data directory.")
    parser.add_argument("--output", type=str, default=None,
                        help="Destination directory (defaults to <base_path>/CIFAR-100-C)")
    parser.add_argument("--url", type=str, default=CIFAR100C_URL,
                        help="Override the download URL (default: official Zenodo mirror)")
    parser.add_argument("--force", action="store_true",
                        help="Redownload and overwrite even if the dataset already exists.")
    args = parser.parse_args()

    target_root = Path(args.output) if args.output is not None else Path(base_path()) / "CIFAR-100-C"
    target_root = target_root.expanduser().resolve()

    if target_root.exists() and not args.force:
        print(f"Target '{target_root}' already exists. Use --force to overwrite.")
        return

    target_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        archive_path = Path(tmp_dir) / "cifar100c.tar"
        _download(args.url, archive_path)
        _extract(archive_path, target_root)

    print(f"CIFAR-100-C ready at '{target_root}'.")


if __name__ == "__main__":
    main()
