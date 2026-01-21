#!/usr/bin/env python3
"""
Utility for measuring training GFLOPs from existing Mammoth/SACK checkpoints.

This script rebuilds the dataset and model defined in a checkpoint, runs a warm-up
pass (optional), restores the weights, and then profiles a single training iteration
using `torch.profiler` with FLOP counting enabled. Multiple checkpoints can be
benchmarked in one invocation by passing repeated `--profile` arguments.

The intent is to compare configurations such as iCaRL versus SACK concept-guided
training without re-running full training jobs.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

HAS_TORCH_PROFILER = False
HAS_FVCORE = False
try:
    from torch.profiler import ProfilerActivity, profile
    HAS_TORCH_PROFILER = True
except ImportError:
    from torch.autograd import profiler as autograd_profiler  # type: ignore[attr-defined]
    try:
        from fvcore.nn import FlopCountAnalysis
        HAS_FVCORE = True
    except ImportError:
        HAS_FVCORE = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone import get_backbone  # noqa: E402
from datasets import get_dataset  # noqa: E402
from datasets.utils.continual_dataset import ContinualDataset  # noqa: E402
from main import check_args, extend_args, lecun_fix  # noqa: E402
from models import get_model  # noqa: E402
from models.utils.continual_model import ContinualModel  # noqa: E402
from utils.checkpoints import mammoth_load_checkpoint  # noqa: E402
from utils.conf import base_path as set_base_path  # noqa: E402
from utils.conf import get_device, set_random_seed  # noqa: E402
from utils.training import _to_device  # noqa: E402


def _parse_profile_entry(entry: str) -> Tuple[str, str]:
    """
    Splits a CLI entry of the form `label=checkpoint_path`.
    """
    if '=' not in entry:
        raise argparse.ArgumentTypeError(f"Invalid profile entry '{entry}'. Use the form label=/path/to/checkpoint.pt")
    label, path = entry.split('=', 1)
    label = label.strip()
    path = path.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"Invalid profile entry '{entry}'. Label cannot be empty.")
    if not path:
        raise argparse.ArgumentTypeError(f"Invalid profile entry '{entry}'. Path cannot be empty.")
    return label, path


def _load_checkpoint_args(path: str) -> Namespace:
    """
    Loads the saved CLI arguments from a Mammoth checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
    payload = torch.load(path, map_location="cpu")
    if "args" not in payload:
        raise ValueError(f"Checkpoint '{path}' does not contain saved arguments.")
    saved_args = payload["args"]
    if isinstance(saved_args, dict):
        saved_args = Namespace(**saved_args)
    if not isinstance(saved_args, Namespace):
        raise TypeError(f"Unexpected args type '{type(saved_args)}' in checkpoint '{path}'.")
    return saved_args


def _clone_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Deep-copies a module state dict onto CPU to allow later restoration.
    """
    state_dict = {}
    for k, v in module.state_dict().items():
        state_dict[k] = v.detach().cpu().clone()
    return state_dict


def _restore_state_dict(module: torch.nn.Module, snapshot: Dict[str, torch.Tensor]) -> None:
    """
    Restores a module state dict snapshot that was cloned with `_clone_state_dict`.
    """
    params = list(module.parameters())
    device = params[0].device if params else torch.device("cpu")
    restored = {k: v.to(device) for k, v in snapshot.items()}
    module.load_state_dict(restored)


def _prepare_environment(
    checkpoint_args: Namespace,
    *,
    checkpoint_path: str,
    forced_device: Optional[str],
    skip_weights: bool,
) -> Tuple[ContinualModel, ContinualDataset, Namespace]:
    """
    Recreates the dataset and model described in a checkpoint.
    """
    args = copy.deepcopy(checkpoint_args)

    # Disable external logging to avoid side effects.
    for attr in ("nowand", "disable_log", "non_verbose", "debug_mode", "inference_only"):
        if hasattr(args, attr):
            setattr(args, attr, 1 if attr in ("nowand", "disable_log", "non_verbose") else 0)

    lecun_fix()

    base = getattr(args, "base_path", "./data/")
    set_base_path(base)

    device_request = forced_device if forced_device is not None else getattr(args, "device", None)
    device = get_device(avail_devices=device_request) if isinstance(device_request, str) or device_request is None else torch.device(device_request)
    args.device = device

    if getattr(args, "seed", None) is not None:
        set_random_seed(args.seed)

    dataset = get_dataset(args)
    extend_args(args, dataset)
    check_args(args, dataset=dataset)

    backbone = get_backbone(args)
    loss_fn = dataset.get_loss()
    model = get_model(args, backbone, loss_fn, dataset.get_transform(), dataset=dataset)
    model.net.to(model.device)

    if not skip_weights:
        args.loadcheck = checkpoint_path
        mammoth_load_checkpoint(args, model)

    model.train()
    return model, dataset, args


def _prepare_task_loader(dataset: ContinualDataset, task_index: int) -> torch.utils.data.DataLoader:
    """
    Advances the dataset iterator to the requested task and returns its training loader.
    """
    if task_index < 0 or task_index >= dataset.N_TASKS:
        raise ValueError(f"Task index {task_index} is outside the valid range [0, {dataset.N_TASKS - 1}].")

    # Each call to get_data_loaders() advances the internal task counter.
    for _ in range(task_index):
        dataset.get_data_loaders()

    train_loader, _ = dataset.get_data_loaders()
    return train_loader


def _iterate_batches(loader: torch.utils.data.DataLoader) -> Iterable[Tuple]:
    """
    Infinite iterator over a dataloader.
    """
    while True:
        for batch in loader:
            yield batch


def _unpack_batch(batch: Tuple, loader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Moves a batch produced by Mammoth datasets to the target device.
    """
    inputs, labels, not_aug_inputs = batch[0], batch[1], batch[2]
    inputs = inputs.to(device)
    labels = labels.to(device, dtype=torch.long)
    not_aug_inputs = not_aug_inputs.to(device)

    extra_fields: Dict[str, torch.Tensor] = {}
    extra_names = getattr(loader.dataset, "extra_return_fields", [])
    for offset in range(len(batch) - 3):
        name = extra_names[offset] if offset < len(extra_names) else f"extra_{offset}"
        extra_fields[name] = _to_device(name, batch[3 + offset], device)
    return inputs, labels, not_aug_inputs, extra_fields


def _determine_activities(device: torch.device) -> List[ProfilerActivity]:
    """
    Chooses profiler activities based on the target device.
    """
    if not HAS_TORCH_PROFILER:
        raise RuntimeError("Torch profiler activities requested but torch.profiler is unavailable.")
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def _measure_gflops(
    model: ContinualModel,
    loader: torch.utils.data.DataLoader,
    *,
    warmup_iters: int,
    profile_iters: int,
) -> Tuple[float, List[float]]:
    """
    Profiles one or more training iterations and returns averaged GFLOPs.
    """
    if profile_iters <= 0:
        raise ValueError("profile_iters must be a positive integer.")

    batch_stream = _iterate_batches(loader)

    initial_state = _clone_state_dict(model)
    initial_opt_state = copy.deepcopy(model.opt.state_dict()) if hasattr(model, "opt") and model.opt is not None else None

    for _ in range(max(warmup_iters, 0)):
        batch = next(batch_stream)
        inputs, labels, not_aug_inputs, extra = _unpack_batch(batch, loader, model.device)
        model.meta_observe(inputs, labels, not_aug_inputs, epoch=0, **extra)

    # Restore weights and optimizer before the profiled iterations.
    _restore_state_dict(model, initial_state)
    if initial_opt_state is not None and hasattr(model, "opt") and model.opt is not None:
        model.opt.load_state_dict(initial_opt_state)
        model.opt.zero_grad(set_to_none=True)

    gflops_per_iter: List[float] = []
    if HAS_TORCH_PROFILER:
        activities = _determine_activities(model.device)
        for _ in range(profile_iters):
            batch = next(batch_stream)
            inputs, labels, not_aug_inputs, extra = _unpack_batch(batch, loader, model.device)
            with profile(activities=activities, record_shapes=False, with_flops=True) as prof:
                model.meta_observe(inputs, labels, not_aug_inputs, epoch=0, **extra)
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            flop_total = 0
            for evt in prof.key_averages():
                if getattr(evt, "flops", None):
                    flop_total += evt.flops
            gflops_per_iter.append(flop_total / 1e9)
    else:
        if not HAS_FVCORE:
            raise RuntimeError(
                "torch.profiler is unavailable in this PyTorch build and fvcore is not installed. "
                "Install fvcore (`pip install fvcore`) or upgrade PyTorch to enable FLOP counting."
            )
        backward_multiplier = 3.0  # approximate forward+backward+optimizer cost
        for _ in range(profile_iters):
            batch = next(batch_stream)
            inputs, labels, not_aug_inputs, extra = _unpack_batch(batch, loader, model.device)

            # Estimate forward FLOPs on a CPU copy of the network.
            net_copy = copy.deepcopy(model.net).to("cpu")
            net_copy.eval()
            sample_inputs = inputs.detach().cpu()
            with torch.no_grad():
                analysis = FlopCountAnalysis(net_copy, sample_inputs)
                forward_flops = analysis.total()
            del net_copy

            if forward_flops <= 0:
                raise RuntimeError("FLOP analysis returned zero or negative value; ensure fvcore supports the current backbone.")

            total_flops = forward_flops * backward_multiplier
            gflops_per_iter.append(total_flops / 1e9)

            # Execute the actual training iteration under the legacy profiler for timing consistency.
            with autograd_profiler.profile(use_cuda=model.device.type == "cuda"):
                model.meta_observe(inputs, labels, not_aug_inputs, epoch=0, **extra)
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)

    avg_gflops = sum(gflops_per_iter) / len(gflops_per_iter)
    return avg_gflops, gflops_per_iter


def _format_iterations(values: List[float]) -> str:
    return ", ".join(f"{v:.3f}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile training GFLOPs for Mammoth/SACK checkpoints.")
    parser.add_argument(
        "--profile",
        action="append",
        required=True,
        help="Profile entry in the form label=/path/to/checkpoint.pt. Repeat for multiple checkpoints.",
    )
    parser.add_argument("--task-index", type=int, default=0, help="Zero-based task index to profile (default: 0).")
    parser.add_argument("--warmup-iters", type=int, default=1, help="Number of warm-up iterations before profiling (default: 1).")
    parser.add_argument("--profile-iters", type=int, default=1, help="Number of iterations to profile and average (default: 1).")
    parser.add_argument("--device", type=str, default=None, help="Override device selection (e.g., '0' or 'cuda:0').")
    parser.add_argument("--skip-checkpoint-weights", action="store_true", help="Do not load weights from the checkpoint (structure only).")
    parser.add_argument("--save-json", type=str, default=None, help="Optional path to store the profiling summary as JSON.")
    args = parser.parse_args()

    entries = [_parse_profile_entry(entry) for entry in args.profile]

    results: List[Dict[str, object]] = []

    for label, ckpt_path in entries:
        ckpt_args = _load_checkpoint_args(ckpt_path)
        model, dataset, effective_args = _prepare_environment(
            ckpt_args,
            checkpoint_path=ckpt_path,
            forced_device=args.device,
            skip_weights=args.skip_checkpoint_weights,
        )

        train_loader = _prepare_task_loader(dataset, args.task_index)

        # Align the continual model with the requested task and epoch structure.
        model._current_task = args.task_index  # noqa: SLF001
        model.meta_begin_task(dataset)
        model.begin_epoch(0, dataset)

        avg_gflops, raw_iters = _measure_gflops(
            model,
            train_loader,
            warmup_iters=args.warmup_iters,
            profile_iters=args.profile_iters,
        )

        model.end_epoch(0, dataset)
        model.meta_end_task(dataset)

        per_sample = avg_gflops / train_loader.batch_size
        print(f"[{label}] task={args.task_index} iter GFLOPs={avg_gflops:.4f} (per-sample {per_sample:.4f}) | runs: { _format_iterations(raw_iters) }")

        results.append(
            {
                "label": label,
                "checkpoint": ckpt_path,
                "task_index": args.task_index,
                "batch_size": train_loader.batch_size,
                "average_gflops": avg_gflops,
                "per_sample_gflops": per_sample,
                "runs": raw_iters,
                "device": str(effective_args.device),
            }
        )

    if args.save_json is not None:
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved profiling summary to {args.save_json}")


if __name__ == "__main__":
    main()
