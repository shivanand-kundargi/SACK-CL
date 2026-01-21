#!/usr/bin/env python3
"""
Grad-CAM visualization helper for SACK experiments.

This script pairs checkpoints produced with and without SACK, extracts a target
image from the first experience, and generates Grad-CAM overlays to highlight
how attention shifts across experiences.
"""
from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch import Tensor  # noqa: E402

# Ensure repository root is on the PYTHONPATH when running the script directly.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from argparse import Namespace  # noqa: E402

from backbone import get_backbone  # noqa: E402
from datasets import get_dataset  # noqa: E402
from models import get_model  # noqa: E402
from utils.checkpoints import mammoth_load_checkpoint  # noqa: E402
from utils.conf import base_path as set_base_path  # noqa: E402

DEFAULT_LAYER_NAME = "layer4"


@dataclass
class SampleBundle:
    """Holds the tensors required for Grad-CAM processing."""

    normalized: Tensor  # shape: (1, C, H, W)
    display: Tensor  # shape: (C, H, W), values in [0, 1]
    label: int
    experience: int


@dataclass
class OverlayBundle:
    """Stores rendered overlays for a single experience."""

    experience: int
    baseline_overlay: np.ndarray  # shape: (H, W, 3)
    sack_overlay: np.ndarray  # shape: (H, W, 3)
    baseline_heatmap: np.ndarray  # shape: (H, W)
    sack_heatmap: np.ndarray  # shape: (H, W)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM comparison between SACK and baseline checkpoints.")
    parser.add_argument("--model-name", required=True, help="Model name used in checkpoint filenames (e.g., icarl, derpp).")
    parser.add_argument("--dataset", required=True, help="Continual dataset identifier used during training (e.g., seq-cifar100).")
    parser.add_argument(
        "--dataset-tag",
        required=True,
        help="Dataset slug used in checkpoint filenames (e.g., cifar100, cub200).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed encoded in the checkpoint filenames.")
    parser.add_argument(
        "--sack-tag",
        default="sack",
        help="Substring that identifies checkpoints trained with SACK inside the filename.",
    )
    parser.add_argument(
        "--baseline-tag",
        default="original",
        help="Substring that identifies checkpoints trained without SACK inside the filename.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        required=True,
        help="Class index (from experience 0) to visualize.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Select the Nth sample (0-indexed) of the target class from the chosen split.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Split to draw the reference image from for experience 0.",
    )
    parser.add_argument(
        "--layer-name",
        default=DEFAULT_LAYER_NAME,
        help="Dot-separated path to the target layer within the backbone (e.g., layer4, blocks.3).",
    )
    parser.add_argument(
        "--mode",
        choices=["gradcam", "attention", "similarity"],
        default="gradcam",
        help="Visualization mode: Grad-CAM, attention, or classifier-weight similarity.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory that contains the checkpoint files.",
    )
    parser.add_argument(
        "--output-dir",
        default="gradcam_outputs",
        help="Directory used to store generated visualizations.",
    )
    parser.add_argument(
        "--experiences",
        type=int,
        nargs="*",
        help="Optional explicit list of experience ids to visualize. Defaults to all overlapping experiences.",
    )
    parser.add_argument(
        "--max-experience",
        type=int,
        help="Optional maximum experience id (inclusive). Useful to trim long sequences.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of experiences considered (starting from the earliest).",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for inference (default: cpu).")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Opacity of the heatmap overlay when mixing with the original image.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def checkpoint_prefix(model_name: str, dataset_tag: str, variant_tag: str, seed: int) -> str:
    return f"{model_name}-{dataset_tag}-{variant_tag}-seed-{seed}"


def collect_checkpoints(directory: Path, prefix: str) -> Dict[int, Path]:
    matches: Dict[int, Path] = {}
    pattern = f"{prefix}_*.pt"
    for path in sorted(directory.glob(pattern)):
        try:
            experience = int(path.stem.split("_")[-1])
        except ValueError:
            logging.debug("Skipping checkpoint without experience suffix: %s", path.name)
            continue
        matches[experience] = path
    logging.debug("Collected %d checkpoints for prefix '%s'.", len(matches), prefix)
    return matches


def load_checkpoint_args(checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "args" not in payload:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain serialized args.")
    return deepcopy(payload["args"])


def get_submodule(module: torch.nn.Module, dotted_path: str) -> torch.nn.Module:
    current = module
    for part in dotted_path.split("."):
        if hasattr(current, "module") and part == "module":
            current = current.module
            continue
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def resolve_target_layer(net: torch.nn.Module, requested: str) -> Tuple[torch.nn.Module, str]:
    try:
        layer = get_submodule(net, requested)
        return layer, requested
    except AttributeError as err:
        available = list(net._modules.keys())
        fallback = None
        if requested == DEFAULT_LAYER_NAME:
            for candidate in ["layer4", "layer3", "layer2", "layer1"]:
                if candidate in available:
                    fallback = candidate
                    break
        if fallback is not None:
            logging.warning("Layer '%s' not found; using '%s' instead.", requested, fallback)
            layer = get_submodule(net, fallback)
            return layer, fallback
        message = (
            f"Could not find layer '{requested}'. "
            f"Top-level modules in backbone: {available}"
        )
        raise AttributeError(message) from err


def normalize_display_tensor(tensor: Tensor) -> Tensor:
    tensor = tensor.clone().detach()
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    if tensor_max - tensor_min < 1e-8:
        return torch.zeros_like(tensor)
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return tensor.clamp_(0.0, 1.0)

def _sanitize_args_dict(args_dict: dict) -> dict:
    args_dict = deepcopy(args_dict)
    if "class_order" in args_dict and not isinstance(args_dict["class_order"], np.ndarray):
        args_dict["class_order"] = np.array(args_dict["class_order"])
    return args_dict



def build_sample(args_dict: dict, class_id: int, sample_index: int, split: str, device: torch.device) -> SampleBundle:
    args_ns = Namespace(**_sanitize_args_dict(args_dict))
    args_ns.device = str(device)

    set_base_path(args_ns.base_path)

    dataset = get_dataset(args_ns)

    num_tasks = getattr(dataset, "N_TASKS", 1)
    for _ in range(num_tasks):
        train_loader, test_loader = dataset.get_data_loaders()
        loader = train_loader if split == "train" else test_loader
        data_source = loader.dataset
        experience_id = getattr(dataset, "c_task", 0)

        target_count = 0
        for idx in range(len(data_source)):
            sample = data_source[idx]
            image_tensor = sample[0]
            label = sample[1]
            raw_image = sample[2] if len(sample) > 2 and isinstance(sample[2], torch.Tensor) else None

            label_value = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
            if label_value != class_id:
                continue

            if target_count < sample_index:
                target_count += 1
                continue

            normalized = image_tensor.unsqueeze(0).detach()

            if raw_image is not None:
                display = raw_image.detach()
                if display.ndim == 4:
                    display = display[0]
            elif hasattr(dataset, "get_denormalization_transform"):
                denorm = dataset.get_denormalization_transform()
                display = denorm(image_tensor.detach().clone())
            else:
                display = normalize_display_tensor(image_tensor.detach().clone())

            display = display.clamp(0.0, 1.0)

            logging.info(
                "Selected sample idx=%d from experience %d (%s split) for class %d.",
                idx,
                experience_id,
                split,
                class_id,
            )
            return SampleBundle(normalized=normalized, display=display, label=label_value, experience=experience_id)

    raise ValueError(f"Could not find sample #{sample_index} for class {class_id} in the {split} split across all experiences.")

def build_model_template(args_dict: dict, device: torch.device, dataset_override: Optional[str]) -> Tuple[Namespace, torch.nn.Module]:
    args_ns = Namespace(**_sanitize_args_dict(args_dict))
    if dataset_override is not None:
        args_ns.dataset = dataset_override

    args_ns.device = str(device)
    set_base_path(args_ns.base_path)

    dataset = get_dataset(args_ns)
    backbone = get_backbone(args_ns)
    backbone.to(device)

    loss_fn = dataset.get_loss()
    model = get_model(args_ns, backbone, loss_fn, dataset.get_transform(), dataset=dataset)
    model.to(device)
    model.eval()
    return args_ns, model


def refresh_model_weights(args_ns: Namespace, model: torch.nn.Module, checkpoint_path: Path) -> None:
    args_ns.loadcheck = str(checkpoint_path)
    try:
        loaded_model, _ = mammoth_load_checkpoint(args_ns, model, ignore_classifier=False)
        loaded_model.eval()
        return
    except RuntimeError as err:
        logging.warning("Standard checkpoint load failed for %s: %s", checkpoint_path.name, err)

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model", payload)
    filtered_state = {
        k: v
        for k, v in state_dict.items()
        if not k.startswith("old_net.") and k not in {"classes_so_far"}
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if unexpected:
        logging.warning("Unexpected keys after fallback load: %s", unexpected)
    if missing:
        logging.warning("Missing keys after fallback load: %s", missing)
    model.eval()


def capture_activation_shape(net: torch.nn.Module, layer: torch.nn.Module, input_tensor: Tensor) -> torch.Size:
    activations: List[Tensor] = []

    def hook(_, __, output):
        activations.append(output.detach())

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            net.eval()
            out = net(input_tensor)
            if isinstance(out, tuple):
                out = out[0]
    finally:
        handle.remove()
    if not activations:
        raise RuntimeError("Could not capture activations to determine shape.")
    return activations[0].shape


def capture_activation(net: torch.nn.Module, layer: torch.nn.Module, input_tensor: Tensor) -> Tensor:
    activations: List[Tensor] = []
    def hook(_, __, output):
        activations.append(output.detach())
    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            net.eval()
            out = net(input_tensor)
            if isinstance(out, tuple):
                out = out[0]
    finally:
        handle.remove()
    if not activations:
        raise RuntimeError("Could not capture activations.")
    return activations[0]

def _channels_from_shape(shape: torch.Size) -> int:
    if len(shape) == 4:
        return shape[1]
    if len(shape) == 3:
        return shape[2]
    return -1

def suggest_fallback_layer(net: torch.nn.Module) -> Optional[str]:
    if hasattr(net, 'feat') and hasattr(getattr(net, 'feat'), 'blocks'):
        blocks = getattr(net.feat, 'blocks')
        if hasattr(blocks, '__len__') and len(blocks) > 0:
            return f"feat.blocks.{len(blocks) - 1}"
    for candidate in ('layer4', 'layer3', 'layer2', 'layer1'):
        if hasattr(net, candidate):
            return candidate
    return None


def compute_gradcam(net: torch.nn.Module, layer: torch.nn.Module, input_tensor: Tensor, class_idx: int) -> Tensor:
    gradients: List[Tensor] = []
    activations: List[Tensor] = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    handle_fwd = layer.register_forward_hook(forward_hook)
    handle_bwd = layer.register_full_backward_hook(backward_hook)

    try:
        net.zero_grad(set_to_none=True)
        net.eval()
        with torch.enable_grad():
            output = net(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            if output.ndim > 2:
                output = output.view(output.size(0), -1)

            if class_idx is None:
                class_idx = int(output.argmax(dim=1).item())

            score = output[:, class_idx].sum()
            score.backward()

        if not gradients or not activations:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")

        grad = gradients[0]
        act = activations[0]

        if act.dim() == 4:
            act_map = act
            grad_map = grad
        elif act.dim() == 3:
            # Expect shape (B, Tokens, Channels). Remove CLS token and reshape to spatial grid.
            if grad.dim() != 3:
                raise RuntimeError(f"Mismatched gradient dimension for transformer activations: {grad.shape}")
            if act.shape[1] <= 1:
                raise RuntimeError("Transformer activations missing patch tokens (only CLS present).")
            tokens = act.shape[1] - 1  # exclude CLS token
            side = int(round(tokens ** 0.5))
            if side * side != tokens:
                raise RuntimeError(f"Cannot reshape {tokens} tokens into a square grid.")
            act_map = act[:, 1:, :].permute(0, 2, 1).reshape(act.shape[0], act.shape[2], side, side)
            grad_map = grad[:, 1:, :].permute(0, 2, 1).reshape(grad.shape[0], grad.shape[2], side, side)
        else:
            raise RuntimeError(f"Unsupported activation dimensionality for Grad-CAM: {act.shape}")

        weights = grad_map.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act_map).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def compute_attention_map(net: torch.nn.Module, layer: torch.nn.Module, input_tensor: Tensor) -> Tensor:
    activations: List[Tensor] = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    handle = layer.register_forward_hook(forward_hook)
    try:
        with torch.no_grad():
            net.eval()
            output = net(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
    finally:
        handle.remove()

    if not activations:
        raise RuntimeError("Attention visualization failed: no activations captured.")

    act = activations[0]

    if act.dim() == 4:
        attn_map = act.mean(dim=1, keepdim=True)
    elif act.dim() == 3:
        if act.shape[1] <= 1:
            raise RuntimeError("Attention activations missing patch tokens (only CLS present).")
        tokens = act.shape[1] - 1
        side = int(round(tokens ** 0.5))
        if side * side != tokens:
            raise RuntimeError(f"Cannot reshape {tokens} tokens into a square grid.")
        cls_token = act[:, 0, :]
        patch_tokens = act[:, 1:, :]
        similarity = torch.einsum('bd,bpd->bp', cls_token, patch_tokens)
        similarity = similarity.view(similarity.size(0), 1, side, side)
        attn_map = similarity
    else:
        raise RuntimeError(f"Unsupported activation dimensionality for attention visualization: {act.shape}")

    attn_map = torch.nn.functional.interpolate(attn_map, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
    attn_map = attn_map.squeeze()
    attn_map = attn_map - attn_map.min()
    attn_map = attn_map / (attn_map.max() + 1e-8)
    return attn_map.detach().cpu()


def _find_classifier_linear(net: torch.nn.Module) -> nn.Linear:
    # Common names for the classification head across backbones
    for name in ("classifier", "last", "head", "fc", "linear"):
        if hasattr(net, name):
            layer = getattr(net, name)
            if isinstance(layer, nn.Linear):
                return layer
    # Fallback: search top-level modules
    for m in net._modules.values():
        if isinstance(m, nn.Linear):
            return m
    raise AttributeError("Could not locate a linear classification head on the backbone.")


def _tokens_to_map(act: Tensor, grad: Tensor = None) -> tuple[Tensor, Tensor]:
    """Convert ViT token sequences (B, N, C) to (B, C, H, W)."""
    if act.shape[1] <= 1:
        raise RuntimeError("Transformer activations missing patch tokens (only CLS present).")
    tokens = act.shape[1] - 1  # exclude CLS token
    side = int(round(tokens ** 0.5))
    if side * side != tokens:
        raise RuntimeError(f"Cannot reshape {tokens} tokens into a square grid.")
    act_map = act[:, 1:, :].permute(0, 2, 1).reshape(act.shape[0], act.shape[2], side, side)
    if grad is None:
        grad_map = None
    else:
        grad_map = grad[:, 1:, :].permute(0, 2, 1).reshape(grad.shape[0], grad.shape[2], side, side)
    return act_map, grad_map


def compute_similarity_map(net: torch.nn.Module, layer: torch.nn.Module, input_tensor: Tensor, class_idx: int) -> Tensor:
    """Classifier-weight cosine similarity per spatial location."""
    activations: list[Tensor] = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    handle = layer.register_forward_hook(forward_hook)
    try:
        with torch.no_grad():
            net.eval()
            out = net(input_tensor)
            if isinstance(out, tuple):
                out = out[0]
    finally:
        handle.remove()

    if not activations:
        raise RuntimeError("Similarity visualization failed: no activations captured.")

    act = activations[0]
    # Build spatial feature map (B, C, H, W)
    if act.dim() == 4:
        feat = act
    elif act.dim() == 3:
        feat, _ = _tokens_to_map(act, None)
    else:
        raise RuntimeError(f"Unsupported activation dimensionality for similarity visualization: {act.shape}")

    # Get classifier weight vector for the target class
    clf = _find_classifier_linear(net)
    w = clf.weight[class_idx].detach()  # (C,)
    # Cosine similarity per-location
    feat_n = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)
    w_n = w / (w.norm() + 1e-8)
    sim = (feat_n * w_n[None, :, None, None]).sum(dim=1, keepdim=True)
    # Normalize to [0, 1]
    sim = sim - sim.min()
    sim = sim / (sim.max() + 1e-8)
    # Upsample to input size
    sim = torch.nn.functional.interpolate(sim, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
    return sim.squeeze().detach().cpu()

def tensor_to_image(tensor: Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor for image conversion, got shape {tuple(tensor.shape)}")
    return tensor.permute(1, 2, 0).numpy()


def resize_image(img: np.ndarray, scale: int = 1) -> np.ndarray:
    if scale <= 1:
        return img
    return np.kron(img, np.ones((scale, scale, 1), dtype=img.dtype))


def apply_colormap(base_image: np.ndarray, heatmap: np.ndarray, alpha: float, upscale: int = 1) -> np.ndarray:
    base = resize_image(base_image, upscale)
    heat = resize_image(heatmap[..., None], upscale)[..., 0]
    cmap = plt.get_cmap("jet")
    colored = cmap(heat)[..., :3]
    overlay = alpha * colored + (1.0 - alpha) * base
    return np.clip(overlay, 0.0, 1.0)


def save_image(array: np.ndarray, output_path: Path, title: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(array)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_grid(
    original_np: np.ndarray,
    overlays: List[OverlayBundle],
    baseline_tag: str,
    sack_tag: str,
    output_path: Path,
) -> None:
    if not overlays:
        logging.warning("No overlays to plot, skipping grid generation.")
        return

    num_rows = len(overlays)
    fig, axes = plt.subplots(num_rows, 3, figsize=(9, 3 * num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, bundle in enumerate(overlays):
        ax_orig, ax_base, ax_sack = axes[row_idx]

        ax_orig.imshow(original_np)
        ax_base.imshow(bundle.baseline_overlay)
        ax_sack.imshow(bundle.sack_overlay)

        if row_idx == 0:
            ax_orig.set_title("Original")
            ax_base.set_title(f"Baseline ({baseline_tag})")
            ax_sack.set_title(f"SACK ({sack_tag})")

        ax_orig.set_ylabel(f"Task {bundle.experience}")

        for ax in (ax_orig, ax_base, ax_sack):
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def filter_experiences(
    experiences: Iterable[int],
    allowed: Optional[Iterable[int]],
    max_experience: Optional[int],
    limit: Optional[int],
) -> List[int]:
    exps = sorted(set(experiences))

    if allowed is not None:
        allowed_set = set(allowed)
        exps = [exp for exp in exps if exp in allowed_set]

    if max_experience is not None:
        exps = [exp for exp in exps if exp <= max_experience]

    if limit is not None:
        exps = exps[:limit]

    return exps


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory '{checkpoint_dir}' does not exist.")

    baseline_prefix = checkpoint_prefix(args.model_name, args.dataset_tag, args.baseline_tag, args.seed)
    sack_prefix = checkpoint_prefix(args.model_name, args.dataset_tag, args.sack_tag, args.seed)

    baseline_ckpts = collect_checkpoints(checkpoint_dir, baseline_prefix)
    sack_ckpts = collect_checkpoints(checkpoint_dir, sack_prefix)

    if not baseline_ckpts:
        raise FileNotFoundError(f"No baseline checkpoints found for prefix '{baseline_prefix}'.")
    if not sack_ckpts:
        raise FileNotFoundError(f"No SACK checkpoints found for prefix '{sack_prefix}'.")

    common_experiences = sorted(set(baseline_ckpts.keys()) & set(sack_ckpts.keys()))
    selected_experiences = filter_experiences(common_experiences, args.experiences, args.max_experience, args.limit)

    if not selected_experiences:
        raise RuntimeError("No overlapping experiences selected for visualization.")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    # Route similarity maps to a separate root folder for clarity
    if args.mode == 'similarity':
        parts = list(output_dir.parts)
        if 'gradcam_outputs' in parts:
            idx = parts.index('gradcam_outputs')
            parts[idx] = 'similarity_maps'
            output_dir = Path(*parts)
        else:
            output_dir = Path('similarity_maps') / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_exp = selected_experiences[0]
    reference_sack_ckpt = sack_ckpts[reference_exp]
    reference_args = load_checkpoint_args(reference_sack_ckpt)
    reference_args["dataset"] = args.dataset

    sample = build_sample(reference_args, args.target_class, args.sample_index, args.split, device)
    original_np = tensor_to_image(sample.display)
    save_image(original_np, output_dir / "reference_image.png", title=f"Class {sample.label}")

    baseline_args = load_checkpoint_args(baseline_ckpts[reference_exp])
    baseline_args["dataset"] = args.dataset
    baseline_ns, baseline_model = build_model_template(baseline_args, device, dataset_override=args.dataset)

    sack_ns, sack_model = build_model_template(reference_args, device, dataset_override=args.dataset)

    baseline_net = baseline_model.net.module if hasattr(baseline_model.net, "module") else baseline_model.net
    sack_net = sack_model.net.module if hasattr(sack_model.net, "module") else sack_model.net

    baseline_layer, resolved_layer_name = resolve_target_layer(baseline_net, args.layer_name)
    sack_layer, _ = resolve_target_layer(sack_net, resolved_layer_name)

    input_tensor = sample.normalized.to(device)

    try:
        activation_shape = capture_activation_shape(baseline_net, baseline_layer, input_tensor)
    except RuntimeError as err:
        logging.warning("Could not inspect activations for layer '%s': %s", resolved_layer_name, err)
        activation_shape = None

    if activation_shape is not None and len(activation_shape) < 3:
        fallback_name = suggest_fallback_layer(baseline_net)
        if fallback_name is not None and fallback_name != resolved_layer_name:
            logging.warning("Layer '%s' produces non-spatial activations; using '%s' for visualization instead.", resolved_layer_name, fallback_name)
            baseline_layer, resolved_layer_name = resolve_target_layer(baseline_net, fallback_name)
            sack_layer, _ = resolve_target_layer(sack_net, resolved_layer_name)
        else:
            logging.warning("Layer '%s' produces non-spatial activations; visualization may fail.", resolved_layer_name)

    # Ensure similarity mode has matching channels with classifier head
    if args.mode == 'similarity':
        try:
            clf = _find_classifier_linear(baseline_net)
            needed_c = clf.in_features
            act = capture_activation(baseline_net, baseline_layer, input_tensor)
            got_c = _channels_from_shape(act.shape)
            if got_c != needed_c:
                candidates: List[str] = []
                avail = list(baseline_net._modules.keys())
                # try common top-level candidates
                for cand in (resolved_layer_name, 'bn_final', 'layer4', 'layer3', 'layer2'):
                    if cand in candidates:
                        continue
                    try:
                        _ = get_submodule(baseline_net, cand)
                        candidates.append(cand)
                    except Exception:
                        pass
                # for ViT-style
                try:
                    blocks = getattr(getattr(baseline_net, 'feat'), 'blocks')
                    last_idx = len(blocks) - 1
                    vit_cand = f'feat.blocks.{last_idx}'
                    if vit_cand not in candidates:
                        _ = get_submodule(baseline_net, vit_cand)
                        candidates.append(vit_cand)
                except Exception:
                    pass
                chosen = None
                for cand in candidates:
                    try:
                        layer_try = get_submodule(baseline_net, cand)
                        shape = capture_activation_shape(baseline_net, layer_try, input_tensor)
                        ch = _channels_from_shape(shape)
                        if ch == needed_c:
                            chosen = (layer_try, cand)
                            break
                    except Exception:
                        continue
                if chosen is not None:
                    baseline_layer, resolved_layer_name = chosen
                    sack_layer, _ = resolve_target_layer(sack_net, resolved_layer_name)
                    logging.warning('Adjusted similarity layer to %s to match classifier in_features=%d.', resolved_layer_name, needed_c)
                else:
                    logging.warning('Could not find a layer with channels=%d; similarity map may fail (got %d).', needed_c, got_c)
        except Exception as e:
            logging.warning('Similarity setup check failed: %s', e)
    overlays: List[OverlayBundle] = []

    for exp in selected_experiences:
        baseline_path = baseline_ckpts[exp]
        sack_path = sack_ckpts[exp]
        logging.info("Processing experience %d (baseline: %s, SACK: %s)", exp, baseline_path.name, sack_path.name)

        refresh_model_weights(baseline_ns, baseline_model, baseline_path)
        refresh_model_weights(sack_ns, sack_model, sack_path)

        with torch.no_grad():
            baseline_model.eval()
            sack_model.eval()

        if args.mode == 'attention':
            baseline_cam = compute_attention_map(baseline_net, baseline_layer, input_tensor).numpy()
            sack_cam = compute_attention_map(sack_net, sack_layer, input_tensor).numpy()
        elif args.mode == 'similarity':
            baseline_cam = compute_similarity_map(baseline_net, baseline_layer, input_tensor, sample.label).numpy()
            sack_cam = compute_similarity_map(sack_net, sack_layer, input_tensor, sample.label).numpy()
        else:
            baseline_cam = compute_gradcam(baseline_net, baseline_layer, input_tensor, sample.label).numpy()
            sack_cam = compute_gradcam(sack_net, sack_layer, input_tensor, sample.label).numpy()

        upscale = 1
        if sample.display.shape[-1] == 32:
            upscale = 4
        baseline_overlay = apply_colormap(original_np, baseline_cam, args.alpha, upscale=upscale)
        sack_overlay = apply_colormap(original_np, sack_cam, args.alpha, upscale=upscale)

        overlays.append(
            OverlayBundle(
                experience=exp,
                baseline_overlay=baseline_overlay,
                sack_overlay=sack_overlay,
                baseline_heatmap=baseline_cam,
                sack_heatmap=sack_cam,
            )
        )

    grid_name = "gradcam_grid.png" if args.mode != 'similarity' else "similarity_grid.png"
    save_grid(original_np, overlays, args.baseline_tag, args.sack_tag, output_dir / grid_name)

    for bundle in overlays:
        exp_dir = output_dir / f"task_{bundle.experience:02d}"
        exp_dir.mkdir(exist_ok=True)
        save_image(bundle.baseline_overlay, exp_dir / "baseline_overlay.png", title=f"Task {bundle.experience} - {args.baseline_tag}")
        save_image(bundle.sack_overlay, exp_dir / "sack_overlay.png", title=f"Task {bundle.experience} - {args.sack_tag}")
        save_image(bundle.baseline_heatmap, exp_dir / "baseline_heatmap.png", title="Heatmap")
        save_image(bundle.sack_heatmap, exp_dir / "sack_heatmap.png", title="Heatmap")

    logging.info("Saved Grad-CAM visualizations in %s", output_dir)


if __name__ == "__main__":
    main()
