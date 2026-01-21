#!/usr/bin/env python3
"""
Compute Expected Calibration Error (ECE) per experience by reusing Mammoth checkpoints.

This utility mirrors the standard Mammoth CLI so it can be executed with the same
arguments used during training. For each requested experience it:
  1. Loads the corresponding checkpoint from disk.
  2. Runs inference on that experience's test loader to gather logits and labels.
  3. Stores the collected logits (and labels) to disk for future reuse.
  4. Computes and reports the ECE value.

Results for all processed experiences are written to a JSON summary file together
with per-experience metadata.
"""

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

# Ensure repo root is on the path (mirrors main.py behaviour).
MAMMOTH_ROOT = os.path.dirname(os.path.abspath(__file__))
if MAMMOTH_ROOT not in sys.path:
    sys.path.insert(0, MAMMOTH_ROOT)

torch.set_num_threads(2)

from backbone import get_backbone  # noqa: E402
from datasets import get_dataset  # noqa: E402
from main import add_help, check_args, extend_args, lecun_fix, load_configs  # noqa: E402
from models import get_all_models, get_model  # noqa: E402
from utils import setup_logging  # noqa: E402
from utils.args import (  # noqa: E402
    add_configuration_args,
    add_dynamic_parsable_args,
    add_experiment_args,
    add_initial_args,
    add_management_args,
    check_multiple_defined_arg_during_string_parse,
    clean_dynamic_args,
    get_single_arg_value,
    update_cli_defaults,
)
from utils.checkpoints import mammoth_load_checkpoint  # noqa: E402
from utils.conf import base_path, get_device  # noqa: E402

setup_logging()


def _parse_task_selection(selection: Optional[str], max_tasks: int) -> List[int]:
    """Parse a comma-separated list / range specification into task indices."""
    if selection is None or selection.strip().lower() == 'all' or selection.strip() == '':
        return list(range(max_tasks))

    indices = set()
    for chunk in selection.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start_str, end_str = chunk.split('-', 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                start, end = end, start
            for idx in range(start, end + 1):
                indices.add(idx)
        else:
            indices.add(int(chunk))

    valid = sorted(idx for idx in indices if 0 <= idx < max_tasks)
    if not valid:
        raise ValueError(f"No valid task indices extracted from '{selection}'.")
    return valid


def _resolve_checkpoint_path(args, task_idx: int) -> str:
    """Resolve the checkpoint path for a given experience index."""
    candidate = args.checkpoint_template.format(prefix=args.checkpoint_prefix, task=task_idx, task1=task_idx + 1)
    if not os.path.isabs(candidate):
        candidate = os.path.join(args.checkpoint_dir, candidate)

    if os.path.exists(candidate):
        return candidate

    import glob

    glob_pattern = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}*_{task_idx}.pt")
    matches = glob.glob(glob_pattern)
    if not matches:
        return candidate

    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _logits_path(args, task_idx: int, width: int) -> str:
    """Build the on-disk path for storing logits for a given experience."""
    prefix_slug = os.path.basename(args.checkpoint_prefix).replace(' ', '_')
    filename = f"{prefix_slug}_exp{task_idx:0{width}d}_logits.pt"
    return os.path.join(args.logits_dir, filename)


def _expected_calibration_error(logits: torch.Tensor, labels: torch.Tensor, n_bins: int) -> float:
    """Compute the expected calibration error for the provided logits and labels."""
    if logits.numel() == 0 or labels.numel() == 0:
        return float('nan')

    probs = torch.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    bin_boundaries = torch.linspace(0.0, 1.0, steps=n_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)

    for bin_idx in range(n_bins):
        bin_lower = bin_boundaries[bin_idx]
        bin_upper = bin_boundaries[bin_idx + 1]
        if bin_idx == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if not torch.any(in_bin):
            continue

        prop = in_bin.float().mean()
        conf_avg = confidences[in_bin].mean()
        acc_avg = accuracies[in_bin].mean()
        ece += torch.abs(acc_avg - conf_avg) * prop

    return ece.item()

def _prepare_model_for_checkpoint(model, checkpoint_path: str) -> None:
    """Ensure model attributes expected by saved checkpoints are instantiated."""
    name = getattr(model, 'NAME', '').lower()
    if name == 'icarl':
        if getattr(model, 'old_net', None) is None:
            model.old_net = deepcopy(model.net)
        expected_len = None
        buffer_payload = None
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                state_dict = ckpt.get('model', ckpt)
                classes_buf = state_dict.get('classes_so_far')
                if isinstance(classes_buf, torch.Tensor):
                    expected_len = classes_buf.numel()
                if isinstance(ckpt, dict):
                    buffer_payload = ckpt.get('buffer')
            except Exception:
                expected_len = None
                buffer_payload = None
        if expected_len is None:
            expected_len = model.num_classes if hasattr(model, 'num_classes') else 0
        if expected_len < 0:
            expected_len = 0
        if (not hasattr(model, 'classes_so_far') or model.classes_so_far is None
                or model.classes_so_far.numel() != expected_len):
            buffer = torch.zeros(expected_len, dtype=torch.long)
            model.register_buffer('classes_so_far', buffer, persistent=True)
        if buffer_payload is not None:
            try:
                model.load_buffer(buffer_payload)
            except Exception:
                logging.debug('Failed to hydrate buffer via load_buffer from %s', checkpoint_path)
                try:
                    from utils.buffer import Buffer
                    buf_size = buffer_payload['examples'].shape[0]
                    new_buffer = Buffer(buf_size)
                    for attr_name, value in buffer_payload.items():
                        setattr(new_buffer, attr_name, value)
                    new_buffer.num_seen_examples = buffer_payload['examples'].shape[0]
                    new_buffer.attributes = list(buffer_payload.keys())
                    model.buffer = new_buffer
                except Exception:
                    logging.warning('Could not rebuild buffer from %s', checkpoint_path)

def _ensure_icarl_class_means_from_checkpoint(model, checkpoint_path: str, batch_size: int = 256) -> bool:
    """If iCaRL class means are missing and the buffer isn't hydrated, compute class means
    directly from the serialized buffer stored in the checkpoint.

    Returns True if class means were computed and set, else False.
    """
    try:
        if getattr(model, 'NAME', '').lower() != 'icarl':
            return False
        if getattr(model, 'class_means', None) is not None:
            return True
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            return False
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        buf = ckpt.get('buffer') if isinstance(ckpt, dict) else None
        if not isinstance(buf, dict) or 'examples' not in buf or 'labels' not in buf:
            return False
        examples = buf['examples']  # [N,C,H,W] on CPU
        labels = buf['labels'].long()
        if examples.numel() == 0:
            return False
        device = model.device
        norm = model.dataset.get_normalization_transform()
        # Move once and (lightly) batch the feature extraction
        examples = examples.to(device)
        labels = labels.to(device)
        try:
            examples = norm(examples)
        except Exception:
            # Fallback: apply per-sample if needed
            examples = torch.stack([norm(x) for x in examples], dim=0)
        feats_list = []
        with torch.no_grad():
            for start in range(0, examples.shape[0], batch_size):
                x = examples[start:start+batch_size]
                try:
                    z = model.net(x, returnt='features')
                except Exception:
                    z = model.net(x)
                z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
                z = z.view(z.size(0), -1)
                feats_list.append(z)
        feats = torch.cat(feats_list, dim=0)
        # Determine class order
        if hasattr(model, 'classes_so_far') and isinstance(model.classes_so_far, torch.Tensor) and model.classes_so_far.numel() > 0:
            class_ids = model.classes_so_far.to(device)
        else:
            class_ids = torch.sort(labels.unique()).values
        means = []
        for c in class_ids:
            mask = labels == c
            if mask.any():
                means.append(feats[mask].mean(dim=0, keepdim=True))
            else:
                # If no exemplars for this class, use zeros to preserve alignment
                means.append(torch.zeros(1, feats.shape[1], device=device))
        class_means = torch.cat(means, dim=0)
        model.class_means = class_means.squeeze()
        return True
    except Exception:
        return False


def _collect_logits(model, dataset, task_idx: int, loader: Iterable[Sequence[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference for a single experience and gather logits + labels."""
    was_training = model.net.training
    model.net.eval()

    # Ensure per-experience task context for models that rely on offsets/current task (e.g., CodaPrompt)
    try:
        start_c, end_c = dataset.get_offsets(task_idx)
    except Exception:
        start_c, end_c = 0, getattr(dataset, 'N_CLASSES', 0)
    # Set internal task index and seen/past classes if present
    try:
        model._current_task = task_idx
    except Exception:
        pass
    try:
        model._n_past_classes, model._n_seen_classes = start_c, end_c
        model._n_remaining_classes = model.N_CLASSES - end_c
    except Exception:
        pass
    # Some models expect explicit offsets
    try:
        model.offset_1 = start_c
        model.offset_2 = end_c
    except Exception:
        pass

    logits_accumulator: List[torch.Tensor] = []
    labels_accumulator: List[torch.Tensor] = []

    n_seen_classes = dataset.N_CLASSES
    if 'class-il' in dataset.SETTING or 'task-il' in dataset.SETTING:
        _, seen = dataset.get_offsets(task_idx)
        n_seen_classes = seen

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(model.device)
            labels = labels.to(model.device, dtype=torch.long)

            if 'class-il' not in model.COMPATIBILITY and 'general-continual' not in model.COMPATIBILITY:
                outputs = model(inputs, task_idx)
            else:
                if getattr(model.args, 'eval_future', False) and task_idx >= model.current_task:
                    outputs = model.future_forward(inputs)
                else:
                    outputs = model(inputs)

            outputs = outputs[:, :n_seen_classes]

            labeled_mask = labels != -1
            if not torch.all(labeled_mask):
                outputs = outputs[labeled_mask]
                labels = labels[labeled_mask]

            if outputs.numel() == 0:
                continue

            logits_accumulator.append(outputs.detach().cpu())
            labels_accumulator.append(labels.detach().cpu())

    model.net.train(was_training)

    if not logits_accumulator:
        return torch.empty((0, n_seen_classes), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    logits_tensor = torch.cat(logits_accumulator, dim=0).to(dtype=torch.float32)
    labels_tensor = torch.cat(labels_accumulator, dim=0).to(dtype=torch.long)
    return logits_tensor, labels_tensor


def parse_ece_args() -> argparse.Namespace:
    """Parse command-line arguments while reusing Mammoth's CLI helpers."""
    check_multiple_defined_arg_during_string_parse()

    parser = argparse.ArgumentParser(
        description='SACK_ECE - Evaluate Expected Calibration Error per experience.',
        allow_abbrev=False,
        add_help=False
    )

    add_initial_args(parser)
    initial_args = parser.parse_known_args()[0]

    if initial_args.backbone is None:
        logging.warning('No backbone specified. Using default backbone defined by the dataset/model configuration.')

    add_configuration_args(parser, initial_args)
    config = load_configs(parser)

    add_help(parser)

    backbone = initial_args.backbone
    if backbone is None:
        if 'backbone' in config:
            backbone = config['backbone']
        else:
            backbone = get_single_arg_value(parser, 'backbone')
    assert backbone is not None, "Backbone could not be inferred. Please pass --backbone explicitly."

    add_dynamic_parsable_args(parser, initial_args.dataset, backbone)
    add_management_args(parser)
    add_experiment_args(parser)

    ece_group = parser.add_argument_group('ECE evaluation arguments', 'Arguments specific to ECE computation.')
    ece_group.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                           help='Directory containing per-experience checkpoints.')
    ece_group.add_argument('--checkpoint_prefix', type=str, required=True,
                           help='Prefix used when saving checkpoints (the part before "_{task}.pt").')
    ece_group.add_argument('--checkpoint_template', type=str, default='{prefix}_{task}.pt',
                           help='Template for per-experience checkpoint filenames. Supported placeholders: '
                           '{prefix}, {task} (0-based), {task1} (1-based).')
    ece_group.add_argument('--logits_dir', type=str, default='ece_logits',
                           help='Directory where logits tensors will be persisted.')
    ece_group.add_argument('--summary_path', type=str, default=None,
                           help='Optional path for the JSON summary file (defaults to <logits_dir>/ece_summary.json).')
    ece_group.add_argument('--ece_bins', type=int, default=15,
                           help='Number of confidence bins to use when computing ECE.')
    ece_group.add_argument('--overwrite_logits', action='store_true',
                           help='Regenerate logits even if the output file already exists.')
    ece_group.add_argument('--tasks', type=str, default=None,
                           help='Optional comma-separated list / ranges of tasks to evaluate (e.g. "0,2-4"). Defaults to all tasks.')
    ece_group.add_argument('--skip_missing_checkpoints', action='store_true',
                           help='Skip experiences without a matching checkpoint instead of raising an error.')

    update_cli_defaults(parser, config)

    for action in parser._actions:
        if action.default is not None and action.type is not None:
            if action.nargs is None or action.nargs == 0:
                action.default = action.type(action.default)
            else:
                if not isinstance(action.default, (list, tuple)):
                    action.default = [action.type(v) for v in action.default]

    args = parser.parse_args()
    args = clean_dynamic_args(args)

    models_dict = get_all_models()
    args.model = models_dict[args.model]

    args.nowand = 1
    args.force_compat = True
    args.disable_log = 1

    if getattr(args, 'summary_path', None) is None:
        args.summary_path = os.path.join(args.logits_dir, 'ece_summary.json')

    return args


def main() -> None:
    lecun_fix()
    args = parse_ece_args()

    device = get_device(avail_devices=args.device)
    args.device = device

    base_path(args.base_path)

    dataset = get_dataset(args)
    extend_args(args, dataset)
    check_args(args, dataset=dataset)

    os.makedirs(args.logits_dir, exist_ok=True)
    if os.path.dirname(args.summary_path):
        os.makedirs(os.path.dirname(args.summary_path), exist_ok=True)

    total_tasks = dataset.N_TASKS
    start_task = args.start_from if args.start_from is not None else 0
    end_task = args.stop_after if args.stop_after is not None else total_tasks
    end_task = min(end_task, total_tasks)
    selected_indices = _parse_task_selection(args.tasks, end_task)
    selected_indices = [idx for idx in selected_indices if start_task <= idx < end_task]
    if not selected_indices:
        raise ValueError("No tasks remain to evaluate after applying start/stop/task filters.")
    selected_set = set(selected_indices)
    digits = max(2, len(str(end_task - 1)))

    loss_fn = dataset.get_loss()
    transform = dataset.get_transform()

    results: List[dict] = []
    total_samples = 0
    weighted_ece_sum = 0.0

    logging.info("Preparing dataloaders for %d experiences.", end_task)

    for task_idx in range(end_task):
        _, test_loader = dataset.get_data_loaders()

        if task_idx not in selected_set:
            continue

        checkpoint_path = _resolve_checkpoint_path(args, task_idx)
        if not os.path.exists(checkpoint_path):
            message = f"Checkpoint for experience {task_idx} not found at '{checkpoint_path}'."
            if args.skip_missing_checkpoints:
                logging.warning("%s Skipping.", message)
                continue
            raise FileNotFoundError(message)

        model = get_model(args, get_backbone(args), loss_fn, transform, dataset=dataset)
        model.to(device)
        _prepare_model_for_checkpoint(model, checkpoint_path)

        args.loadcheck = checkpoint_path
        model, _ = mammoth_load_checkpoint(args, model, ignore_classifier=False)
        if getattr(model, 'NAME', '').lower() == 'icarl':
            _ = _ensure_icarl_class_means_from_checkpoint(model, checkpoint_path)
        buffer_missing = False
        if hasattr(model, 'buffer'):
            has_attr = hasattr(model.buffer, 'examples')
            buffer_missing = not has_attr or (has_attr and model.buffer.examples.numel() == 0)
        if buffer_missing:
            try:
                saved_obj = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                if isinstance(saved_obj, dict) and 'buffer' in saved_obj:
                    model.load_buffer(saved_obj['buffer'])
            except Exception:
                logging.warning('Could not reload buffer from %s', checkpoint_path)
        if getattr(model, 'NAME', '').lower() == 'icarl':
            has_examples = hasattr(model.buffer, 'examples') if hasattr(model, 'buffer') else False
            size = None
            if has_examples:
                try:
                    size = tuple(model.buffer.examples.shape)
                except Exception:
                    size = 'unavailable'
            logging.debug('iCaRL buffer loaded: %s size=%s', has_examples, size)

        logits_path = _logits_path(args, task_idx, digits)

        if os.path.exists(logits_path) and not args.overwrite_logits:
            cached = torch.load(logits_path, map_location='cpu')
            logits_tensor = cached['logits']
            labels_tensor = cached['labels']
            logging.info("Loaded cached logits for experience %d from %s.", task_idx, logits_path)
        else:
            logits_tensor, labels_tensor = _collect_logits(model, dataset, task_idx, test_loader)
            torch.save({'logits': logits_tensor, 'labels': labels_tensor}, logits_path)
            logging.info("Stored logits for experience %d to %s.", task_idx, logits_path)

        num_samples = int(labels_tensor.shape[0])
        if num_samples == 0:
            logging.warning("Experience %d produced no labeled samples; skipping ECE computation.", task_idx)
            continue

        ece_value = _expected_calibration_error(logits_tensor, labels_tensor, args.ece_bins)
        weighted_ece_sum += ece_value * num_samples
        total_samples += num_samples

        results.append({
            'experience': task_idx,
            'checkpoint': checkpoint_path,
            'logits_path': logits_path,
            'samples': num_samples,
            'ece': ece_value
        })

        logging.info("Experience %d | samples=%d | ECE=%.6f", task_idx, num_samples, ece_value)

        del model  # free GPU memory between iterations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("No experiences were processed; nothing to report.")

    weighted_ece = weighted_ece_sum / total_samples if total_samples else float('nan')
    summary = {
        'dataset': dataset.NAME,
        'model': args.model,
        'checkpoint_prefix': args.checkpoint_prefix,
        'bins': args.ece_bins,
        'weighted_ece': weighted_ece,
        'total_samples': total_samples,
        'experiences': results
    }

    with open(args.summary_path, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, indent=2)
    logging.info("ECE summary written to %s", args.summary_path)


if __name__ == '__main__':
    main()
