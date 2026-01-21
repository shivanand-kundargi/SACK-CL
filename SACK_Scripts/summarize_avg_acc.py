#!/usr/bin/env python3
import argparse
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple

# Patterns for metrics
RE_RESULT_TASK_MEAN = re.compile(r"RESULT_task_mean_accs\s+([0-9]+(?:\.[0-9]+)?)")
RE_RESULT_CLASS_MEAN = re.compile(r"RESULT_class_mean_accs\s+([0-9]+(?:\.[0-9]+)?)")
RE_ACC_SUMMARY_TASK = re.compile(
    r"Accuracy for\s+\d+\s+task\(s\):.*?\[Task-IL\]:\s*([0-9]+(?:\.[0-9]+)?)\s*%"
)
RE_ACC_SUMMARY_CLASS = re.compile(
    r"Accuracy for\s+\d+\s+task\(s\):.*?\[Class-IL\]:\s*([0-9]+(?:\.[0-9]+)?)\s*%"
)
RE_RAW_TASK_IL = re.compile(r"Raw accuracy values:.*?Task-IL\s*\[([^\]]+)\]")
# Patterns for identifying method/seed
RE_NAMESPACE_MODEL = re.compile(r"\bmodel='([^']+)'\b")
RE_NAMESPACE_SEED = re.compile(r"\bseed=(\d+)\b")
RE_FILENAME_SEED = re.compile(r"seed[-_]?(\d+)", re.IGNORECASE)

def sanitize_method(name: str) -> str:
    if not name:
        return ""
    # Lowercase and strip non-alphanumerics
    norm = re.sub(r"[^A-Za-z0-9]+", "", name.lower())
    # Normalize common variants
    if norm == "codaprompt" or norm == "codaprompt":  # both paths normalize the same
        return "codaprompt"
    return norm

def method_from_filename(filename: str) -> Optional[str]:
    # e.g., codaprompt_last_blocks... -> codaprompt; der_layer... -> der
    m = re.match(r"([A-Za-z]+)[-_]", filename)
    if m:
        return sanitize_method(m.group(1))
    return None

def parse_seed_from_name_or_line(base: str, line: Optional[str] = None) -> Optional[int]:
    # Prefer explicit seed in content line if present
    if line:
        s = RE_NAMESPACE_SEED.search(line)
        if s:
            try:
                return int(s.group(1))
            except ValueError:
                pass
    # Fallback to filename patterns (seed0, seed-0, seed_0)
    s = RE_FILENAME_SEED.search(base)
    if s:
        try:
            return int(s.group(1))
        except ValueError:
            pass
    return None

def parse_raw_list(s: str) -> List[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            vals.append(float(x))
        except ValueError:
            pass
    return vals

def parse_log_for_metric(path: str, metric: str = "task", verbose: bool = False
) -> Tuple[Optional[float], Optional[str], Optional[int]]:
    """
    Returns (avg_acc, method, seed) for the requested metric ("task" or "class").
    Scans the entire file and uses the last seen value among multiple patterns.
    """
    avg_acc: Optional[float] = None
    method: Optional[str] = None
    seed: Optional[int] = None

    last_task_mean: Optional[float] = None
    last_class_mean: Optional[float] = None
    last_task_summary: Optional[float] = None
    last_class_summary: Optional[float] = None
    last_task_il_list: Optional[List[float]] = None
    model_name: Optional[str] = None

    base = os.path.basename(path)

    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                # model + seed from content
                if model_name is None:
                    m = RE_NAMESPACE_MODEL.search(line)
                    if m:
                        model_name = m.group(1).strip()
                if seed is None:
                    s_val = RE_NAMESPACE_SEED.search(line)
                    if s_val:
                        try:
                            seed = int(s_val.group(1))
                        except ValueError:
                            pass

                # RESULTS metrics (W&B dumps)
                m_task = RE_RESULT_TASK_MEAN.search(line)
                if m_task:
                    try:
                        last_task_mean = float(m_task.group(1))
                    except ValueError:
                        pass
                m_class = RE_RESULT_CLASS_MEAN.search(line)
                if m_class:
                    try:
                        last_class_mean = float(m_class.group(1))
                    except ValueError:
                        pass

                # Summary lines "Accuracy for N task(s)"
                m_sum_task = RE_ACC_SUMMARY_TASK.search(line)
                if m_sum_task:
                    try:
                        last_task_summary = float(m_sum_task.group(1))
                    except ValueError:
                        pass
                m_sum_class = RE_ACC_SUMMARY_CLASS.search(line)
                if m_sum_class:
                    try:
                        last_class_summary = float(m_sum_class.group(1))
                    except ValueError:
                        pass

                # Raw Task-IL list fallback
                m_raw = RE_RAW_TASK_IL.search(line)
                if m_raw:
                    vals = parse_raw_list(m_raw.group(1))
                    if vals:
                        last_task_il_list = vals

        # Determine method
        # Prefer filename prefix if present to align with method naming in results
        method_file = method_from_filename(base)
        method_model = sanitize_method(model_name or "")
        method = method_file or method_model or None

        # Seed from filename if not found in content
        if seed is None:
            seed = parse_seed_from_name_or_line(base)

        # Decide which metric to return
        if metric == "task":
            if last_task_mean is not None:
                avg_acc = last_task_mean
            elif last_task_summary is not None:
                avg_acc = last_task_summary
            elif last_task_il_list:
                avg_acc = sum(last_task_il_list) / len(last_task_il_list)
        elif metric == "class":
            if last_class_mean is not None:
                avg_acc = last_class_mean
            elif last_class_summary is not None:
                avg_acc = last_class_summary
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if verbose:
            print(f"[parse] {path}")
            print(f"        method={method} seed={seed} avg_{metric}={avg_acc}")

        return avg_acc, method, seed
    except FileNotFoundError:
        return None, None, None

def scan_results(results_dir: str, pattern_suffix: str, metric: str, verbose: bool
) -> Dict[str, Dict[int, float]]:
    """
    Walk the directory and parse logs matching suffix.
    Returns mapping: method -> { seed -> avg_acc }
    """
    summary: Dict[str, Dict[int, float]] = {}
    for root, _dirs, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith(pattern_suffix):
                continue
            fpath = os.path.join(root, fname)
            avg, method, seed = parse_log_for_metric(fpath, metric=metric, verbose=verbose)
            if avg is None or method is None or seed is None:
                continue
            summary.setdefault(method, {})[seed] = avg
    return summary

def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)  # population std

def main():
    parser = argparse.ArgumentParser(description="Summarize avg.acc per method from logs")
    parser.add_argument("--results-dir", default=os.path.join("SACK", "results"),
                        help="Root directory containing result logs (default: SACK/results)")
    parser.add_argument("--subdir", default=None,
                        help="Optional subdir under results-dir (e.g., concept_reg_layer_sweep)")
    parser.add_argument("--metric", choices=["task", "class"], default="task",
                        help="Which metric to extract (default: task)")
    parser.add_argument("--suffix", default="_output.log",
                        help="Filename suffix to match logs (default: _output.log)")
    parser.add_argument("--sort", choices=["method", "mean"], default="method",
                        help="Sort methods alphabetically or by mean accuracy (default: method)")
    parser.add_argument("--verbose", action="store_true", help="Print per-file matches")
    args = parser.parse_args()

    target_dir = args.results_dir if not args.subdir else os.path.join(args.results_dir, args.subdir)
    summary = scan_results(target_dir, pattern_suffix=args.suffix, metric=args.metric, verbose=args.verbose)

    items = []
    for method, seed_map in summary.items():
        vals = list(seed_map.values())
        m, s = mean_std(vals)
        items.append((method, seed_map, m, s))

    if args.sort == "method":
        items.sort(key=lambda x: x[0].lower())
    else:
        items.sort(key=lambda x: x[2], reverse=True)

    print(f"Results from: {os.path.abspath(target_dir)}")
    print(f"Metric: {args.metric} (mean ± std across seeds)\n")

    if not items:
        print("No matching logs found.")
        return

    for method, seed_map, m, s in items:
        seeds_str = ", ".join(f"seed{int(k)}={seed_map[k]:.2f}" for k in sorted(seed_map))
        print(f"- {method}: {seeds_str}")
        print(f"  => mean ± std: {m:.2f} ± {s:.2f} (n={len(seed_map)})\n")

if __name__ == "__main__":
    main()
