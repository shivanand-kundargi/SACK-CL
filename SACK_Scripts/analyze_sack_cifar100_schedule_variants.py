#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    from scipy.stats import ttest_rel
except Exception:
    ttest_rel = None


RE_FLOAT = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
RE_ACCMEAN = re.compile(rf"'accmean_task(\d+)':\s*(None|{RE_FLOAT})")
RE_RUN_LOG_NAME = re.compile(
    r"^cifar100-(?P<method>.+)-(?P<variant>baseline|w_to_u|u_to_w|wbar_to_u|u_to_wbar|u_to_random|u_to_random_fixed|random_to_u)-seed(?P<seed>-?\d+)\.log$"
)
RE_RAW_ACCS = re.compile(r"Raw accuracy values:\s*Class-IL\s*(\[[^\]]*\])\s*\|\s*Task-IL\s*(\[[^\]]*\])")

METHOD_ORDER = ["lwf", "icarl", "der", "derpp", "coda-prompt"]
METHOD_LABELS = {
    "lwf": "LWF",
    "icarl": "iCaRL",
    "der": "DER",
    "derpp": "DER++",
    "coda-prompt": "CODA-Prompt",
}

VARIANT_ORDER = ["baseline", "w_to_u", "u_to_w", "wbar_to_u", "u_to_wbar", "u_to_random", "u_to_random_fixed", "random_to_u"]
SACK_ONLY_VARIANTS = [variant for variant in VARIANT_ORDER if variant != "baseline"]
VARIANT_LABELS = {
    "baseline": "Baseline",
    "w_to_u": r"$w \rightarrow u$",
    "u_to_w": r"$u \rightarrow w$",
    "wbar_to_u": r"$\bar{w} \rightarrow u$",
    "u_to_wbar": r"$u \rightarrow \bar{w}$",
    "u_to_random": r"$u \rightarrow \mathrm{random}$",
    "u_to_random_fixed": r"$u \rightarrow \mathrm{random}_{\mathrm{fixed}}$",
    "random_to_u": r"$\mathrm{random} \rightarrow u$",
}
VARIANT_SHORT_LABELS = {
    "baseline": "base",
    "w_to_u": "w->u",
    "u_to_w": "u->w",
    "wbar_to_u": "wbar->u",
    "u_to_wbar": "u->wbar",
    "u_to_random": "u->rnd",
    "u_to_random_fixed": "u->rnd-fix",
    "random_to_u": "rnd->u",
}

SETTING_ORDER = ["class-il", "task-il"]
SETTING_LABELS = {"class-il": "CIL", "task-il": "TIL"}

LEGACY_SACK_SCORE_TYPE_MAP = {
    0: "w_to_u",
    1: "u_to_w",
    2: "u_to_random",
    3: "wbar_to_u",
    4: "u_to_wbar",
}

PALETTE = {
    "baseline": "#4D4D4D",
    "w_to_u": "#E69F00",
    "u_to_w": "#0072B2",
    "wbar_to_u": "#D55E00",
    "u_to_wbar": "#009E73",
    "u_to_random": "#CC79A7",
    "u_to_random_fixed": "#7A68A6",
    "random_to_u": "#7B61FF",
}

FIG7_REF_WIDTH_PT = 381.6
FIG7_REF_HEIGHT_PT = 215.04
FIG7_REF_WIDTH_IN = FIG7_REF_WIDTH_PT / 72.0
FIG7_REF_HEIGHT_IN = FIG7_REF_HEIGHT_PT / 72.0


@dataclass
class RunRecord:
    setting: str
    dataset: str
    method: str
    variant: str
    seed: int
    avg_acc: float
    bwt: float


@dataclass
class SummaryRecord:
    setting: str
    method: str
    variant: str
    seeds: List[int]
    avg_acc_mean: float
    avg_acc_std: float
    bwt_mean: float
    bwt_std: float
    runs_by_seed: Dict[int, RunRecord]


@dataclass
class TrajectoryRunRecord:
    dataset: str
    method: str
    variant: str
    seed: int
    class_il_avg_by_experience: List[float]


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def resolve_method_order(summary: Dict[Tuple[str, str, str], SummaryRecord]) -> List[str]:
    available_methods = {method for _, method, _ in summary.keys()}
    ordered_methods = [method for method in METHOD_ORDER if method in available_methods]
    ordered_methods.extend(sorted(available_methods - set(ordered_methods)))
    return ordered_methods


def normalize_method(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    method = value.strip().lower().replace("_", "-")
    if method in {"codaprompt", "coda-prompt"}:
        return "coda-prompt"
    return method


def normalize_variant(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    variant = value.strip().lower().replace("-", "_")
    if variant in VARIANT_ORDER:
        return variant
    return None


def canonicalize_variant_for_analysis(variant: Optional[str]) -> Optional[str]:
    normalized = normalize_variant(variant)
    if normalized is None:
        return None
    if normalized == "u_to_random":
        return None
    if normalized == "u_to_random_fixed":
        return "u_to_random"
    return normalized


def extract_string(line: str, key: str) -> Optional[str]:
    match = re.search(rf"'{re.escape(key)}':\s*'([^']*)'", line)
    return match.group(1) if match else None


def extract_int(line: str, key: str) -> Optional[int]:
    match = re.search(rf"'{re.escape(key)}':\s*(-?\d+)", line)
    return int(match.group(1)) if match else None


def extract_float_or_none(line: str, key: str) -> float:
    match = re.search(rf"'{re.escape(key)}':\s*(None|{RE_FLOAT})", line)
    if not match or match.group(1) == "None":
        return math.nan
    return float(match.group(1))


def extract_final_avg_acc(line: str) -> float:
    best_task = None
    best_value = math.nan
    for task_idx_str, value_str in RE_ACCMEAN.findall(line):
        if value_str == "None":
            continue
        task_idx = int(task_idx_str)
        if best_task is None or task_idx > best_task:
            best_task = task_idx
            best_value = float(value_str)
    return best_value


def resolve_variant(line: str) -> Optional[str]:
    effective_variant = canonicalize_variant_for_analysis(extract_string(line, "sack_effective_variant"))
    if effective_variant is not None:
        return effective_variant

    cog_cl = extract_int(line, "cog_cl")
    if cog_cl == 0:
        return "baseline"

    schedule_variant = canonicalize_variant_for_analysis(extract_string(line, "sack_schedule_variant"))
    if schedule_variant is not None:
        return schedule_variant

    sack_scores_type = extract_int(line, "sack_scores_type")
    if sack_scores_type in LEGACY_SACK_SCORE_TYPE_MAP:
        return canonicalize_variant_for_analysis(LEGACY_SACK_SCORE_TYPE_MAP[sack_scores_type])
    return None


def parse_log_line(line: str, setting: str, dataset_hint: str, method_hint: str) -> Optional[RunRecord]:
    dataset = extract_string(line, "dataset") or dataset_hint
    method = normalize_method(extract_string(line, "model") or method_hint)
    if dataset not in {"seq-cifar100", "seq-cifar100-224"} or method not in METHOD_ORDER:
        return None

    seed = extract_int(line, "seed")
    if seed is None:
        return None

    variant = resolve_variant(line)
    if variant not in VARIANT_ORDER:
        return None

    avg_acc = extract_final_avg_acc(line)
    if not math.isfinite(avg_acc):
        return None

    bwt = extract_float_or_none(line, "backward_transfer")
    return RunRecord(
        setting=setting,
        dataset=dataset,
        method=method,
        variant=variant,
        seed=seed,
        avg_acc=avg_acc,
        bwt=bwt,
    )


def backward_transfer_from_matrix(results: List[List[float]]) -> float:
    n_tasks = len(results)
    if n_tasks <= 1:
        return math.nan
    final_row = results[-1]
    if len(final_row) < n_tasks:
        return math.nan
    deltas = []
    for i in range(n_tasks - 1):
        if len(results[i]) <= i:
            return math.nan
        deltas.append(final_row[i] - results[i][i])
    if not deltas:
        return math.nan
    return float(np.mean(np.asarray(deltas, dtype=float)))


def split_attempt_sections(lines: List[str]) -> List[List[str]]:
    start_indices = [idx for idx, line in enumerate(lines) if line.startswith("=== launcher attempt ")]
    if not start_indices:
        return [lines]
    sections: List[List[str]] = []
    for section_index, start_idx in enumerate(start_indices):
        end_idx = start_indices[section_index + 1] if section_index + 1 < len(start_indices) else len(lines)
        sections.append(lines[start_idx:end_idx])
    return sections


def resolve_dataset_from_section(section_lines: List[str]) -> Optional[str]:
    section_text = "\n".join(section_lines)
    command_match = re.search(r"--dataset=([^\s]+)", section_text)
    if command_match:
        return command_match.group(1)
    namespace_match = re.search(r"dataset='([^']+)'", section_text)
    if namespace_match:
        return namespace_match.group(1)
    return None


def parse_run_log(log_path: Path) -> List[RunRecord]:
    name_match = RE_RUN_LOG_NAME.match(log_path.name)
    if not name_match:
        return []

    method = normalize_method(name_match.group("method"))
    variant = canonicalize_variant_for_analysis(name_match.group("variant"))
    seed = int(name_match.group("seed"))
    if method not in METHOD_ORDER or variant not in VARIANT_ORDER:
        return []

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    completed_sections = [
        section for section in split_attempt_sections(lines)
        if any("Logging results and arguments in " in line for line in section)
    ]
    if not completed_sections:
        return []

    section_lines = completed_sections[-1]
    dataset = resolve_dataset_from_section(section_lines)
    if dataset not in {"seq-cifar100", "seq-cifar100-224"}:
        return []

    class_results: List[List[float]] = []
    task_results: List[List[float]] = []
    for line in section_lines:
        raw_match = RE_RAW_ACCS.search(line)
        if not raw_match:
            continue
        try:
            class_row = [float(value) for value in ast.literal_eval(raw_match.group(1))]
            task_row = [float(value) for value in ast.literal_eval(raw_match.group(2))]
        except Exception:
            continue
        if not class_row or not task_row:
            continue
        class_results.append(class_row)
        task_results.append(task_row)

    if not class_results or not task_results:
        return []

    final_class = class_results[-1]
    final_task = task_results[-1]
    return [
        RunRecord(
            setting="class-il",
            dataset=dataset,
            method=method,
            variant=variant,
            seed=seed,
            avg_acc=float(np.mean(np.asarray(final_class, dtype=float))),
            bwt=backward_transfer_from_matrix(class_results),
        ),
        RunRecord(
            setting="task-il",
            dataset=dataset,
            method=method,
            variant=variant,
            seed=seed,
            avg_acc=float(np.mean(np.asarray(final_task, dtype=float))),
            bwt=backward_transfer_from_matrix(task_results),
        ),
    ]


def parse_run_log_trajectory(log_path: Path) -> Optional[TrajectoryRunRecord]:
    name_match = RE_RUN_LOG_NAME.match(log_path.name)
    if not name_match:
        return None

    method = normalize_method(name_match.group("method"))
    variant = canonicalize_variant_for_analysis(name_match.group("variant"))
    seed = int(name_match.group("seed"))
    if method not in METHOD_ORDER or variant not in VARIANT_ORDER:
        return None

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    completed_sections = [
        section for section in split_attempt_sections(lines)
        if any("Logging results and arguments in " in line for line in section)
    ]
    if not completed_sections:
        return None

    section_lines = completed_sections[-1]
    dataset = resolve_dataset_from_section(section_lines)
    if dataset not in {"seq-cifar100", "seq-cifar100-224"}:
        return None

    class_results: List[List[float]] = []
    for line in section_lines:
        raw_match = RE_RAW_ACCS.search(line)
        if not raw_match:
            continue
        try:
            class_row = [float(value) for value in ast.literal_eval(raw_match.group(1))]
        except Exception:
            continue
        if not class_row:
            continue
        class_results.append(class_row)

    if not class_results:
        return None

    class_il_avg_by_experience = [
        float(np.mean(np.asarray(class_row, dtype=float)))
        for class_row in class_results
    ]
    return TrajectoryRunRecord(
        dataset=dataset,
        method=method,
        variant=variant,
        seed=seed,
        class_il_avg_by_experience=class_il_avg_by_experience,
    )


def scan_runs_from_logs_pyd(results_root: Path) -> List[RunRecord]:
    deduped: Dict[Tuple[str, str, str, int], RunRecord] = {}
    for setting in SETTING_ORDER:
        setting_dir = results_root / setting
        if not setting_dir.exists():
            continue
        for log_path in setting_dir.glob("**/logs.pyd"):
            dataset_hint = log_path.parent.parent.name
            method_hint = log_path.parent.name
            with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    record = parse_log_line(line, setting=setting, dataset_hint=dataset_hint, method_hint=method_hint)
                    if record is None:
                        continue
                    deduped[(record.setting, record.method, record.variant, record.seed)] = record
    return list(deduped.values())


def scan_runs(results_root: Path) -> List[RunRecord]:
    run_log_dir = results_root / "run_logs"
    if run_log_dir.exists():
        run_log_records: List[RunRecord] = []
        for log_path in sorted(run_log_dir.glob("*.log")):
            run_log_records.extend(parse_run_log(log_path))
        if run_log_records:
            return run_log_records
    return scan_runs_from_logs_pyd(results_root)


def scan_cil_trajectories(results_root: Path) -> List[TrajectoryRunRecord]:
    run_log_dir = results_root / "run_logs"
    if not run_log_dir.exists():
        return []

    trajectories: List[TrajectoryRunRecord] = []
    for log_path in sorted(run_log_dir.glob("*.log")):
        record = parse_run_log_trajectory(log_path)
        if record is not None:
            trajectories.append(record)
    return trajectories


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def summarize_runs(runs: Iterable[RunRecord]) -> Dict[Tuple[str, str, str], SummaryRecord]:
    grouped: Dict[Tuple[str, str, str], Dict[int, RunRecord]] = defaultdict(dict)
    for run in runs:
        grouped[(run.setting, run.method, run.variant)][run.seed] = run

    summary: Dict[Tuple[str, str, str], SummaryRecord] = {}
    for (setting, method, variant), runs_by_seed in grouped.items():
        ordered_seeds = sorted(runs_by_seed)
        avg_acc_values = [runs_by_seed[seed].avg_acc for seed in ordered_seeds]
        bwt_values = [runs_by_seed[seed].bwt for seed in ordered_seeds]
        avg_acc_mean, avg_acc_std = mean_std(avg_acc_values)
        bwt_mean, bwt_std = mean_std(bwt_values)
        summary[(setting, method, variant)] = SummaryRecord(
            setting=setting,
            method=method,
            variant=variant,
            seeds=ordered_seeds,
            avg_acc_mean=avg_acc_mean,
            avg_acc_std=avg_acc_std,
            bwt_mean=bwt_mean,
            bwt_std=bwt_std,
            runs_by_seed=runs_by_seed,
        )
    return summary


def paired_pvalue(baseline_runs: Dict[int, RunRecord],
                  variant_runs: Dict[int, RunRecord],
                  metric: str) -> float:
    shared_seeds = sorted(set(baseline_runs) & set(variant_runs))
    if len(shared_seeds) < 2:
        return math.nan

    baseline_values = np.asarray([getattr(baseline_runs[seed], metric) for seed in shared_seeds], dtype=float)
    variant_values = np.asarray([getattr(variant_runs[seed], metric) for seed in shared_seeds], dtype=float)
    mask = np.isfinite(baseline_values) & np.isfinite(variant_values)
    baseline_values = baseline_values[mask]
    variant_values = variant_values[mask]

    if baseline_values.size < 2:
        return math.nan

    diff = variant_values - baseline_values
    if np.allclose(diff, 0.0):
        return 1.0
    if diff.size > 1 and np.isclose(np.std(diff, ddof=1), 0.0):
        return 0.0
    if ttest_rel is None:
        return math.nan

    result = ttest_rel(variant_values, baseline_values)
    if result is None or result.pvalue is None:
        return math.nan
    return float(result.pvalue)


def compute_significance(summary: Dict[Tuple[str, str, str], SummaryRecord],
                         methods: List[str]) -> Dict[Tuple[str, str, str], float]:
    pvalues: Dict[Tuple[str, str, str], float] = {}
    for setting in SETTING_ORDER:
        for method in methods:
            baseline = summary.get((setting, method, "baseline"))
            if baseline is None:
                continue
            for variant in SACK_ONLY_VARIANTS:
                record = summary.get((setting, method, variant))
                if record is None:
                    continue
                pvalues[(setting, method, variant)] = paired_pvalue(
                    baseline_runs=baseline.runs_by_seed,
                    variant_runs=record.runs_by_seed,
                    metric="avg_acc",
                )
    return pvalues


def format_mean_std(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "NA"
    if not math.isfinite(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def write_seed_csv(runs: Iterable[RunRecord], output_path: Path, methods: List[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_positions = {method: index for index, method in enumerate(methods)}
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["setting", "dataset", "method", "method_label", "variant", "variant_label", "seed", "avg_acc", "bwt"])
        for run in sorted(runs, key=lambda item: (item.setting, method_positions.get(item.method, len(methods)), VARIANT_ORDER.index(item.variant), item.seed)):
            writer.writerow([
                run.setting,
                run.dataset,
                run.method,
                method_label(run.method),
                run.variant,
                VARIANT_SHORT_LABELS[run.variant],
                run.seed,
                f"{run.avg_acc:.6f}",
                "" if not math.isfinite(run.bwt) else f"{run.bwt:.6f}",
            ])


def write_summary_csv(summary: Dict[Tuple[str, str, str], SummaryRecord],
                      pvalues: Dict[Tuple[str, str, str], float],
                      output_path: Path,
                      methods: List[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "setting", "setting_label", "method", "method_label", "variant", "variant_label",
            "n_seeds", "seeds", "avg_acc_mean", "avg_acc_std", "bwt_mean", "bwt_std",
            "avg_acc_p_vs_baseline", "avg_acc_significant",
        ])
        for setting in SETTING_ORDER:
            for method in methods:
                for variant in VARIANT_ORDER:
                    record = summary.get((setting, method, variant))
                    if record is None:
                        continue
                    pvalue = pvalues.get((setting, method, variant), math.nan)
                    writer.writerow([
                        setting,
                        SETTING_LABELS[setting],
                        method,
                        method_label(method),
                        variant,
                        VARIANT_SHORT_LABELS[variant],
                        len(record.seeds),
                        ",".join(str(seed) for seed in record.seeds),
                        f"{record.avg_acc_mean:.6f}",
                        f"{record.avg_acc_std:.6f}",
                        "" if not math.isfinite(record.bwt_mean) else f"{record.bwt_mean:.6f}",
                        "" if not math.isfinite(record.bwt_std) else f"{record.bwt_std:.6f}",
                        "" if not math.isfinite(pvalue) else f"{pvalue:.6f}",
                        int(math.isfinite(pvalue) and pvalue < 0.05),
                    ])


def write_summary_markdown(summary: Dict[Tuple[str, str, str], SummaryRecord],
                           pvalues: Dict[Tuple[str, str, str], float],
                           output_path: Path,
                           methods: List[str]) -> None:
    lines = [
        "# SACK CIFAR-100 Schedule Variant Summary",
        "",
        "Mean ± sample std across seeds. CIL and TIL are both produced by the same class-incremental runs in this repo.",
        "",
        "| Setting | Method | Variant | Seeds | Avg.ACC | BWT | p vs baseline |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for setting in SETTING_ORDER:
        for method in methods:
            for variant in VARIANT_ORDER:
                record = summary.get((setting, method, variant))
                if record is None:
                    continue
                pvalue = pvalues.get((setting, method, variant), math.nan)
                pvalue_str = "NA" if not math.isfinite(pvalue) else f"{pvalue:.4f}"
                lines.append(
                    f"| {SETTING_LABELS[setting]} | {method_label(method)} | {VARIANT_SHORT_LABELS[variant]} | "
                    f"{', '.join(str(seed) for seed in record.seeds)} | {format_mean_std(record.avg_acc_mean, record.avg_acc_std)} | "
                    f"{format_mean_std(record.bwt_mean, record.bwt_std)} | {pvalue_str} |"
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def summarize_experience_trajectory(records: List[TrajectoryRunRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_len = max((len(record.class_il_avg_by_experience) for record in records), default=0)
    if max_len == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)

    means: List[float] = []
    stds: List[float] = []
    xs: List[float] = []
    for exp_idx in range(max_len):
        values = np.asarray(
            [
                record.class_il_avg_by_experience[exp_idx]
                for record in records
                if len(record.class_il_avg_by_experience) > exp_idx
            ],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        xs.append(float(exp_idx + 1))
        means.append(float(values.mean()))
        stds.append(0.0 if values.size == 1 else float(values.std(ddof=1)))
    return np.asarray(xs, dtype=float), np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def plot_cil_grouped_bars(summary: Dict[Tuple[str, str, str], SummaryRecord],
                          output_path: Path,
                          methods: List[str]) -> None:
    configure_matplotlib()

    setting = "class-il"
    plot_variants = [variant for variant in VARIANT_ORDER if any((setting, method, variant) in summary for method in methods)]
    if not plot_variants:
        return

    fig_width = max(FIG7_REF_WIDTH_IN, 1.15 * len(methods) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, FIG7_REF_HEIGHT_IN))

    centers = np.arange(len(methods), dtype=float)
    bar_width = min(0.11, 0.78 / max(1, len(plot_variants)))
    offsets = (np.arange(len(plot_variants), dtype=float) - (len(plot_variants) - 1) / 2.0) * bar_width

    ymin = math.inf
    ymax = -math.inf
    for variant_index, variant in enumerate(plot_variants):
        xs: List[float] = []
        ys: List[float] = []
        errs: List[float] = []
        for method_index, method in enumerate(methods):
            record = summary.get((setting, method, variant))
            if record is None:
                continue
            xs.append(float(centers[method_index] + offsets[variant_index]))
            ys.append(record.avg_acc_mean)
            errs.append(0.0 if not math.isfinite(record.avg_acc_std) else record.avg_acc_std)
            ymin = min(ymin, record.avg_acc_mean - errs[-1])
            ymax = max(ymax, record.avg_acc_mean + errs[-1])

        if not xs:
            continue
        ax.bar(
            xs,
            ys,
            width=bar_width * 0.92,
            color=PALETTE[variant],
            edgecolor="black",
            linewidth=0.35,
            yerr=errs,
            error_kw={"elinewidth": 0.6, "capsize": 1.8},
            label=VARIANT_LABELS[variant],
            zorder=3,
        )

    if not math.isfinite(ymin) or not math.isfinite(ymax):
        ymin, ymax = 0.0, 100.0
    else:
        ymin = max(0.0, math.floor(ymin / 5.0) * 5.0 - 1.0)
        ymax = min(100.0, math.ceil(ymax / 5.0) * 5.0 + 2.0)

    ax.set_xticks(centers, [method_label(method) for method in methods])
    ax.set_ylabel("CIL Avg.ACC (%)")
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        ncol=min(4, max(1, len(plot_variants))),
        frameon=True,
        framealpha=0.95,
        edgecolor="#D0D0D0",
    )

    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.22, top=0.94)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def plot_cil_experience_trajectories(trajectories: List[TrajectoryRunRecord],
                                     output_path: Path,
                                     methods: List[str]) -> None:
    configure_matplotlib()

    grouped: Dict[Tuple[str, str], List[TrajectoryRunRecord]] = defaultdict(list)
    for record in trajectories:
        grouped[(record.method, record.variant)].append(record)

    plot_variants = ["baseline"] + [variant for variant in SACK_ONLY_VARIANTS if any((method, variant) in grouped for method in methods)]
    if not plot_variants:
        return

    n_methods = len(methods)
    ncols = 1
    nrows = n_methods
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(FIG7_REF_WIDTH_IN, FIG7_REF_HEIGHT_IN * max(1, nrows)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    all_means: List[float] = []
    max_experience = 0
    for method in methods:
        for variant in plot_variants:
            xs, means, stds = summarize_experience_trajectory(grouped.get((method, variant), []))
            if xs.size == 0:
                continue
            all_means.extend((means - stds).tolist())
            all_means.extend((means + stds).tolist())
            max_experience = max(max_experience, int(xs[-1]))

    if all_means:
        y_min = max(0.0, math.floor(min(all_means) / 5.0) * 5.0 - 2.0)
        y_max = min(100.0, math.ceil(max(all_means) / 5.0) * 5.0 + 2.0)
    else:
        y_min, y_max = 0.0, 100.0

    for ax_index, method in enumerate(methods):
        ax = axes[ax_index, 0]

        for variant in plot_variants:
            records = grouped.get((method, variant), [])
            xs, means, stds = summarize_experience_trajectory(records)
            if xs.size == 0:
                continue

            is_baseline = variant == "baseline"
            ax.plot(
                xs,
                means,
                color=PALETTE[variant],
                linewidth=1.1 if is_baseline else 1.4,
                linestyle=(0, (4, 2)) if is_baseline else "-",
                label=VARIANT_LABELS[variant],
                zorder=4 if not is_baseline else 3,
            )
            ax.scatter(
                [xs[-1]],
                [means[-1]],
                color=PALETTE[variant],
                s=14 if is_baseline else 18,
                zorder=5,
            )

        ax.set_xlim(1, max_experience if max_experience > 0 else 10)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(np.arange(1, max_experience + 1))
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        if ax_index == nrows - 1:
            ax.set_xlabel("Experience")
        ax.set_ylabel("CIL Avg.ACC (%)")

    if nrows == 1:
        axes[0, 0].legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=PALETTE[variant],
                    linewidth=1.1 if variant == "baseline" else 1.4,
                    linestyle=(0, (4, 2)) if variant == "baseline" else "-",
                    marker="o",
                    markersize=4.5,
                    label=VARIANT_LABELS[variant],
                )
                for variant in plot_variants
            ],
            loc="upper center",
            ncol=3,
            frameon=True,
            framealpha=0.95,
            edgecolor="#D0D0D0",
        )
        fig.subplots_adjust(left=0.12, right=0.985, bottom=0.18, top=0.965)
    else:
        fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=PALETTE[variant],
                    linewidth=1.1 if variant == "baseline" else 1.4,
                    linestyle=(0, (4, 2)) if variant == "baseline" else "-",
                    marker="o",
                    markersize=4.5,
                    label=VARIANT_LABELS[variant],
                )
                for variant in plot_variants
            ],
            labels=[VARIANT_LABELS[variant] for variant in plot_variants],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
        )
        fig.subplots_adjust(left=0.12, right=0.985, bottom=0.07, top=0.96, hspace=0.30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def build_best_variant_matrix(summary: Dict[Tuple[str, str, str], SummaryRecord],
                              methods: List[str]) -> Tuple[np.ndarray, np.ndarray, List[List[Tuple[str, float]]]]:
    columns = [
        ("class-il", "avg_acc"),
        ("class-il", "bwt"),
        ("task-il", "avg_acc"),
        ("task-il", "bwt"),
    ]
    values = np.full((len(methods), len(columns)), np.nan, dtype=float)
    normalized = np.full_like(values, np.nan)
    annotations: List[List[Tuple[str, float]]] = [[("NA", math.nan) for _ in columns] for _ in methods]

    for row_index, method in enumerate(methods):
        for col_index, (setting, metric_name) in enumerate(columns):
            best_variant = None
            best_value = -math.inf
            for variant in SACK_ONLY_VARIANTS:
                record = summary.get((setting, method, variant))
                if record is None:
                    continue
                metric_value = record.avg_acc_mean if metric_name == "avg_acc" else record.bwt_mean
                if not math.isfinite(metric_value):
                    continue
                if best_variant is None or metric_value > best_value:
                    best_variant = variant
                    best_value = metric_value
            if best_variant is None:
                continue
            values[row_index, col_index] = best_value
            annotations[row_index][col_index] = (best_variant, best_value)

    for col_index in range(values.shape[1]):
        column = values[:, col_index]
        finite_mask = np.isfinite(column)
        if not finite_mask.any():
            continue
        finite_values = column[finite_mask]
        col_min = float(finite_values.min())
        col_max = float(finite_values.max())
        if math.isclose(col_min, col_max):
            normalized[finite_mask, col_index] = 0.5
        else:
            normalized[finite_mask, col_index] = (finite_values - col_min) / (col_max - col_min)

    return values, normalized, annotations


def plot_best_variant_heatmap(summary: Dict[Tuple[str, str, str], SummaryRecord],
                              output_path: Path,
                              methods: List[str]) -> None:
    configure_matplotlib()
    values, normalized, annotations = build_best_variant_matrix(summary, methods)

    fig_height = max(2.5, 0.42 * len(methods) + 0.9)
    fig, ax = plt.subplots(figsize=(6.2, fig_height), constrained_layout=True)
    masked = np.ma.masked_invalid(normalized)
    image = ax.imshow(masked, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(
        np.arange(4),
        ["CIL-ACC", "CIL-BWT", "TIL-ACC", "TIL-BWT"],
    )
    ax.set_yticks(np.arange(len(methods)), [method_label(method) for method in methods])
    ax.set_title("Best SACK Variant Pattern")

    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(len(methods)):
        for col_index in range(4):
            variant, metric_value = annotations[row_index][col_index]
            if variant == "NA":
                ax.text(col_index, row_index, "NA", ha="center", va="center", color="black")
                continue
            ax.text(
                col_index,
                row_index - 0.12,
                VARIANT_SHORT_LABELS[variant],
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="black",
            )
            ax.text(
                col_index,
                row_index + 0.18,
                f"{metric_value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Column-wise normalized performance")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Aggregate and plot CIFAR-100 SACK schedule variant results.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "data" / "results" / "sack_cifar100_schedule_variants_standard",
        help="Root directory containing the class-il/ and task-il/ results trees.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for aggregated CSV/Markdown summaries and the PDF figures. Defaults to <results-root>/analysis.",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir is not None else (results_root / "analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = scan_runs(results_root)
    if not runs:
        raise SystemExit(f"No CIFAR-100 runs found under {results_root}")

    summary = summarize_runs(runs)
    methods = resolve_method_order(summary)
    if not methods:
        raise SystemExit(f"No supported methods found under {results_root}")
    pvalues = compute_significance(summary, methods)

    write_seed_csv(runs, output_dir / "seed_metrics.csv", methods)
    write_summary_csv(summary, pvalues, output_dir / "summary_metrics.csv", methods)
    write_summary_markdown(summary, pvalues, output_dir / "summary_metrics.md", methods)
    plot_cil_grouped_bars(summary, output_dir / "plot_a_cil_grouped_avg_acc.pdf", methods)
    plot_best_variant_heatmap(summary, output_dir / "plot_b_best_variant_heatmap.pdf", methods)

    print(f"Wrote analysis outputs to: {output_dir}")


if __name__ == "__main__":
    main()
