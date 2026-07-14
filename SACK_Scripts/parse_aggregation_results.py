#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sack_cl_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

AGGREGATION_VARIANTS = [
    "max-mean",
    "mean-mean",
    "min-mean",
    "top3-mean",
    "top5-mean",
    "softmax-sharp",
    "softmax-smooth",
    "max-max",
]
BAR_PLOT_EXCLUDED_VARIANTS = {"top3-mean", "top5-mean", "min-mean"}

FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
RE_ACC_SUMMARY = re.compile(
    rf"Accuracy for\s+(\d+)\s+task\(s\):.*?\[Class-IL\]:\s*({FLOAT_RE})\s*%.*?\[Task-IL\]:\s*({FLOAT_RE})\s*%"
)
RE_RAW_ACCS = re.compile(
    r"Raw accuracy values:\s*Class-IL\s*(\[[^\]]*\])\s*\|\s*Task-IL\s*(\[[^\]]*\])"
)
RE_PYD_ACCMEAN = re.compile(r"'accmean_task(\d+)':\s*(None|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
RE_PYD_ACCURACY = re.compile(
    r"'accuracy_(\d+)_task(\d+)':\s*(None|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
RE_SECTION_START = re.compile(r"^=== launcher attempt \d+/\d+ @ ")
RE_LOG_VARIANT = re.compile(r"^(?P<variant>[A-Za-z0-9-]+)_(?:cil|til)\.log$")
RE_BASELINE_LOG = re.compile(r"cifar100-icarl-baseline-seed(?P<seed>-?\d+)\.log$")
RE_SCHEDULE_VARIANT = re.compile(r"cifar100-icarl-(?P<variant>[A-Za-z0-9_-]+)-seed(?P<seed>-?\d+)\.log$")
RE_FIELD_STRING = {
    "dataset": re.compile(r"(?:\bdataset='([^']+)'|'dataset':\s*'([^']+)')"),
    "model": re.compile(r"(?:\bmodel='([^']+)'|'model':\s*'([^']+)')"),
    "conf_timestamp": re.compile(r"(?:\bconf_timestamp='([^']+)'|'conf_timestamp':\s*'([^']+)')"),
    "ckpt_name": re.compile(r"(?:\bckpt_name='([^']+)'|'ckpt_name':\s*'([^']+)')"),
    "sack_aggregation": re.compile(r"(?:\bsack_aggregation='([^']+)'|'sack_aggregation':\s*'([^']+)')"),
    "results_path": re.compile(r"(?:\bresults_path='([^']+)'|'results_path':\s*'([^']+)')"),
    "sack_effective_variant": re.compile(
        r"(?:\bsack_effective_variant='([^']+)'|'sack_effective_variant':\s*'([^']+)')"
    ),
}
RE_FIELD_INT = {
    "seed": re.compile(r"(?:\bseed=(\d+)|'seed':\s*(\d+))"),
}
RE_CKPT_TIMESTAMP = re.compile(r"_(\d{8}-\d{6})(?:_|$)")
OKABE_ITO = {
    "blue": "#0072B2",
    "green": "#009E73",
    "red": "#D55E00",
    "orange": "#E69F00",
    "black": "#000000",
    "gray": "#666666",
}
LIGHT_BLUE_BAND = "#BFDFF4"
NEURIPS_SINGLE_COL_WIDTH_IN = 3.25
REF_PLOT_WIDTH_PT = 381.6
REF_PLOT_HEIGHT_PT = 215.04
REF_PLOT_WIDTH_IN = REF_PLOT_WIDTH_PT / 72.0
REF_PLOT_HEIGHT_IN = REF_PLOT_HEIGHT_PT / 72.0
REF_PLOT_RCPARAMS = {
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
}


@dataclass
class RunMetrics:
    variant: str
    seed: int
    dataset: str
    source_log: Path
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    class_rows: List[List[float]]
    task_rows: List[List[float]]
    cil_by_experience_override: Optional[List[float]] = None
    til_by_experience_override: Optional[List[float]] = None
    cil_bwt_override: Optional[float] = None
    til_bwt_override: Optional[float] = None

    @property
    def cil_by_experience(self) -> List[float]:
        if self.cil_by_experience_override is not None:
            return list(self.cil_by_experience_override)
        return [float(np.mean(np.asarray(row, dtype=float))) for row in self.class_rows]

    @property
    def til_by_experience(self) -> List[float]:
        if self.til_by_experience_override is not None:
            return list(self.til_by_experience_override)
        return [float(np.mean(np.asarray(row, dtype=float))) for row in self.task_rows]

    @property
    def cil_final(self) -> float:
        values = self.cil_by_experience
        return values[-1] if values else math.nan

    @property
    def til_final(self) -> float:
        values = self.til_by_experience
        return values[-1] if values else math.nan

    @property
    def cil_bwt(self) -> float:
        if self.cil_bwt_override is not None:
            return float(self.cil_bwt_override)
        return backward_transfer_from_matrix(self.class_rows)

    @property
    def til_bwt(self) -> float:
        if self.til_bwt_override is not None:
            return float(self.til_bwt_override)
        return backward_transfer_from_matrix(self.task_rows)

    @property
    def duration_seconds(self) -> float:
        if self.start_time is None or self.end_time is None:
            return math.nan
        return max(0.0, float((self.end_time - self.start_time).total_seconds()))


def split_attempt_sections(lines: Sequence[str]) -> List[List[str]]:
    start_indices = [idx for idx, line in enumerate(lines) if RE_SECTION_START.match(line)]
    if not start_indices:
        return [list(lines)]

    sections: List[List[str]] = []
    for section_idx, start_idx in enumerate(start_indices):
        end_idx = start_indices[section_idx + 1] if section_idx + 1 < len(start_indices) else len(lines)
        sections.append(list(lines[start_idx:end_idx]))
    return sections


def last_completed_section(lines: Sequence[str]) -> List[str]:
    sections = split_attempt_sections(lines)
    completed = [section for section in sections if any("Logging results and arguments in" in line for line in section)]
    return completed[-1] if completed else sections[-1]


def parse_literal_list(raw_list: str) -> List[float]:
    values = ast.literal_eval(raw_list)
    return [float(value) for value in values]


def backward_transfer_from_matrix(results: Sequence[Sequence[float]]) -> float:
    n_tasks = len(results)
    if n_tasks <= 1:
        return math.nan

    final_row = list(results[-1])
    if len(final_row) < n_tasks:
        return math.nan

    deltas: List[float] = []
    for task_idx in range(n_tasks - 1):
        row = list(results[task_idx])
        if len(row) <= task_idx:
            return math.nan
        deltas.append(float(final_row[task_idx]) - float(row[task_idx]))

    if not deltas:
        return math.nan
    return float(np.mean(np.asarray(deltas, dtype=float)))


def extract_string(text: str, field: str) -> Optional[str]:
    match = RE_FIELD_STRING[field].search(text)
    if not match:
        return None
    for group in match.groups():
        if group is not None:
            return group
    return None


def extract_int(text: str, field: str) -> Optional[int]:
    match = RE_FIELD_INT[field].search(text)
    if not match:
        return None
    for group in match.groups():
        if group is not None:
            return int(group)
    return None


def parse_datetime(value: str) -> Optional[datetime]:
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y%m%d-%H%M%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_start_time(section_text: str) -> Optional[datetime]:
    launcher_match = re.search(r"=== launcher attempt \d+/\d+ @ (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", section_text)
    if launcher_match:
        parsed = parse_datetime(launcher_match.group(1))
        if parsed is not None:
            return parsed

    conf_timestamp = extract_string(section_text, "conf_timestamp")
    if conf_timestamp:
        parsed = parse_datetime(conf_timestamp)
        if parsed is not None:
            return parsed

    ckpt_name = extract_string(section_text, "ckpt_name")
    if ckpt_name:
        match = RE_CKPT_TIMESTAMP.search(ckpt_name)
        if match:
            parsed = parse_datetime(match.group(1))
            if parsed is not None:
                return parsed

    save_match = re.search(r"Saving checkpoint into\s+(\S+)", section_text)
    if save_match:
        match = RE_CKPT_TIMESTAMP.search(save_match.group(1))
        if match:
            return parse_datetime(match.group(1))

    return None


def parse_variant_from_text(text: str) -> Optional[str]:
    direct = extract_string(text, "sack_aggregation")
    if direct in AGGREGATION_VARIANTS:
        return direct

    effective_variant = extract_string(text, "sack_effective_variant")
    if effective_variant == "baseline":
        return "baseline"
    if effective_variant in AGGREGATION_VARIANTS:
        return effective_variant

    results_path = extract_string(text, "results_path")
    if results_path:
        for variant in AGGREGATION_VARIANTS:
            if f"/{variant}" in results_path or results_path.endswith(variant):
                return variant
        if "baseline" in results_path:
            return "baseline"

    return None


def infer_variant(log_path: Path, section_text: str) -> Optional[str]:
    file_match = RE_LOG_VARIANT.match(log_path.name)
    if file_match:
        variant = file_match.group("variant")
        if variant in AGGREGATION_VARIANTS:
            return variant

    baseline_match = RE_BASELINE_LOG.search(log_path.name)
    if baseline_match:
        return "baseline"

    schedule_match = RE_SCHEDULE_VARIANT.search(log_path.name)
    if schedule_match:
        raw_variant = schedule_match.group("variant").replace("_", "-")
        if raw_variant in AGGREGATION_VARIANTS:
            return raw_variant
        if schedule_match.group("variant") == "baseline":
            return "baseline"

    for part in log_path.parts:
        if part in AGGREGATION_VARIANTS:
            return part

    return parse_variant_from_text(section_text)


def parse_completed_run(log_path: Path, expected_variant: Optional[str] = None) -> Optional[RunMetrics]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return None

    section_lines = last_completed_section(lines)
    section_text = "\n".join(section_lines)
    if "Logging results and arguments in" not in section_text:
        return None

    variant = expected_variant or infer_variant(log_path, section_text)
    if variant is None:
        return None

    dataset = extract_string(section_text, "dataset") or "seq-cifar100"
    seed = extract_int(section_text, "seed")
    if seed is None:
        if variant == "baseline":
            baseline_name = RE_BASELINE_LOG.search(log_path.name)
            if baseline_name:
                seed = int(baseline_name.group("seed"))
        if seed is None:
            seed = 0

    class_rows_by_experience: Dict[int, List[float]] = {}
    task_rows_by_experience: Dict[int, List[float]] = {}
    current_experience: Optional[int] = None
    for line in section_lines:
        acc_match = RE_ACC_SUMMARY.search(line)
        if acc_match:
            current_experience = int(acc_match.group(1))

        raw_match = RE_RAW_ACCS.search(line)
        if not raw_match:
            continue

        try:
            class_row = parse_literal_list(raw_match.group(1))
            task_row = parse_literal_list(raw_match.group(2))
        except Exception:
            continue

        if not class_row or not task_row:
            continue

        experience = current_experience or len(class_row)
        class_rows_by_experience[experience] = class_row
        task_rows_by_experience[experience] = task_row

    if not class_rows_by_experience or not task_rows_by_experience:
        return None

    experiences = sorted(set(class_rows_by_experience) & set(task_rows_by_experience))
    class_rows = [class_rows_by_experience[exp] for exp in experiences]
    task_rows = [task_rows_by_experience[exp] for exp in experiences]

    start_time = parse_start_time(section_text)
    end_time = datetime.fromtimestamp(log_path.stat().st_mtime)
    return RunMetrics(
        variant=variant,
        seed=seed,
        dataset=dataset,
        source_log=log_path,
        start_time=start_time,
        end_time=end_time,
        class_rows=class_rows,
        task_rows=task_rows,
    )


def discover_aggregation_logs(repo_root: Path) -> Tuple[Dict[str, Path], List[Path]]:
    candidate_roots = [
        repo_root / "results" / "aggregation_ablation" / "runs",
        repo_root / "results" / "aggregation_ablation" / "logs",
    ]

    all_logs: List[Path] = []
    searched_roots: List[Path] = []
    for root in candidate_roots:
        if not root.exists():
            continue
        searched_roots.append(root)
        all_logs.extend(path for path in root.rglob("*.log") if path.is_file())

    if not all_logs:
        searched = ", ".join(str(root) for root in searched_roots or candidate_roots)
        raise FileNotFoundError(f"No aggregation ablation log files were found under: {searched}")

    selected: Dict[str, Path] = {}
    for log_path in sorted(all_logs):
        section_text = log_path.read_text(encoding="utf-8", errors="ignore")
        variant = infer_variant(log_path, section_text)
        if variant not in AGGREGATION_VARIANTS:
            continue

        current = selected.get(variant)
        if current is None:
            selected[variant] = log_path
            continue

        current_rank = 0 if current.name.endswith("_cil.log") else 1
        new_rank = 0 if log_path.name.endswith("_cil.log") else 1
        if new_rank < current_rank:
            selected[variant] = log_path
            continue
        if new_rank == current_rank and log_path.stat().st_mtime > current.stat().st_mtime:
            selected[variant] = log_path

    return selected, searched_roots


def discover_baseline_log(repo_root: Path, explicit_log: Optional[Path]) -> RunMetrics:
    if explicit_log is not None:
        run = parse_completed_run(explicit_log, expected_variant="baseline")
        if run is None:
            raise ValueError(f"Could not parse baseline log: {explicit_log}")
        return run

    search_root = repo_root / "data" / "results" / "sack_cifar100_schedule_variants_standard"
    candidates = sorted(search_root.rglob("cifar100-icarl-baseline-seed0.log"))
    parsed_candidates: List[RunMetrics] = []
    for candidate in candidates:
        run = parse_completed_run(candidate, expected_variant="baseline")
        if run is None:
            continue
        if run.dataset != "seq-cifar100":
            continue
        parsed_candidates.append(run)

    if not parsed_candidates:
        raise FileNotFoundError(
            "Could not find an iCaRL baseline log. Pass --baseline-log explicitly if it lives elsewhere."
        )

    parsed_candidates.sort(
        key=lambda run: (
            run.start_time or datetime.min,
            run.end_time or datetime.min,
            str(run.source_log),
        )
    )
    return parsed_candidates[-1]


def load_cached_run(repo_root: Path, variant: str) -> Optional[RunMetrics]:
    summary_path = repo_root / "results" / "aggregation_ablation" / "summary.csv"
    per_experience_path = repo_root / "results" / "aggregation_ablation" / "per_experience.csv"
    if not summary_path.exists() or not per_experience_path.exists():
        return None

    summary_row: Optional[Dict[str, str]] = None
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == variant:
                summary_row = row
                break
    if summary_row is None:
        return None

    cil_by_experience: Dict[int, float] = {}
    til_by_experience: Dict[int, float] = {}
    with per_experience_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") != variant:
                continue
            try:
                experience = int(row["experience"])
                avg_acc = float(row["avg_acc"])
            except (KeyError, TypeError, ValueError):
                continue
            evaluation = row.get("evaluation", "").strip().upper()
            if evaluation == "CIL":
                cil_by_experience[experience] = avg_acc
            elif evaluation == "TIL":
                til_by_experience[experience] = avg_acc

    if not cil_by_experience or not til_by_experience:
        return None

    def parse_optional_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    start_time = None
    end_time = None
    source_log = Path(summary_row.get("source_log", summary_path))
    if summary_row.get("start_time") and summary_row["start_time"] != "unknown":
        start_time = parse_datetime(summary_row["start_time"])
    if summary_row.get("end_time") and summary_row["end_time"] != "unknown":
        end_time = parse_datetime(summary_row["end_time"])

    return RunMetrics(
        variant=variant,
        seed=int(summary_row.get("seed", 0)),
        dataset=summary_row.get("dataset", "seq-cifar100"),
        source_log=source_log,
        start_time=start_time,
        end_time=end_time,
        class_rows=[],
        task_rows=[],
        cil_by_experience_override=[cil_by_experience[idx] for idx in sorted(cil_by_experience)],
        til_by_experience_override=[til_by_experience[idx] for idx in sorted(til_by_experience)],
        cil_bwt_override=parse_optional_float(summary_row.get("cil_bwt")),
        til_bwt_override=parse_optional_float(summary_row.get("til_bwt")),
    )


def parse_results_pyd_log(log_path: Path, expected_dataset: str = "seq-cifar100", expected_model: str = "icarl") -> Optional[Dict[str, object]]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    dataset = extract_string(text, "dataset")
    model = extract_string(text, "model")
    if dataset != expected_dataset or model != expected_model:
        return None

    seed = extract_int(text, "seed")
    conf_timestamp = extract_string(text, "conf_timestamp")
    means: Dict[int, float] = {}
    matrix_rows: Dict[int, Dict[int, float]] = {}

    for match in RE_PYD_ACCMEAN.finditer(text):
        task_idx = int(match.group(1))
        raw_value = match.group(2)
        if raw_value == "None":
            continue
        means[task_idx] = float(raw_value)

    for match in RE_PYD_ACCURACY.finditer(text):
        col_idx = int(match.group(1))
        row_idx = int(match.group(2))
        raw_value = match.group(3)
        if raw_value == "None":
            continue
        matrix_rows.setdefault(row_idx, {})[col_idx] = float(raw_value)

    if not means:
        return None

    max_task = max(means)
    rows: List[List[float]] = []
    for row_idx in range(1, max_task + 1):
        row_values = matrix_rows.get(row_idx, {})
        ordered_row = [row_values[col_idx] for col_idx in range(1, row_idx + 1) if col_idx in row_values]
        rows.append(ordered_row)

    start_time = parse_datetime(conf_timestamp) if conf_timestamp else None
    end_time = datetime.fromtimestamp(log_path.stat().st_mtime)
    return {
        "seed": 0 if seed is None else seed,
        "dataset": dataset,
        "start_time": start_time,
        "end_time": end_time,
        "means": [means[idx] for idx in sorted(means)],
        "rows": rows,
        "source_log": log_path,
    }


def load_run_from_results_tree(repo_root: Path, variant: str) -> Optional[RunMetrics]:
    variant_root = repo_root / "data" / "results" / "aggregation_ablation" / "runs" / variant
    class_log = variant_root / "class-il" / "seq-cifar100" / "icarl" / "logs.pyd"
    task_log = variant_root / "task-il" / "seq-cifar100" / "icarl" / "logs.pyd"
    if not class_log.exists() or not task_log.exists():
        return None

    class_record = parse_results_pyd_log(class_log)
    task_record = parse_results_pyd_log(task_log)
    if class_record is None or task_record is None:
        return None

    return RunMetrics(
        variant=variant,
        seed=int(class_record["seed"]),
        dataset=str(class_record["dataset"]),
        source_log=class_log,
        start_time=class_record["start_time"],
        end_time=class_record["end_time"],
        class_rows=list(class_record["rows"]),
        task_rows=list(task_record["rows"]),
        cil_by_experience_override=list(class_record["means"]),
        til_by_experience_override=list(task_record["means"]),
    )


def load_runs(repo_root: Path, baseline_log: Optional[Path]) -> Dict[str, RunMetrics]:
    aggregation_logs, searched_roots = discover_aggregation_logs(repo_root)
    runs_root = repo_root / "results" / "aggregation_ablation" / "runs"
    if runs_root in searched_roots and not any(runs_root in path.parents for path in aggregation_logs.values()):
        print(
            "Note: no .log files were found under results/aggregation_ablation/runs; "
            "falling back to results/aggregation_ablation/logs.",
            file=sys.stderr,
        )

    runs: Dict[str, RunMetrics] = {}
    for variant in AGGREGATION_VARIANTS:
        log_path = aggregation_logs.get(variant)
        if log_path is None:
            print(
                f"Warning: missing log for aggregation variant '{variant}'. Trying cached outputs/results tree.",
                file=sys.stderr,
            )
            run = load_cached_run(repo_root, variant) or load_run_from_results_tree(repo_root, variant)
            if run is None:
                continue
            runs[variant] = run
            continue
        run = parse_completed_run(log_path, expected_variant=variant)
        if run is None:
            print(
                f"Warning: could not parse log {log_path}. Trying cached outputs/results tree.",
                file=sys.stderr,
            )
            run = load_cached_run(repo_root, variant) or load_run_from_results_tree(repo_root, variant)
            if run is None:
                continue
        runs[variant] = run

    runs["baseline"] = discover_baseline_log(repo_root, baseline_log)
    return runs


def make_output_dir(repo_root: Path) -> Path:
    output_dir = repo_root / "results" / "aggregation_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_summary_csv(summary_path: Path, runs: Dict[str, RunMetrics], sort_order: Sequence[str]) -> None:
    fieldnames = [
        "variant",
        "seed",
        "dataset",
        "source_log",
        "start_time",
        "end_time",
        "duration_seconds",
        "cil_avg_acc",
        "cil_bwt",
        "til_avg_acc",
        "til_bwt",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in sort_order:
            run = runs.get(variant)
            if run is None:
                continue
            writer.writerow(
                {
                    "variant": variant,
                    "seed": run.seed,
                    "dataset": run.dataset,
                    "source_log": str(run.source_log),
                    "start_time": format_datetime(run.start_time),
                    "end_time": format_datetime(run.end_time),
                    "duration_seconds": format_float(run.duration_seconds),
                    "cil_avg_acc": format_float(run.cil_final),
                    "cil_bwt": format_float(run.cil_bwt),
                    "til_avg_acc": format_float(run.til_final),
                    "til_bwt": format_float(run.til_bwt),
                }
            )


def write_per_experience_csv(per_experience_path: Path, runs: Dict[str, RunMetrics], sort_order: Sequence[str]) -> None:
    fieldnames = ["variant", "seed", "dataset", "evaluation", "experience", "avg_acc"]
    with per_experience_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in sort_order:
            run = runs.get(variant)
            if run is None:
                continue
            for evaluation, values in (("CIL", run.cil_by_experience), ("TIL", run.til_by_experience)):
                for experience, avg_acc in enumerate(values, start=1):
                    writer.writerow(
                        {
                            "variant": variant,
                            "seed": run.seed,
                            "dataset": run.dataset,
                            "evaluation": evaluation,
                            "experience": experience,
                            "avg_acc": format_float(avg_acc),
                        }
                    )


def format_float(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def format_datetime(value: Optional[datetime]) -> str:
    if value is None:
        return "unknown"
    return value.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def write_run_summary(summary_log_path: Path, runs: Dict[str, RunMetrics], sort_order: Sequence[str]) -> None:
    lines: List[str] = []
    for variant in sort_order:
        run = runs.get(variant)
        if run is None:
            continue
        lines.append(f"variant: {variant}")
        lines.append(f"seed: {run.seed}")
        lines.append(f"dataset: {run.dataset}")
        lines.append(f"source_log: {run.source_log}")
        lines.append(f"start_time: {format_datetime(run.start_time)}")
        lines.append(f"end_time: {format_datetime(run.end_time)}")
        lines.append(f"duration: {format_duration(run.duration_seconds)}")
        lines.append(f"cil_final_avg_acc: {format_float(run.cil_final)}")
        lines.append(f"cil_bwt: {format_float(run.cil_bwt)}")
        lines.append(f"til_final_avg_acc: {format_float(run.til_final)}")
        lines.append(f"til_bwt: {format_float(run.til_bwt)}")
        lines.append("")
    summary_log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def print_table(runs: Dict[str, RunMetrics], sort_order: Sequence[str]) -> None:
    headers = ["Variant", "Seed", "CIL Avg.ACC", "CIL BWT", "TIL Avg.ACC", "TIL BWT"]
    rows: List[List[str]] = []
    for variant in sort_order:
        run = runs.get(variant)
        if run is None:
            continue
        rows.append(
            [
                variant,
                str(run.seed),
                f"{run.cil_final:.2f}",
                "NA" if not math.isfinite(run.cil_bwt) else f"{run.cil_bwt:.2f}",
                f"{run.til_final:.2f}",
                "NA" if not math.isfinite(run.til_bwt) else f"{run.til_bwt:.2f}",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def render(values: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    print(render(headers))
    print(separator)
    for row in rows:
        print(render(row))


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 6.5,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "legend.fontsize": 5.8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sorted_variant_order(runs: Dict[str, RunMetrics]) -> List[str]:
    available_aggregations = [variant for variant in AGGREGATION_VARIANTS if variant in runs]
    available_aggregations.sort(
        key=lambda variant: (
            -runs[variant].cil_final,
            variant,
        )
    )
    if "baseline" in runs:
        available_aggregations.append("baseline")
    return available_aggregations


def sorted_variant_order_for_evaluation(runs: Dict[str, RunMetrics], evaluation: str) -> List[str]:
    metric = (lambda run: run.cil_final) if evaluation == "CIL" else (lambda run: run.til_final)
    ordered = [variant for variant in AGGREGATION_VARIANTS if variant in runs]
    ordered.sort(key=lambda variant: (-metric(runs[variant]), variant))
    if "baseline" in runs:
        ordered.append("baseline")
    return ordered


def experience_axis_and_values(run: RunMetrics, evaluation: str) -> Tuple[np.ndarray, np.ndarray]:
    values = run.cil_by_experience if evaluation == "CIL" else run.til_by_experience
    xs = np.arange(1, len(values) + 1, dtype=float)
    ys = np.asarray(values, dtype=float)
    return xs, ys


def set_experience_axis(ax: plt.Axes, max_experience: int, y_values: Iterable[float], ylabel: str) -> None:
    y_array = np.asarray(list(y_values), dtype=float)
    finite = y_array[np.isfinite(y_array)]
    if finite.size == 0:
        y_min, y_max = 0.0, 100.0
    else:
        y_min = max(0.0, math.floor(float(finite.min()) / 2.0) * 2.0 - 1.0)
        y_max = min(100.0, math.ceil(float(finite.max()) / 2.0) * 2.0 + 1.0)

    ax.set_xlim(1, max_experience)
    ax.set_xticks(np.arange(1, max_experience + 1))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Experience")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)


def plot_band_plot(output_path: Path, runs: Dict[str, RunMetrics]) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(NEURIPS_SINGLE_COL_WIDTH_IN, 1.75),
        sharex=True,
    )

    max_experience = max(len(run.cil_by_experience) for run in runs.values())
    legend_handles = [
        Line2D([0], [0], color=OKABE_ITO["black"], linestyle=(0, (4, 2)), linewidth=1.1, label="iCaRL (baseline)"),
        Line2D([0], [0], color=OKABE_ITO["blue"], linestyle="-", linewidth=1.4, label="SACK (max-mean)"),
        Line2D([0], [0], color=LIGHT_BLUE_BAND, linewidth=6, alpha=0.9, label="SACK (all variants range)"),
    ]

    for ax, evaluation in zip(axes, ("CIL", "TIL")):
        variant_matrix = []
        all_values: List[float] = []
        for variant in AGGREGATION_VARIANTS:
            if variant not in runs:
                continue
            _, ys = experience_axis_and_values(runs[variant], evaluation)
            padded = np.full(max_experience, np.nan, dtype=float)
            padded[: len(ys)] = ys
            variant_matrix.append(padded)
            all_values.extend(ys.tolist())

        matrix = np.vstack(variant_matrix) if variant_matrix else np.full((0, max_experience), np.nan, dtype=float)
        xs = np.arange(1, max_experience + 1, dtype=float)
        if matrix.size > 0:
            lower = np.nanmin(matrix, axis=0)
            upper = np.nanmax(matrix, axis=0)
            ax.fill_between(xs, lower, upper, color=LIGHT_BLUE_BAND, alpha=0.75, linewidth=0.0, zorder=1)

        baseline_x, baseline_y = experience_axis_and_values(runs["baseline"], evaluation)
        ax.plot(
            baseline_x,
            baseline_y,
            color=OKABE_ITO["black"],
            linewidth=1.1,
            linestyle=(0, (4, 2)),
            zorder=3,
        )

        if "max-mean" in runs:
            max_mean_x, max_mean_y = experience_axis_and_values(runs["max-mean"], evaluation)
            all_values.extend(max_mean_y.tolist())
            ax.plot(
                max_mean_x,
                max_mean_y,
                color=OKABE_ITO["blue"],
                linewidth=1.4,
                linestyle="-",
                zorder=4,
            )

        all_values.extend(baseline_y.tolist())
        set_experience_axis(ax, max_experience, all_values, "Avg.ACC (%)")
        ax.set_title(evaluation)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=1,
        frameon=False,
    )
    fig.subplots_adjust(left=0.12, right=0.995, top=0.88, bottom=0.37, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def distinct_best_worst_variants(runs: Dict[str, RunMetrics], evaluation: str) -> Tuple[str, str]:
    metric = (lambda run: run.cil_final) if evaluation == "CIL" else (lambda run: run.til_final)
    ranked = sorted(
        [variant for variant in AGGREGATION_VARIANTS if variant in runs],
        key=lambda variant: (-metric(runs[variant]), variant),
    )
    if not ranked:
        raise ValueError(f"No aggregation runs available for {evaluation}.")

    best_variant = ranked[0]
    worst_variant = sorted(
        [variant for variant in AGGREGATION_VARIANTS if variant in runs],
        key=lambda variant: (metric(runs[variant]), variant),
    )[0]
    return best_variant, worst_variant


def plot_top_variants(output_path: Path, runs: Dict[str, RunMetrics]) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(NEURIPS_SINGLE_COL_WIDTH_IN, 1.85),
        sharex=True,
    )
    max_experience = max(len(run.cil_by_experience) for run in runs.values())

    for ax, evaluation in zip(axes, ("CIL", "TIL")):
        best_variant, worst_variant = distinct_best_worst_variants(runs, evaluation)
        y_values: List[float] = []

        baseline_x, baseline_y = experience_axis_and_values(runs["baseline"], evaluation)
        y_values.extend(baseline_y.tolist())
        ax.plot(
            baseline_x,
            baseline_y,
            color=OKABE_ITO["black"],
            linewidth=1.1,
            linestyle=(0, (4, 2)),
            label=f"iCaRL (baseline, {baseline_y[-1]:.2f})",
            zorder=3,
        )

        max_mean_x, max_mean_y = experience_axis_and_values(runs["max-mean"], evaluation)
        y_values.extend(max_mean_y.tolist())
        ax.plot(
            max_mean_x,
            max_mean_y,
            color=OKABE_ITO["blue"],
            linewidth=1.4,
            label=f"max-mean ({max_mean_y[-1]:.2f})",
            zorder=4,
        )

        best_x, best_y = experience_axis_and_values(runs[best_variant], evaluation)
        y_values.extend(best_y.tolist())
        ax.plot(
            best_x,
            best_y,
            color=OKABE_ITO["green"],
            linewidth=1.3,
            label=f"{best_variant} ({best_y[-1]:.2f})",
            zorder=5,
        )

        worst_x, worst_y = experience_axis_and_values(runs[worst_variant], evaluation)
        y_values.extend(worst_y.tolist())
        ax.plot(
            worst_x,
            worst_y,
            color=OKABE_ITO["red"],
            linewidth=1.3,
            label=f"{worst_variant} ({worst_y[-1]:.2f})",
            zorder=4,
        )

        set_experience_axis(ax, max_experience, y_values, "Avg.ACC (%)")
        ax.set_title(evaluation)
        ax.legend(loc="lower left", frameon=False, handlelength=2.2)

    fig.subplots_adjust(left=0.12, right=0.995, top=0.88, bottom=0.2, wspace=0.32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_final_bar_plot(output_path: Path, runs: Dict[str, RunMetrics], evaluation: str) -> None:
    order = [
        variant
        for variant in sorted_variant_order_for_evaluation(runs, evaluation)
        if variant not in BAR_PLOT_EXCLUDED_VARIANTS
    ]
    metric = (lambda run: run.cil_final) if evaluation == "CIL" else (lambda run: run.til_final)

    labels = list(reversed(order))
    values = np.asarray([metric(runs[variant]) for variant in labels], dtype=float)
    y_positions = np.arange(len(labels))

    colors = []
    edgecolors = []
    for variant in labels:
        if variant == "baseline":
            colors.append("#BDBDBD")
            edgecolors.append(OKABE_ITO["black"])
        elif variant == "max-mean":
            colors.append(OKABE_ITO["blue"])
            edgecolors.append(OKABE_ITO["blue"])
        else:
            colors.append("#BFDFF4")
            edgecolors.append("#7AAFD6")

    with plt.rc_context(REF_PLOT_RCPARAMS):
        fig, ax = plt.subplots(figsize=(REF_PLOT_WIDTH_IN, REF_PLOT_HEIGHT_IN))
        bars = ax.barh(
            y_positions,
            values,
            color=colors,
            edgecolor=edgecolors,
            linewidth=1.0,
            height=0.62,
            zorder=3,
        )

        finite = values[np.isfinite(values)]
        if finite.size == 0:
            x_min, x_max = 0.0, 100.0
        else:
            x_min = max(0.0, math.floor(float(finite.min()) / 2.0) * 2.0 - 1.0)
            x_max = min(100.0, math.ceil(float(finite.max()) / 2.0) * 2.0 + 1.8)

        for bar, value, variant in zip(bars, values, labels):
            text = f"{value:.2f}"
            if variant == "max-mean":
                text = f"{text} *"
            ax.text(
                value + 0.22,
                bar.get_y() + bar.get_height() / 2.0,
                text,
                va="center",
                ha="left",
                color=OKABE_ITO["black"],
            )

        display_labels = []
        for variant in labels:
            if variant == "baseline":
                display_labels.append("baseline (iCaRL)")
            elif variant == "max-mean":
                display_labels.append("max-mean *")
            else:
                display_labels.append(variant)

        ax.set_yticks(y_positions, display_labels)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Final Avg.ACC (%)")
        ax.set_title(f"{evaluation} Final Avg.ACC")
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.45, zorder=0)
        ax.set_axisbelow(True)

        fig.subplots_adjust(left=0.31, right=0.97, top=0.9, bottom=0.2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf")
        plt.close(fig)


def build_heatmap_data(runs: Dict[str, RunMetrics], sort_order: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    row_labels = list(sort_order)
    values = np.full((len(row_labels), 4), np.nan, dtype=float)
    for row_idx, variant in enumerate(row_labels):
        run = runs.get(variant)
        if run is None:
            continue
        values[row_idx] = np.asarray(
            [run.cil_final, run.cil_bwt, run.til_final, run.til_bwt],
            dtype=float,
        )

    normalized = np.full_like(values, np.nan)
    for col_idx in range(values.shape[1]):
        column = values[:, col_idx]
        finite_mask = np.isfinite(column)
        if not finite_mask.any():
            continue
        finite_values = column[finite_mask]
        col_min = float(finite_values.min())
        col_max = float(finite_values.max())
        if math.isclose(col_min, col_max):
            normalized[finite_mask, col_idx] = 0.5
        else:
            normalized[finite_mask, col_idx] = (finite_values - col_min) / (col_max - col_min)

    return values, normalized, row_labels


def plot_heatmap(output_path: Path, runs: Dict[str, RunMetrics], sort_order: Sequence[str]) -> None:
    configure_matplotlib()
    values, normalized, row_labels = build_heatmap_data(runs, sort_order)
    fig_height = max(2.25, 0.29 * len(row_labels) + 0.85)
    fig, ax = plt.subplots(figsize=(NEURIPS_SINGLE_COL_WIDTH_IN, fig_height))

    image = ax.imshow(np.ma.masked_invalid(normalized), cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(4), ["CIL-Acc", "CIL-BWT", "TIL-Acc", "TIL-BWT"])
    display_rows = [f"{label} *" if label == "max-mean" else label for label in row_labels]
    ax.set_yticks(np.arange(len(row_labels)), display_rows)

    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    best_indices = {}
    for col_idx in range(values.shape[1]):
        column = values[:, col_idx]
        finite_mask = np.isfinite(column)
        if not finite_mask.any():
            continue
        best_indices[col_idx] = int(np.nanargmax(column))

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if not math.isfinite(value):
                ax.text(col_idx, row_idx, "NA", ha="center", va="center", color="black", fontsize=6)
                continue
            normed = normalized[row_idx, col_idx]
            text_color = "black" if not math.isfinite(normed) or normed < 0.75 else "white"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold" if best_indices.get(col_idx) == row_idx else "normal",
                color=text_color,
            )

    if "max-mean" in row_labels:
        max_mean_idx = row_labels.index("max-mean")
        ax.axhline(max_mean_idx + 0.5, color=OKABE_ITO["black"], linewidth=0.9, linestyle="--")

    if "baseline" in row_labels:
        baseline_idx = row_labels.index("baseline")
        ax.axhline(baseline_idx - 0.5, color=OKABE_ITO["gray"], linewidth=0.9)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("Column-wise normalized value")

    fig.subplots_adjust(left=0.26, right=0.95, top=0.97, bottom=0.11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def build_arg_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse SACK aggregation-ablation logs and generate summary artifacts.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root. Defaults to the parent of SACK_Scripts/.",
    )
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=None,
        help="Optional explicit iCaRL baseline log to use for the comparison plots.",
    )
    return parser


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    args = build_arg_parser(repo_root).parse_args()
    repo_root = args.repo_root.resolve()

    runs = load_runs(repo_root, args.baseline_log.resolve() if args.baseline_log else None)
    sort_order = sorted_variant_order(runs)
    output_dir = make_output_dir(repo_root)

    summary_csv = output_dir / "summary.csv"
    per_experience_csv = output_dir / "per_experience.csv"
    band_plot_pdf = output_dir / "band_plot.pdf"
    top_variants_pdf = output_dir / "top_variants.pdf"
    heatmap_pdf = output_dir / "heatmap.pdf"
    cil_bar_plot_pdf = output_dir / "bar_plot_cil.pdf"
    til_bar_plot_pdf = output_dir / "bar_plot_til.pdf"
    run_summary_log = output_dir / "run_summary.log"

    write_summary_csv(summary_csv, runs, sort_order)
    write_per_experience_csv(per_experience_csv, runs, sort_order)
    write_run_summary(run_summary_log, runs, sort_order)
    plot_band_plot(band_plot_pdf, runs)
    plot_top_variants(top_variants_pdf, runs)
    plot_heatmap(heatmap_pdf, runs, sort_order)
    plot_final_bar_plot(cil_bar_plot_pdf, runs, evaluation="CIL")
    plot_final_bar_plot(til_bar_plot_pdf, runs, evaluation="TIL")
    print_table(runs, sort_order)


if __name__ == "__main__":
    main()
