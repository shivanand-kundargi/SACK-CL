#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
VALUE_RE = r"(?:np\.float64\()?[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\)?|None"
RE_ENTRY_START = re.compile(r"\{'dataset':")
RE_ACCMEAN = re.compile(rf"'accmean_task(\d+)':\s*({VALUE_RE})")
RE_ACCURACY = re.compile(rf"'accuracy_(\d+)_task(\d+)':\s*({VALUE_RE})")
RE_BACKWARD_TRANSFER = re.compile(rf"'backward_transfer':\s*({VALUE_RE})")
RE_RUNTIME_LINE = re.compile(
    rf"\[Runtime\]\s*Task\s+\d+:\s*(?:(?:total_time|task_time)=({FLOAT_RE})s,\s*)?"
    rf"train_time=({FLOAT_RE})s,\s*eval_time=({FLOAT_RE})s"
)
RE_RUN_LOG_NAME = re.compile(
    r"(?P<dataset_tag>cub200|cifar100)-(?P<model>.+)-(?P<granularity>class|sample|baseline)-seed(?P<seed>-?\d+)\.log$"
)
LEGACY_VARIANTS = {
    0: "w_to_u",
    1: "u_to_w",
    2: "u_to_random",
    3: "wbar_to_u",
    4: "u_to_wbar",
}
MODEL_DISPLAY = {
    "icarl": "iCaRL",
    "lwf": "LwF",
    "coda_prompt": "CODA-Prompt",
}


@dataclass(frozen=True)
class RunRow:
    setting: str
    dataset: str
    model: str
    seed: int
    granularity: str
    variant: str
    topk: Optional[int]
    aggregation: str
    final_task: int
    final_acc: float
    bwt: float
    runtime_s: float
    train_time_s: float
    eval_time_s: float
    source_log: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a rebuttal table for SACK class/sample weighting runs.")
    parser.add_argument("--results-root", type=Path, default=Path("data/results/sack_sample_weighting_cub200"),
                        help="Directory containing Mammoth logs.pyd files.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory. Defaults to <results-root>/tables.")
    parser.add_argument("--table-name", type=str, default="sample_weighting",
                        help="Prefix for generated CSV/Markdown files.")
    return parser.parse_args()


def split_entries(text: str) -> List[str]:
    starts = [match.start() for match in RE_ENTRY_START.finditer(text)]
    if not starts:
        return []
    entries = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        entries.append(text[start:end])
    return entries


def parse_float_token(token: str) -> float:
    token = token.strip()
    if token == "None":
        return math.nan
    if token.startswith("np.float64(") and token.endswith(")"):
        token = token[len("np.float64("):-1]
    return float(token)


def extract_string(entry: str, field: str) -> Optional[str]:
    matches = re.findall(rf"'{re.escape(field)}':\s*'([^']*)'", entry)
    return matches[-1] if matches else None


def extract_int(entry: str, field: str) -> Optional[int]:
    matches = re.findall(rf"'{re.escape(field)}':\s*(-?\d+)", entry)
    return int(matches[-1]) if matches else None


def extract_float(entry: str, field: str) -> float:
    matches = re.findall(rf"'{re.escape(field)}':\s*({VALUE_RE})", entry)
    if not matches:
        return math.nan
    return parse_float_token(matches[-1])


def extract_float_list(entry: str, field: str) -> List[float]:
    matches = re.findall(rf"'{re.escape(field)}':\s*\[([^\]]*)\]", entry)
    if not matches:
        return []
    values = []
    for token in re.findall(VALUE_RE, matches[-1]):
        value = parse_float_token(token)
        if math.isfinite(value):
            values.append(value)
    return values


def runtime_from_entry(entry: str) -> Tuple[float, float, float]:
    train_values = extract_float_list(entry, "runtime_train_time_s")
    eval_values = extract_float_list(entry, "runtime_eval_time_s")
    task_values = extract_float_list(entry, "runtime_task_time_s")
    train_time = sum(train_values) if train_values else extract_float(entry, "runtime_total_train_time_s")
    eval_time = sum(eval_values) if eval_values else extract_float(entry, "runtime_total_eval_time_s")
    runtime = extract_float(entry, "runtime_total_wall_time_s")
    if not math.isfinite(runtime) and task_values:
        runtime = sum(task_values)
    if not math.isfinite(runtime) and math.isfinite(train_time) and math.isfinite(eval_time):
        runtime = train_time + eval_time
    return runtime, train_time, eval_time


def setting_from_path(path: Path) -> str:
    for part in path.parts:
        if part in {"class-il", "task-il"}:
            return part
    return "unknown"


def fallback_dataset_model(path: Path) -> Tuple[str, str]:
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if part in {"class-il", "task-il"} and idx + 2 < len(parts):
            return parts[idx + 1], parts[idx + 2]
    return "unknown", "unknown"


def final_accmean(entry: str) -> Tuple[int, float]:
    values = [(int(task), parse_float_token(value)) for task, value in RE_ACCMEAN.findall(entry)]
    values = [(task, value) for task, value in values if math.isfinite(value)]
    if not values:
        return 0, math.nan
    task, value = max(values, key=lambda item: item[0])
    return task, value


def compute_bwt(entry: str, final_task: int) -> float:
    explicit = RE_BACKWARD_TRANSFER.findall(entry)
    if explicit:
        value = parse_float_token(explicit[-1])
        if math.isfinite(value):
            return value

    if final_task <= 1:
        return math.nan

    accuracies: Dict[Tuple[int, int], float] = {}
    for task_idx, eval_task, value in RE_ACCURACY.findall(entry):
        parsed = parse_float_token(value)
        if math.isfinite(parsed):
            accuracies[(int(task_idx), int(eval_task))] = parsed

    deltas = []
    for task_idx in range(1, final_task):
        first_seen = accuracies.get((task_idx, task_idx))
        final_seen = accuracies.get((task_idx, final_task))
        if first_seen is not None and final_seen is not None:
            deltas.append(final_seen - first_seen)
    return mean(deltas) if deltas else math.nan


def parse_entry(entry: str, source_log: Path) -> Optional[RunRow]:
    final_task, final_acc = final_accmean(entry)
    if final_task == 0 or not math.isfinite(final_acc):
        return None

    fallback_dataset, fallback_model = fallback_dataset_model(source_log)
    dataset = extract_string(entry, "dataset") or fallback_dataset
    model = extract_string(entry, "model") or fallback_model
    seed = extract_int(entry, "seed")
    if seed is None:
        seed = -1

    granularity = (
        extract_string(entry, "sack_effective_granularity")
        or extract_string(entry, "sack_weight_granularity")
        or "class"
    )
    if granularity == "none":
        granularity = "baseline"

    variant = extract_string(entry, "sack_effective_variant") or extract_string(entry, "sack_schedule_variant")
    if variant is None:
        variant = LEGACY_VARIANTS.get(extract_int(entry, "sack_scores_type") or 0, "unknown")

    topk = extract_int(entry, "sack_sample_topk_concepts")
    aggregation = extract_string(entry, "sack_aggregation") or "max-mean"
    bwt = compute_bwt(entry, final_task)
    runtime_s, train_time_s, eval_time_s = runtime_from_entry(entry)
    return RunRow(
        setting=setting_from_path(source_log),
        dataset=dataset,
        model=model,
        seed=seed,
        granularity=granularity,
        variant=variant,
        topk=topk,
        aggregation=aggregation,
        final_task=final_task,
        final_acc=final_acc,
        bwt=bwt,
        runtime_s=runtime_s,
        train_time_s=train_time_s,
        eval_time_s=eval_time_s,
        source_log=source_log,
    )


def parse_command_arg(text: str, arg_name: str) -> Optional[str]:
    matches = re.findall(rf"--{re.escape(arg_name)}(?:=|\s+)([^\s]+)", text)
    return matches[-1] if matches else None


def runtime_from_run_log(log_path: Path) -> Optional[Tuple[Tuple[str, str, int, str], Tuple[float, float, float]]]:
    match = RE_RUN_LOG_NAME.match(log_path.name)
    if not match:
        return None
    text = log_path.read_text(errors="replace")
    dataset_tag = match.group("dataset_tag")
    model = match.group("model")
    granularity = match.group("granularity")
    seed = int(match.group("seed"))
    fallback_dataset = "seq-cifar100" if dataset_tag == "cifar100" else "seq-cub200"
    dataset = parse_command_arg(text, "dataset") or fallback_dataset
    if model == "coda-prompt":
        model = "coda_prompt"

    train_total = 0.0
    eval_total = 0.0
    runtime_total = 0.0
    seen_runtime = False
    for runtime_match in RE_RUNTIME_LINE.finditer(text):
        task_total_raw, train_raw, eval_raw = runtime_match.groups()
        train_value = float(train_raw)
        eval_value = float(eval_raw)
        task_value = float(task_total_raw) if task_total_raw is not None else train_value + eval_value
        train_total += train_value
        eval_total += eval_value
        runtime_total += task_value
        seen_runtime = True
    if not seen_runtime:
        return None
    return (dataset, model, seed, granularity), (runtime_total, train_total, eval_total)


def read_runtime_logs(results_root: Path) -> Dict[Tuple[str, str, int, str], Tuple[float, float, float]]:
    runtimes: Dict[Tuple[str, str, int, str], Tuple[float, float, float]] = {}
    for log_path in sorted((results_root / "run_logs").glob("*.log")):
        parsed = runtime_from_run_log(log_path)
        if parsed is not None:
            key, value = parsed
            runtimes[key] = value
    return runtimes


def read_runs(results_root: Path) -> List[RunRow]:
    rows: List[RunRow] = []
    runtime_logs = read_runtime_logs(results_root)
    for log_path in sorted(results_root.rglob("logs.pyd")):
        text = log_path.read_text(errors="replace")
        for entry in split_entries(text):
            row = parse_entry(entry, log_path)
            if row is not None:
                runtime_key = (row.dataset, row.model, row.seed, row.granularity)
                if not math.isfinite(row.runtime_s) and runtime_key in runtime_logs:
                    runtime_s, train_time_s, eval_time_s = runtime_logs[runtime_key]
                    row = replace(row, runtime_s=runtime_s, train_time_s=train_time_s, eval_time_s=eval_time_s)
                rows.append(row)

    deduped: Dict[Tuple[str, str, str, int, str, str, Optional[int], str], RunRow] = {}
    for row in rows:
        key = (row.setting, row.dataset, row.model, row.seed, row.granularity, row.variant, row.topk, row.aggregation)
        deduped[key] = row
    return list(deduped.values())


def finite_values(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def format_metric(values: Iterable[float]) -> str:
    clean = finite_values(values)
    if not clean:
        return "-"
    if len(clean) == 1:
        return f"{clean[0]:.2f}"
    return f"{mean(clean):.2f} +/- {stdev(clean):.2f}"


def method_name(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def write_raw_csv(rows: List[RunRow], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "dataset", "setting", "model", "seed", "granularity", "variant", "topk",
            "aggregation", "final_task", "final_acc", "bwt", "runtime_s",
            "train_time_s", "eval_time_s", "source_log"
        ])
        for row in sorted(rows, key=lambda item: (item.dataset, item.model, item.granularity, item.setting, item.seed)):
            writer.writerow([
                row.dataset, row.setting, row.model, row.seed, row.granularity, row.variant,
                "" if row.topk is None else row.topk, row.aggregation, row.final_task,
                f"{row.final_acc:.6f}", "" if not math.isfinite(row.bwt) else f"{row.bwt:.6f}",
                "" if not math.isfinite(row.runtime_s) else f"{row.runtime_s:.6f}",
                "" if not math.isfinite(row.train_time_s) else f"{row.train_time_s:.6f}",
                "" if not math.isfinite(row.eval_time_s) else f"{row.eval_time_s:.6f}",
                row.source_log,
            ])


def aggregate_rows(rows: List[RunRow]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str, Optional[int], str], List[RunRow]] = {}
    for row in rows:
        key = (row.dataset, row.model, row.granularity, row.variant, row.topk, row.aggregation)
        grouped.setdefault(key, []).append(row)

    aggregate = []
    for key, group_rows in grouped.items():
        dataset, model, granularity, variant, topk, aggregation = key
        seeds = sorted({row.seed for row in group_rows})
        class_rows = [row for row in group_rows if row.setting == "class-il"]
        task_rows = [row for row in group_rows if row.setting == "task-il"]
        runtime_by_seed = {}
        train_time_by_seed = {}
        eval_time_by_seed = {}
        for row in group_rows:
            if math.isfinite(row.runtime_s):
                runtime_by_seed[row.seed] = row.runtime_s
            if math.isfinite(row.train_time_s):
                train_time_by_seed[row.seed] = row.train_time_s
            if math.isfinite(row.eval_time_s):
                eval_time_by_seed[row.seed] = row.eval_time_s
        aggregate.append({
            "dataset": dataset,
            "model": model,
            "granularity": granularity,
            "variant": variant,
            "topk": topk,
            "aggregation": aggregation,
            "seeds": seeds,
            "class_acc": finite_values(row.final_acc for row in class_rows),
            "class_bwt": finite_values(row.bwt for row in class_rows),
            "task_acc": finite_values(row.final_acc for row in task_rows),
            "task_bwt": finite_values(row.bwt for row in task_rows),
            "runtime": finite_values(runtime_by_seed.values()),
            "train_time": finite_values(train_time_by_seed.values()),
            "eval_time": finite_values(eval_time_by_seed.values()),
        })
    return sorted(aggregate, key=lambda item: (item["dataset"], item["model"], item["granularity"], item["variant"]))


def write_table_csv(aggregate: List[Dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "dataset", "method", "granularity", "variant", "topk", "aggregation", "seeds",
            "train_time_min", "eval_time_min", "total_runtime_min",
            "class_il_avg_acc", "class_il_bwt", "task_il_avg_acc", "task_il_bwt"
        ])
        for row in aggregate:
            writer.writerow([
                row["dataset"],
                method_name(str(row["model"])),
                row["granularity"],
                row["variant"],
                "-" if row["granularity"] == "class" else ("" if row["topk"] is None else row["topk"]),
                row["aggregation"],
                ",".join(str(seed) for seed in row["seeds"]),
                format_metric([value / 60.0 for value in row["train_time"]]),
                format_metric([value / 60.0 for value in row["eval_time"]]),
                format_metric([value / 60.0 for value in row["runtime"]]),
                format_metric(row["class_acc"]),
                format_metric(row["class_bwt"]),
                format_metric(row["task_acc"]),
                format_metric(row["task_bwt"]),
            ])


def write_markdown_table(aggregate: List[Dict[str, object]], path: Path) -> None:
    headers = [
        "Dataset", "Method", "Granularity", "Variant", "Top-k", "Aggregation", "Seeds",
        "Train Time (min)", "Eval Time (min)", "Total Runtime (min)",
        "Class-IL Avg. Acc", "Class-IL BWT", "Task-IL Avg. Acc", "Task-IL BWT"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in aggregate:
        values = [
            str(row["dataset"]),
            method_name(str(row["model"])),
            str(row["granularity"]),
            str(row["variant"]),
            "-" if row["granularity"] == "class" else ("" if row["topk"] is None else str(row["topk"])),
            str(row["aggregation"]),
            ",".join(str(seed) for seed in row["seeds"]),
            format_metric([value / 60.0 for value in row["train_time"]]),
            format_metric([value / 60.0 for value in row["eval_time"]]),
            format_metric([value / 60.0 for value in row["runtime"]]),
            format_metric(row["class_acc"]),
            format_metric(row["class_bwt"]),
            format_metric(row["task_acc"]),
            format_metric(row["task_bwt"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    out_dir = (args.out_dir or (results_root / "tables")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_runs(results_root)
    if not rows:
        raise SystemExit(f"No completed logs.pyd entries found under {results_root}")

    aggregate = aggregate_rows(rows)
    raw_csv = out_dir / f"{args.table_name}_raw.csv"
    table_csv = out_dir / f"{args.table_name}_table.csv"
    table_md = out_dir / f"{args.table_name}_table.md"
    write_raw_csv(rows, raw_csv)
    write_table_csv(aggregate, table_csv)
    write_markdown_table(aggregate, table_md)

    print(f"Wrote raw runs: {raw_csv}")
    print(f"Wrote table CSV: {table_csv}")
    print(f"Wrote markdown table: {table_md}")


if __name__ == "__main__":
    main()
