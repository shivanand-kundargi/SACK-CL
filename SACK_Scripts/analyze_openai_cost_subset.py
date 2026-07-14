#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List


FULL_CLASS_COUNTS = {
    "cifar100": 100,
    "cub200": 200,
    "imagenet-r": 200,
}

DATASET_LABELS = {
    "cifar100": "CIFAR-100",
    "cub200": "CUB-200",
    "imagenet-r": "ImageNet-R",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrapolate full concept-generation cost from a small OpenAI API subset run."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--input-price-per-1m", type=float, required=True)
    parser.add_argument("--output-price-per-1m", type=float, required=True)
    return parser.parse_args()


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def as_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def format_float(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def read_records(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def summarize_dataset(records: List[Dict[str, str]], dataset: str, input_price: float, output_price: float) -> Dict[str, object]:
    rows = [
        row for row in records
        if row.get("dataset") == dataset and not row.get("error", "").strip()
    ]
    sampled = len(rows)
    full_classes = FULL_CLASS_COUNTS[dataset]
    scale = full_classes / sampled if sampled else math.nan

    latencies = [as_float(row.get("latency_s", "")) for row in rows]
    latencies = [value for value in latencies if math.isfinite(value)]
    concepts = [as_float(row.get("num_concepts", "")) for row in rows]
    concepts = [value for value in concepts if math.isfinite(value)]
    prompt_tokens = sum(as_int(row.get("prompt_tokens", "")) for row in rows)
    completion_tokens = sum(as_int(row.get("completion_tokens", "")) for row in rows)
    total_tokens = sum(as_int(row.get("total_tokens", "")) for row in rows)

    observed_cost = (
        prompt_tokens * input_price / 1_000_000.0
        + completion_tokens * output_price / 1_000_000.0
    )
    estimated_prompt_tokens = prompt_tokens * scale if sampled else math.nan
    estimated_completion_tokens = completion_tokens * scale if sampled else math.nan
    estimated_total_tokens = total_tokens * scale if sampled else math.nan
    estimated_cost = observed_cost * scale if sampled else math.nan
    observed_time_s = sum(latencies)
    estimated_time_s = observed_time_s * scale if sampled else math.nan

    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "sampled_classes": sampled,
        "full_classes": full_classes,
        "avg_concepts_per_class": sum(concepts) / len(concepts) if concepts else math.nan,
        "mean_latency_s": observed_time_s / len(latencies) if latencies else math.nan,
        "observed_time_s": observed_time_s,
        "estimated_full_time_s": estimated_time_s,
        "observed_prompt_tokens": prompt_tokens,
        "observed_completion_tokens": completion_tokens,
        "observed_total_tokens": total_tokens,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_completion_tokens": estimated_completion_tokens,
        "estimated_total_tokens": estimated_total_tokens,
        "observed_cost": observed_cost,
        "estimated_full_cost": estimated_cost,
    }


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    headers = [
        "dataset", "sampled_classes", "full_classes", "avg_concepts_per_class",
        "mean_latency_s", "observed_time_s", "estimated_full_time_s",
        "observed_prompt_tokens", "observed_completion_tokens", "observed_total_tokens",
        "estimated_prompt_tokens", "estimated_completion_tokens", "estimated_total_tokens",
        "observed_cost", "estimated_full_cost",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "dataset": row["dataset_label"],
                "sampled_classes": row["sampled_classes"],
                "full_classes": row["full_classes"],
                "avg_concepts_per_class": format_float(float(row["avg_concepts_per_class"]), 2),
                "mean_latency_s": format_float(float(row["mean_latency_s"]), 2),
                "observed_time_s": format_float(float(row["observed_time_s"]), 2),
                "estimated_full_time_s": format_float(float(row["estimated_full_time_s"]), 2),
                "observed_prompt_tokens": row["observed_prompt_tokens"],
                "observed_completion_tokens": row["observed_completion_tokens"],
                "observed_total_tokens": row["observed_total_tokens"],
                "estimated_prompt_tokens": format_float(float(row["estimated_prompt_tokens"]), 0),
                "estimated_completion_tokens": format_float(float(row["estimated_completion_tokens"]), 0),
                "estimated_total_tokens": format_float(float(row["estimated_total_tokens"]), 0),
                "observed_cost": format_float(float(row["observed_cost"]), 6),
                "estimated_full_cost": format_float(float(row["estimated_full_cost"]), 6),
            })


def write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    headers = [
        "Dataset", "Measured Classes", "Full Classes", "Avg Concepts/Class",
        "Mean Latency/Class (s)", "Observed Cost", "Estimated Full Cost",
        "Estimated Full Time (s)", "Estimated Full Tokens"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            str(row["dataset_label"]),
            str(row["sampled_classes"]),
            str(row["full_classes"]),
            format_float(float(row["avg_concepts_per_class"]), 2),
            format_float(float(row["mean_latency_s"]), 2),
            f"${float(row['observed_cost']):.6f}",
            f"${float(row['estimated_full_cost']):.6f}",
            format_float(float(row["estimated_full_time_s"]), 2),
            format_float(float(row["estimated_total_tokens"]), 0),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    per_class_path = args.results_dir / "concept_generation_per_class.csv"
    records = read_records(per_class_path)
    rows = [
        summarize_dataset(records, dataset, args.input_price_per_1m, args.output_price_per_1m)
        for dataset in FULL_CLASS_COUNTS
        if any(row.get("dataset") == dataset for row in records)
    ]
    write_csv(rows, args.results_dir / "openai_api_cost_extrapolated.csv")
    write_markdown(rows, args.results_dir / "openai_api_cost_extrapolated.md")
    print((args.results_dir / "openai_api_cost_extrapolated.md").read_text())


if __name__ == "__main__":
    main()
