#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED=0

VARIANTS=(
  "max-mean"
  "mean-mean"
  "min-mean"
  "top3-mean"
  "top5-mean"
  "softmax-sharp"
  "softmax-smooth"
  "max-max"
)

RESULTS_ROOT="${ROOT_DIR}/results/aggregation_ablation/coda-prompt/"
LOG_DIR="${RESULTS_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

COMMON_FLAGS=(
  "--dataset=seq-cifar100-224"
  "--model=coda-prompt"
  "--model_config=best"
  "--sack=1"
  "--sack_scores_type=0"
  "--enable_other_metrics=True"
  "--savecheck=task"
  "--permute_classes=True"
  "--log_perf_metrics=1"
  "--seed=${SEED}"
)

for variant in "${VARIANTS[@]}"; do
  cil_log="${LOG_DIR}/${variant}_cil.log"
  til_log="${LOG_DIR}/${variant}_til.log"
  run_results_path="results/aggregation_ablation/runs/${variant}"
  ckpt_name="icarl-seq-cifar100-sack-aggregation-${variant}-seed-${SEED}"

  cmd=(
    "${PYTHON_BIN}" "-u" "main.py"
    "${COMMON_FLAGS[@]}"
    "--sack_aggregation=${variant}"
    "--results_path=${run_results_path}"
    "--ckpt_name=${ckpt_name}"
  )

  echo "============================================================"
  echo "Running iCaRL + SACK aggregation ablation: ${variant}"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  echo "Logs: ${cil_log} and ${til_log}"
  echo

  # In this repo, a single class-incremental run already prints both Class-IL and Task-IL metrics.
  # We mirror the same stdout/stderr into both requested log files.
  set +e
  "${cmd[@]}" 2>&1 | tee "${cil_log}" | tee "${til_log}"
  cmd_status=${PIPESTATUS[0]}
  set -e

  if [[ ${cmd_status} -ne 0 ]]; then
    echo "Run failed for variant ${variant} with exit code ${cmd_status}" >&2
    exit "${cmd_status}"
  fi
done
