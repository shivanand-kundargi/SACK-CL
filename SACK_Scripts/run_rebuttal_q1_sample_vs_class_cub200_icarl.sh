#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[info] This Q1 launcher now defaults to CIFAR-100. Delegating to run_rebuttal_q1_sample_vs_class_cifar100_icarl.sh"
exec "${SCRIPT_DIR}/run_rebuttal_q1_sample_vs_class_cifar100_icarl.sh" "$@"
