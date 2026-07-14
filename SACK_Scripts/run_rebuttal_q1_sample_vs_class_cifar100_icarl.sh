#!/usr/bin/env bash
set -Eeuo pipefail

# Q1 rebuttal experiment:
#   Does sample-level concept weighting change the conclusion compared with
#   class-level concept weighting?
#
# Runs iCaRL on CIFAR-100 for:
#   1. baseline iCaRL
#   2. iCaRL + SACK with class-level weights
#   3. iCaRL + SACK with sample-level weights
#
# Example:
#   bash SACK_Scripts/run_rebuttal_q1_sample_vs_class_cifar100_icarl.sh
#
# Useful overrides:
#   SEEDS="0,1,2"
#   VARIANTS="baseline,class,sample"
#   DRY_RUN=1
#   NON_VERBOSE=1
#   NUM_WORKERS=4
#   SACK_TOPK=5
#   SACK_AGGREGATION=max-mean

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        PYTHON_BIN="python3"
    fi
fi
BASE_PATH="${BASE_PATH:-${ROOT_DIR}/data}"
RESULTS_PATH="${RESULTS_PATH:-results/rebuttal_q1_sample_vs_class_cifar100_icarl/${RUN_TAG}}"
RESULTS_ROOT="${BASE_PATH%/}/${RESULTS_PATH}"
LOG_DIR="${RESULTS_ROOT}/run_logs"
TABLE_DIR="${RESULTS_ROOT}/tables"
MANIFEST_PATH="${RESULTS_ROOT}/manifest.txt"
COMMANDS_PATH="${RESULTS_ROOT}/commands.sh"

SEEDS="${SEEDS:-0}"
VARIANTS="${VARIANTS:-baseline,class,sample}"
BUFFER_SIZE="${BUFFER_SIZE:-2000}"
SACK_SCHEDULE_VARIANT="${SACK_SCHEDULE_VARIANT:-w_to_u}"
SACK_AGGREGATION="${SACK_AGGREGATION:-max-mean}"
SACK_PERCENTILE="${SACK_PERCENTILE:-75}"
SACK_TOPK="${SACK_TOPK:-5}"
SACK_SAMPLE_BATCH_SIZE="${SACK_SAMPLE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NON_VERBOSE="${NON_VERBOSE:-0}"
DRY_RUN="${DRY_RUN:-0}"
ANALYZE_AFTER="${ANALYZE_AFTER:-1}"
SKIP_COMPLETED_RUNS="${SKIP_COMPLETED_RUNS:-1}"
NOTES="${NOTES:-rebuttal_q1_sample_vs_class_cifar100_icarl_${RUN_TAG}}"

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"

mkdir -p "${LOG_DIR}" "${TABLE_DIR}"

{
    echo "run_tag=${RUN_TAG}"
    echo "created_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "root_dir=${ROOT_DIR}"
    echo "base_path=${BASE_PATH}"
    echo "dataset=seq-cifar100"
    echo "results_path=${RESULTS_PATH}"
    echo "results_root=${RESULTS_ROOT}"
    echo "seeds=${SEEDS}"
    echo "variants=${VARIANTS}"
    echo "buffer_size=${BUFFER_SIZE}"
    echo "sack_schedule_variant=${SACK_SCHEDULE_VARIANT}"
    echo "sack_aggregation=${SACK_AGGREGATION}"
    echo "sack_percentile=${SACK_PERCENTILE}"
    echo "sack_topk=${SACK_TOPK}"
    echo "sack_sample_batch_size=${SACK_SAMPLE_BATCH_SIZE}"
    echo "num_workers=${NUM_WORKERS}"
    echo "notes=${NOTES}"
} > "${MANIFEST_PATH}"

{
    printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n\n'
} > "${COMMANDS_PATH}"

normalize_variant() {
    local value
    value="$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | xargs)"
    case "${value}" in
        baseline|original|og)
            echo "baseline"
            ;;
        class|class_level|class_level_sack|class-level)
            echo "class"
            ;;
        sample|sample_level|sample_level_sack|sample-level)
            echo "sample"
            ;;
        *)
            echo "[error] Unsupported variant: $1" >&2
            exit 1
            ;;
    esac
}

run_log_completed() {
    local log_file="$1"
    [[ -f "${log_file}" ]] || return 1
    grep -Fq "Logging results and arguments in " "${log_file}"
}

append_optional_wandb_flags() {
    local -n cmd_ref="$1"
    local run_name="$2"
    if [[ -n "${WANDB_PROJECT:-}" ]]; then
        cmd_ref+=(--wandb_project="${WANDB_PROJECT}" --wandb_name="${run_name}")
    fi
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        cmd_ref+=(--wandb_entity="${WANDB_ENTITY}")
    fi
}

append_optional_flags() {
    local -n cmd_ref="$1"
    if [[ -n "${SAVECHECK:-}" ]]; then
        cmd_ref+=(--savecheck="${SAVECHECK}" --ckpt_name="${RUN_NAME_FOR_FLAGS}")
    fi
    if [[ -n "${STOP_AFTER:-}" ]]; then
        cmd_ref+=(--stop_after="${STOP_AFTER}")
    fi
    if [[ -n "${DEVICE:-}" ]]; then
        cmd_ref+=(--device="${DEVICE}")
    fi
    if [[ -n "${EXTRA_ARGS:-}" ]]; then
        read -r -a user_extra_args <<< "${EXTRA_ARGS}"
        cmd_ref+=("${user_extra_args[@]}")
    fi
}

run_one() {
    local variant="$1"
    local seed="$2"
    local run_name="cifar100-icarl-${variant}-seed${seed}"
    local log_file="${LOG_DIR}/${run_name}.log"
    local -a cmd

    if [[ "${SKIP_COMPLETED_RUNS}" == "1" ]] && run_log_completed "${log_file}"; then
        echo "[skip] ${run_name} already completed according to ${log_file}"
        return 0
    fi

    cmd=(
        "${PYTHON_BIN}" -u main.py
        --base_path="${BASE_PATH}"
        --results_path="${RESULTS_PATH}"
        --dataset=seq-cifar100
        --model=icarl
        --buffer_size="${BUFFER_SIZE}"
        --model_config=best
        --sack_scores_type=0
        --enable_other_metrics=True
        --permute_classes=True
        --log_perf_metrics=1
        --num_workers="${NUM_WORKERS}"
        --non_verbose="${NON_VERBOSE}"
        --notes="${NOTES}"
        --seed="${seed}"
    )

    if [[ "${variant}" == "baseline" ]]; then
        cmd+=(--sack=0)
    else
        cmd+=(
            --sack=1
            --sack_schedule_variant="${SACK_SCHEDULE_VARIANT}"
            --sack_weight_granularity="${variant}"
            --sack_aggregation="${SACK_AGGREGATION}"
            --sack_similarity_percentile="${SACK_PERCENTILE}"
        )
        if [[ "${variant}" == "sample" ]]; then
            cmd+=(
                --sack_sample_topk_concepts="${SACK_TOPK}"
                --sack_sample_score_batch_size="${SACK_SAMPLE_BATCH_SIZE}"
            )
        fi
    fi

    append_optional_wandb_flags cmd "${run_name}"
    RUN_NAME_FOR_FLAGS="${run_name}"
    append_optional_flags cmd

    {
        printf '\n# %s\n' "${run_name}"
        printf ' '
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } >> "${COMMANDS_PATH}"

    echo
    echo "============================================================"
    echo "[run] ${run_name}"
    echo "[log] ${log_file}"
    printf '[cmd]'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    set +e
    "${cmd[@]}" 2>&1 | tee "${log_file}"
    local status="${PIPESTATUS[0]}"
    set -e
    if [[ "${status}" != "0" ]]; then
        echo "[error] ${run_name} failed with exit code ${status}. See ${log_file}" >&2
        exit "${status}"
    fi
}

echo "[info] Results root: ${RESULTS_ROOT}"
echo "[info] Logs: ${LOG_DIR}"
echo "[info] Manifest: ${MANIFEST_PATH}"
echo "[info] Commands: ${COMMANDS_PATH}"

for seed in "${SEED_LIST[@]}"; do
    seed="$(echo "${seed}" | xargs)"
    for raw_variant in "${VARIANT_LIST[@]}"; do
        variant="$(normalize_variant "${raw_variant}")"
        run_one "${variant}" "${seed}"
    done
done

if [[ "${ANALYZE_AFTER}" == "1" && "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" SACK_Scripts/parse_sample_weighting_results.py \
        --results-root "${RESULTS_ROOT}" \
        --out-dir "${TABLE_DIR}" \
        --table-name "q1_sample_vs_class"

    echo
    echo "============================================================"
    echo "[done] Q1 table:"
    cat "${TABLE_DIR}/q1_sample_vs_class_table.md"
fi

echo
echo "[done] Results root: ${RESULTS_ROOT}"
