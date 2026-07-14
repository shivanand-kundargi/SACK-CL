#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
DATASET="${DATASET:-seq-cub200}"
METHODS="${METHODS:-icarl,lwf,coda_prompt}"
SEEDS="${SEEDS:-0}"
GRANULARITIES="${GRANULARITIES:-sample}"
RESULTS_PATH="${RESULTS_PATH:-results/sack_sample_weighting_cub200/${RUN_TAG}}"
RESULTS_ROOT="${ROOT_DIR}/data/${RESULTS_PATH}"
LOG_DIR="${RESULTS_ROOT}/run_logs"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    elif [[ -x "/usr/workspace/kundargi1/anaconda3/envs/clip-dissect/bin/python" ]]; then
        PYTHON_BIN="/usr/workspace/kundargi1/anaconda3/envs/clip-dissect/bin/python"
    else
        PYTHON_BIN="python3"
    fi
fi
SACK_SCHEDULE_VARIANT="${SACK_SCHEDULE_VARIANT:-w_to_u}"
SACK_AGGREGATION="${SACK_AGGREGATION:-max-mean}"
SACK_PERCENTILE="${SACK_PERCENTILE:-75}"
SACK_TOPK="${SACK_TOPK:-5}"
SACK_SAMPLE_BATCH_SIZE="${SACK_SAMPLE_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NON_VERBOSE="${NON_VERBOSE:-0}"
DRY_RUN="${DRY_RUN:-0}"
ANALYZE_AFTER="${ANALYZE_AFTER:-1}"
PREPARE_CUB200="${PREPARE_CUB200:-1}"

mkdir -p "${LOG_DIR}"

IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a GRANULARITY_LIST <<< "${GRANULARITIES}"

if [[ "${DATASET}" == "seq-cub200" && "${PREPARE_CUB200}" == "1" && "${DRY_RUN}" != "1" ]]; then
    if [[ ! -f "${ROOT_DIR}/data/CUB200/train_data.npz" || ! -f "${ROOT_DIR}/data/CUB200/test_data.npz" ]]; then
        "${PYTHON_BIN}" scripts/prepare_cub200_npz.py --root "${ROOT_DIR}/data/CUB200"
    fi
fi

method_args() {
    local method="$1"
    case "${method}" in
        icarl)
            printf '%s\n' "--model=icarl" "--buffer_size=2000" "--model_config=best"
            ;;
        lwf)
            printf '%s\n' "--model=lwf" "--lr=0.03" "--backbone=resnet50"
            ;;
        coda_prompt|coda-prompt)
            printf '%s\n' "--model=coda_prompt" "--model_config=best"
            ;;
        *)
            printf 'Unknown method: %s\n' "${method}" >&2
            return 1
            ;;
    esac
}

run_one() {
    local method="$1"
    local seed="$2"
    local granularity="$3"
    local model_name="${method//-/_}"
    local run_name="cub200-${model_name}-${granularity}-seed${seed}"
    local log_file="${LOG_DIR}/${run_name}.log"
    local -a cmd
    local -a extra_method_args

    mapfile -t extra_method_args < <(method_args "${method}")
    cmd=(
        "${PYTHON_BIN}" main.py
        --dataset="${DATASET}"
        --sack=1
        --sack_scores_type=0
        --sack_schedule_variant="${SACK_SCHEDULE_VARIANT}"
        --sack_weight_granularity="${granularity}"
        --sack_sample_topk_concepts="${SACK_TOPK}"
        --sack_sample_score_batch_size="${SACK_SAMPLE_BATCH_SIZE}"
        --sack_aggregation="${SACK_AGGREGATION}"
        --sack_similarity_percentile="${SACK_PERCENTILE}"
        --enable_other_metrics=True
        --permute_classes=True
        --log_perf_metrics=1
        --num_workers="${NUM_WORKERS}"
        --non_verbose="${NON_VERBOSE}"
        --results_path="${RESULTS_PATH}"
        --notes="sack-${granularity}-weighting-top${SACK_TOPK}"
        --seed="${seed}"
    )
    cmd+=("${extra_method_args[@]}")

    if [[ -n "${SAVECHECK:-}" ]]; then
        cmd+=(--savecheck="${SAVECHECK}" --ckpt_name="${run_name}")
    fi
    if [[ -n "${WANDB_PROJECT:-}" ]]; then
        cmd+=(--wandb_project="${WANDB_PROJECT}" --wandb_name="${run_name}")
    fi
    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        cmd+=(--wandb_entity="${WANDB_ENTITY}")
    fi
    if [[ -n "${EXTRA_ARGS:-}" ]]; then
        read -r -a user_extra_args <<< "${EXTRA_ARGS}"
        cmd+=("${user_extra_args[@]}")
    fi

    printf '\n=== %s ===\n' "${run_name}"
    printf 'Results path: %s\n' "${RESULTS_PATH}"
    printf 'Log file: %s\n' "${log_file}"
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    "${cmd[@]}" 2>&1 | tee "${log_file}"
}

for seed in "${SEED_LIST[@]}"; do
    for granularity in "${GRANULARITY_LIST[@]}"; do
        for method in "${METHOD_LIST[@]}"; do
            run_one "${method}" "${seed}" "${granularity}"
        done
    done
done

if [[ "${ANALYZE_AFTER}" == "1" && "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" SACK_Scripts/parse_sample_weighting_results.py \
        --results-root "${RESULTS_ROOT}" \
        --out-dir "${RESULTS_ROOT}/tables"
fi

printf '\nDone. Results root: %s\n' "${RESULTS_ROOT}"
