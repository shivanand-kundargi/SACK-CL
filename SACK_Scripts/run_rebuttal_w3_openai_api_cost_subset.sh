#!/usr/bin/env bash
set -Eeuo pipefail

# W3 rebuttal experiment, API-only:
#   Generate concepts from CIFAR-100, CUB-200, and ImageNet-R using the
#   OpenAI API, then report observed and extrapolated all-class preprocessing cost.
#
# Required:
#   export OPENAI_API_KEY=...
#
# Defaults use gpt-4.1-mini pricing from the OpenAI API pricing page:
#   input  = $0.40 / 1M tokens
#   output = $1.60 / 1M tokens
#
# Run all classes:
#   LIMIT_CLASSES=all bash SACK_Scripts/run_rebuttal_w3_openai_api_cost_subset.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BASE_PATH="${BASE_PATH:-${ROOT_DIR}/data}"
RESULTS_PATH="${RESULTS_PATH:-results/rebuttal_w3_openai_api_cost_subset/${RUN_TAG}}"
RESULTS_ROOT="${BASE_PATH%/}/${RESULTS_PATH}"
LOG_FILE="${RESULTS_ROOT}/openai_api_concept_generation.log"
COMMANDS_PATH="${RESULTS_ROOT}/command.sh"

DATASETS="${DATASETS:-cifar100,cub200,imagenet-r}"
LIMIT_CLASSES="${LIMIT_CLASSES:-5}"
LIMIT_CLASSES_NORMALIZED="$(printf '%s' "${LIMIT_CLASSES}" | tr '[:upper:]' '[:lower:]')"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4.1-mini}"
OPENAI_INPUT_PRICE_PER_1M="${OPENAI_INPUT_PRICE_PER_1M:-0.40}"
OPENAI_OUTPUT_PRICE_PER_1M="${OPENAI_OUTPUT_PRICE_PER_1M:-1.60}"
SACK_TEMPERATURE="${SACK_TEMPERATURE:-0.2}"
SACK_MAX_TOKENS="${SACK_MAX_TOKENS:-220}"
SACK_TIMEOUT="${SACK_TIMEOUT:-120}"
SACK_SLEEP_SECS="${SACK_SLEEP_SECS:-0}"
SACK_RETRIES="${SACK_RETRIES:-2}"
RESUME="${RESUME:-1}"
SKIP_ENDPOINT_CHECK="${SKIP_ENDPOINT_CHECK:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DRY_RUN}" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[error] OPENAI_API_KEY is required." >&2
    echo "        Run: export OPENAI_API_KEY=..." >&2
    exit 1
fi

mkdir -p "${RESULTS_ROOT}"

cmd=(
    "${PYTHON_BIN}" -u SACK_Scripts/generate_concept_costs.py
    --datasets="${DATASETS}"
    --out-dir="${RESULTS_ROOT}"
    --backend=openai-compatible
    --base-url="${OPENAI_BASE_URL}"
    --model="${OPENAI_MODEL}"
    --temperature="${SACK_TEMPERATURE}"
    --max-tokens="${SACK_MAX_TOKENS}"
    --timeout="${SACK_TIMEOUT}"
    --sleep-secs="${SACK_SLEEP_SECS}"
    --retries="${SACK_RETRIES}"
    --input-price-per-1m="${OPENAI_INPUT_PRICE_PER_1M}"
    --output-price-per-1m="${OPENAI_OUTPUT_PRICE_PER_1M}"
)

if [[ "${LIMIT_CLASSES_NORMALIZED}" != "all" && "${LIMIT_CLASSES_NORMALIZED}" != "full" && "${LIMIT_CLASSES_NORMALIZED}" != "none" && "${LIMIT_CLASSES_NORMALIZED}" != "0" ]]; then
    cmd+=(--limit-classes="${LIMIT_CLASSES}")
fi

if [[ "${RESUME}" == "1" ]]; then
    cmd+=(--resume)
fi
if [[ "${SKIP_ENDPOINT_CHECK}" == "1" ]]; then
    cmd+=(--skip-endpoint-check)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run --skip-endpoint-check)
fi

analyze_cmd=(
    "${PYTHON_BIN}" -u SACK_Scripts/analyze_openai_cost_subset.py
    --results-dir="${RESULTS_ROOT}"
    --input-price-per-1m="${OPENAI_INPUT_PRICE_PER_1M}"
    --output-price-per-1m="${OPENAI_OUTPUT_PRICE_PER_1M}"
)

{
    printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n\n'
    printf 'cd %q\n' "${ROOT_DIR}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        printf 'export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}"\n'
    fi
    printf ' '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    printf ' '
    printf '%q ' "${analyze_cmd[@]}"
    printf '\n'
} > "${COMMANDS_PATH}"
chmod +x "${COMMANDS_PATH}"

echo "[info] Results root: ${RESULTS_ROOT}"
echo "[info] Log file: ${LOG_FILE}"
echo "[info] Command file: ${COMMANDS_PATH}"
echo "[info] Backend: OpenAI API"
echo "[info] Base URL: ${OPENAI_BASE_URL}"
echo "[info] Model: ${OPENAI_MODEL}"
echo "[info] Datasets: ${DATASETS}"
if [[ "${LIMIT_CLASSES_NORMALIZED}" == "all" || "${LIMIT_CLASSES_NORMALIZED}" == "full" || "${LIMIT_CLASSES_NORMALIZED}" == "none" || "${LIMIT_CLASSES_NORMALIZED}" == "0" ]]; then
    echo "[info] Classes per dataset: all"
else
    echo "[info] Classes per dataset: ${LIMIT_CLASSES}"
fi
echo "[info] Prices per 1M tokens: input=\$${OPENAI_INPUT_PRICE_PER_1M}, output=\$${OPENAI_OUTPUT_PRICE_PER_1M}"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[info] DRY_RUN=1, no API calls will be made."
fi

set +e
OPENAI_API_KEY="${OPENAI_API_KEY:-}" "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
status="${PIPESTATUS[0]}"
set -e

if [[ "${status}" != "0" ]]; then
    echo "[error] OpenAI API concept generation failed with exit code ${status}. See ${LOG_FILE}" >&2
    exit "${status}"
fi

echo
echo "[info] Writing full-dataset cost table..."
"${analyze_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"

echo
echo "[done] Summary: ${RESULTS_ROOT}/concept_generation_summary.md"
echo "[done] Overall timing: ${RESULTS_ROOT}/concept_generation_overall.md"
echo "[done] Full cost table: ${RESULTS_ROOT}/openai_api_cost_extrapolated.md"
echo "[done] Per-class CSV: ${RESULTS_ROOT}/concept_generation_per_class.csv"
