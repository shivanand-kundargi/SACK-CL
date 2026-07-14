#!/usr/bin/env bash
set -Eeuo pipefail

# W3 rebuttal experiment:
#   Measure one-time concept-generation preprocessing cost for SACK.
#
# This script loads a local/Hugging Face model directly with Transformers
# and regenerates concept banks for:
#   CIFAR-100, CUB-200, and ImageNet-R.
#
# Typical environment variables:
#   SACK_LLM_MODEL=openai/gpt-oss-120b
#
# By default, missing Hugging Face model weights are downloaded/cached under:
#   ${BASE_PATH}/hf_cache
#
# If you still want to use a vLLM/OpenAI-compatible server:
#   SACK_LLM_BACKEND=openai-compatible
#   SACK_LLM_BASE_URL=http://localhost:8000/v1
#
# Optional cost reporting:
#   SACK_INPUT_PRICE_PER_1M=0
#   SACK_OUTPUT_PRICE_PER_1M=0
#
# Debug without endpoint calls:
#   DRY_RUN=1 LIMIT_CLASSES=3 bash SACK_Scripts/run_rebuttal_w3_concept_generation_cost.sh
#
# Install/check GPT-OSS Python dependencies without loading the 120B model:
#   CHECK_DEPS_ONLY=1 bash SACK_Scripts/run_rebuttal_w3_concept_generation_cost.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
USER_PYTHON_BIN="${PYTHON_BIN:-}"
PYTHON_BIN="${USER_PYTHON_BIN}"

BASE_PATH="${BASE_PATH:-${ROOT_DIR}/data}"
RESULTS_PATH="${RESULTS_PATH:-results/rebuttal_w3_concept_generation_cost/${RUN_TAG}}"
RESULTS_ROOT="${BASE_PATH%/}/${RESULTS_PATH}"
LOG_FILE="${RESULTS_ROOT}/concept_generation.log"
COMMANDS_PATH="${RESULTS_ROOT}/command.sh"

DATASETS="${DATASETS:-cifar100,cub200,imagenet-r}"
SACK_LLM_BACKEND="${SACK_LLM_BACKEND:-transformers}"
SACK_LLM_BASE_URL="${SACK_LLM_BASE_URL:-${OPENAI_BASE_URL:-}}"
SACK_LLM_MODEL="${SACK_LLM_MODEL:-openai/gpt-oss-120b}"
SACK_LLM_API_KEY="${SACK_LLM_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
export SACK_LLM_API_KEY
SACK_TEMPERATURE="${SACK_TEMPERATURE:-0.2}"
SACK_MAX_TOKENS="${SACK_MAX_TOKENS:-220}"
SACK_TIMEOUT="${SACK_TIMEOUT:-120}"
SACK_SLEEP_SECS="${SACK_SLEEP_SECS:-0}"
SACK_RETRIES="${SACK_RETRIES:-2}"
SACK_TORCH_DTYPE="${SACK_TORCH_DTYPE:-auto}"
SACK_DEVICE_MAP="${SACK_DEVICE_MAP:-auto}"
SACK_MODEL_CACHE_DIR="${SACK_MODEL_CACHE_DIR:-${BASE_PATH%/}/hf_cache}"
SACK_PYDEPS_DIR="${SACK_PYDEPS_DIR:-${BASE_PATH%/}/w3_pydeps_gptoss_v3}"
SACK_AUTO_INSTALL_DEPS="${SACK_AUTO_INSTALL_DEPS:-1}"
SACK_AUTO_DOWNLOAD_MODEL="${SACK_AUTO_DOWNLOAD_MODEL:-1}"
SACK_LOCAL_FILES_ONLY="${SACK_LOCAL_FILES_ONLY:-0}"
SACK_TRUST_REMOTE_CODE="${SACK_TRUST_REMOTE_CODE:-1}"
SACK_INPUT_PRICE_PER_1M="${SACK_INPUT_PRICE_PER_1M:-0}"
SACK_OUTPUT_PRICE_PER_1M="${SACK_OUTPUT_PRICE_PER_1M:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_ENDPOINT_CHECK="${SKIP_ENDPOINT_CHECK:-0}"
LIMIT_CLASSES="${LIMIT_CLASSES:-}"
CHECK_DEPS_ONLY="${CHECK_DEPS_ONLY:-0}"

mkdir -p "${RESULTS_ROOT}"

resolve_python_candidate() {
    local candidate="$1"
    if [[ "${candidate}" == */* ]]; then
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
        fi
    else
        command -v "${candidate}" 2>/dev/null || true
    fi
}

probe_python_runtime() {
    local python_bin="$1"
    "${python_bin}" - <<'PY'
import sys

print(f"python={sys.version.split()[0]}", end="")
if sys.version_info < (3, 9):
    print(" status=bad reason=python_lt_3.9")
    raise SystemExit(1)

try:
    import torch
except Exception as exc:
    print(f" status=bad reason=torch_import_failed detail={exc}")
    raise SystemExit(1)

torch_version = getattr(torch, "__version__", "unknown")
has_xpu = hasattr(torch, "xpu")
print(f" torch={torch_version} torch_xpu={int(has_xpu)}", end="")
if not has_xpu:
    print(" status=bad reason=torch_missing_xpu")
    raise SystemExit(1)

print(" status=ok")
PY
}

select_python_bin() {
    local candidate resolved probe_status probe_code seen_key
    local -a candidates=()
    local -A seen=()

    if [[ -n "${USER_PYTHON_BIN}" ]]; then
        resolved="$(resolve_python_candidate "${USER_PYTHON_BIN}")"
        if [[ -z "${resolved}" ]]; then
            echo "[error] PYTHON_BIN was set to '${USER_PYTHON_BIN}', but it was not found." >&2
            exit 1
        fi
        set +e
        probe_status="$(probe_python_runtime "${resolved}" 2>&1)"
        probe_code="$?"
        set -e
        if [[ "${probe_code}" != "0" ]]; then
            echo "[error] PYTHON_BIN=${resolved} is not compatible with GPT-OSS Transformers." >&2
            echo "${probe_status}" >&2
            echo "[error] Unset PYTHON_BIN to let this script auto-select a compatible Python." >&2
            exit 1
        fi
        PYTHON_BIN="${resolved}"
        echo "[info] Using user-provided Python: ${PYTHON_BIN} (${probe_status})"
        return
    fi

    candidates+=(
        "python3"
        "/usr/tce/bin/python3"
        "/usr/bin/python3.13"
        "/usr/bin/python3.12"
        "/usr/bin/python3.11"
        "/usr/bin/python3.10"
        "/usr/bin/python3.9"
        "python3.13"
        "python3.12"
        "python3.11"
        "python3.10"
        "python3.9"
        "python"
    )

    local diagnostics=""
    for candidate in "${candidates[@]}"; do
        resolved="$(resolve_python_candidate "${candidate}")"
        if [[ -z "${resolved}" ]]; then
            continue
        fi
        seen_key="${resolved}"
        if [[ -n "${seen[${seen_key}]:-}" ]]; then
            continue
        fi
        seen["${seen_key}"]=1

        set +e
        probe_status="$(probe_python_runtime "${resolved}" 2>&1)"
        probe_code="$?"
        set -e
        diagnostics+=$'\n'"  - ${resolved}: ${probe_status}"
        if [[ "${probe_code}" == "0" ]]; then
            PYTHON_BIN="${resolved}"
            echo "[info] Auto-selected Python: ${PYTHON_BIN} (${probe_status})"
            return
        fi
    done

    echo "[error] Could not find a Python runtime compatible with GPT-OSS." >&2
    echo "[error] Need python>=3.9 plus a PyTorch build exposing torch.xpu; torch 2.2.x is too old." >&2
    echo "[error] Probed candidates:${diagnostics}" >&2
    exit 1
}

select_basic_python_bin() {
    if [[ -n "${USER_PYTHON_BIN}" ]]; then
        PYTHON_BIN="${USER_PYTHON_BIN}"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "[error] Could not find python3 or python on PATH." >&2
        exit 1
    fi
}

check_transformers_deps() {
    "${PYTHON_BIN}" - <<'PY'
import re
import sys
from importlib import metadata

def version_tuple(value):
    parts = [int(part) for part in re.findall(r"\d+", str(value))[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)

issues = []
if sys.version_info < (3, 9):
    issues.append(f"python>={3.9} required for the current GPT-OSS Transformers stack; found {sys.version.split()[0]}")

try:
    import torch
except Exception as exc:
    issues.append(f"torch import failed: {exc}")
else:
    if not hasattr(torch, "xpu"):
        issues.append(f"torch with torch.xpu attribute required; found torch {getattr(torch, '__version__', 'unknown')}")

for package, minimum in (
    ("transformers", "4.55.0"),
    ("tokenizers", "0.22.0"),
    ("accelerate", "0.26.0"),
    ("kernels", "0.14.1"),
    ("triton", "3.4.0"),
):
    try:
        found = metadata.version(package)
    except metadata.PackageNotFoundError:
        issues.append(f"{package}>={minimum} missing")
        continue
    if version_tuple(found) < version_tuple(minimum):
        issues.append(f"{package}>={minimum} required; found {found}")

if issues:
    print("\n".join(issues))
    raise SystemExit(1)
print("ok")
PY
}

patch_local_kernels_package() {
    "${PYTHON_BIN}" - "${SACK_PYDEPS_DIR}" <<'PY'
from pathlib import Path
import sys

deps_path = Path(sys.argv[1]) / "kernels" / "deps.py"
if not deps_path.exists():
    raise SystemExit(0)

text = deps_path.read_text()
if "import_name: str | None" not in text:
    raise SystemExit(0)

if "from typing import Optional" not in text:
    text = text.replace("from pathlib import Path\n", "from pathlib import Path\nfrom typing import Optional\n")
text = text.replace("import_name: str | None", "import_name: Optional[str]")
deps_path.write_text(text)
print(f"[info] Patched local kernels dependency metadata: {deps_path}")
PY
}

if [[ "${SACK_LLM_BACKEND}" == "transformers" && "${DRY_RUN}" != "1" ]]; then
    select_python_bin
else
    select_basic_python_bin
fi

if [[ "${SACK_LLM_BACKEND}" == "transformers" && "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${SACK_PYDEPS_DIR}"
    export PYTHONPATH="${SACK_PYDEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

    set +e
    dep_status="$(check_transformers_deps 2>&1)"
    dep_code="$?"
    set -e
    if [[ "${dep_code}" != "0" ]]; then
        echo "[warn] GPT-OSS dependency check failed:"
        echo "${dep_status}"
        if [[ "${SACK_AUTO_INSTALL_DEPS}" != "1" ]]; then
            echo "[error] Set SACK_AUTO_INSTALL_DEPS=1 or install: transformers>=4.55.0 tokenizers>=0.22.0 accelerate kernels" >&2
            exit 1
        fi
        echo "[info] Installing GPT-OSS Python dependencies into ${SACK_PYDEPS_DIR}"
        set +e
        "${PYTHON_BIN}" -m pip install --upgrade --no-deps --target "${SACK_PYDEPS_DIR}" \
            "transformers>=4.55.0,<5" \
            "tokenizers>=0.22.0,<0.23" \
            "accelerate>=0.26.0,<2" \
            "huggingface-hub>=0.34.0,<1.0" \
            "kernels==0.14.1" \
            "kernels-data==0.14.1" \
            "triton>=3.4.0,<4" \
            "tomlkit>=0.13.3" 2>&1 | tee -a "${LOG_FILE}"
        pip_status="${PIPESTATUS[0]}"
        set -e
        if [[ "${pip_status}" != "0" ]]; then
            echo "[error] Dependency installation failed with exit code ${pip_status}. See ${LOG_FILE}" >&2
            exit "${pip_status}"
        fi
        patch_local_kernels_package

        set +e
        dep_status="$(check_transformers_deps 2>&1)"
        dep_code="$?"
        set -e
        if [[ "${dep_code}" != "0" ]]; then
            echo "[error] Dependency installation finished, but the GPT-OSS check still failed:" >&2
            echo "${dep_status}" >&2
            exit 1
        fi
    fi
    patch_local_kernels_package
fi

if [[ "${CHECK_DEPS_ONLY}" == "1" ]]; then
    echo "[done] Dependency check completed. Exiting before model load because CHECK_DEPS_ONLY=1."
    exit 0
fi

cmd=(
    "${PYTHON_BIN}" -u SACK_Scripts/generate_concept_costs.py
    --datasets="${DATASETS}"
    --out-dir="${RESULTS_ROOT}"
    --backend="${SACK_LLM_BACKEND}"
    --model="${SACK_LLM_MODEL}"
    --temperature="${SACK_TEMPERATURE}"
    --max-tokens="${SACK_MAX_TOKENS}"
    --timeout="${SACK_TIMEOUT}"
    --sleep-secs="${SACK_SLEEP_SECS}"
    --retries="${SACK_RETRIES}"
    --torch-dtype="${SACK_TORCH_DTYPE}"
    --device-map="${SACK_DEVICE_MAP}"
    --input-price-per-1m="${SACK_INPUT_PRICE_PER_1M}"
    --output-price-per-1m="${SACK_OUTPUT_PRICE_PER_1M}"
)

if [[ "${SACK_LLM_BACKEND}" == "openai-compatible" ]]; then
    if [[ -z "${SACK_LLM_BASE_URL}" ]]; then
        echo "[error] SACK_LLM_BASE_URL is required when SACK_LLM_BACKEND=openai-compatible" >&2
        exit 1
    fi
    cmd+=(--base-url="${SACK_LLM_BASE_URL}")
fi
if [[ -n "${SACK_MODEL_CACHE_DIR}" ]]; then
    cmd+=(--model-cache-dir="${SACK_MODEL_CACHE_DIR}")
fi
if [[ "${SACK_LLM_BACKEND}" == "transformers" && "${SACK_AUTO_DOWNLOAD_MODEL}" == "1" ]]; then
    cmd+=(--auto-download)
fi
if [[ "${SACK_LOCAL_FILES_ONLY}" == "1" ]]; then
    cmd+=(--local-files-only)
fi
if [[ "${SACK_TRUST_REMOTE_CODE}" == "0" ]]; then
    cmd+=(--no-trust-remote-code)
fi

if [[ "${RESUME}" == "1" ]]; then
    cmd+=(--resume)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run)
fi
if [[ "${SKIP_ENDPOINT_CHECK}" == "1" ]]; then
    cmd+=(--skip-endpoint-check)
fi
if [[ -n "${LIMIT_CLASSES}" ]]; then
    cmd+=(--limit-classes="${LIMIT_CLASSES}")
fi

{
    printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n\n'
    printf 'cd %q\n' "${ROOT_DIR}"
    if [[ "${SACK_LLM_BACKEND}" == "transformers" ]]; then
        printf 'export PYTHONPATH=%q\n' "${PYTHONPATH:-}"
    fi
    printf ' '
    printf '%q ' "${cmd[@]}"
    printf '\n'
} > "${COMMANDS_PATH}"
chmod +x "${COMMANDS_PATH}"

echo "[info] Results root: ${RESULTS_ROOT}"
echo "[info] Log file: ${LOG_FILE}"
echo "[info] Command file: ${COMMANDS_PATH}"
echo "[info] Backend: ${SACK_LLM_BACKEND}"
if [[ "${SACK_LLM_BACKEND}" == "openai-compatible" ]]; then
    echo "[info] Endpoint: ${SACK_LLM_BASE_URL}"
fi
echo "[info] Model: ${SACK_LLM_MODEL}"
echo "[info] Datasets: ${DATASETS}"
if [[ "${SACK_LLM_BACKEND}" == "transformers" ]]; then
    echo "[info] Model cache: ${SACK_MODEL_CACHE_DIR}"
    echo "[info] Python deps overlay: ${SACK_PYDEPS_DIR}"
    echo "[info] Auto-download missing model: ${SACK_AUTO_DOWNLOAD_MODEL}"
    if [[ "${SACK_AUTO_DOWNLOAD_MODEL}" == "1" && -z "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" ]]; then
        echo "[info] HF_TOKEN is not set. Public models can still download; gated models require HF_TOKEN."
    fi
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[info] DRY_RUN=1, no endpoint calls will be made."
fi

set +e
"${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
status="${PIPESTATUS[0]}"
set -e

if [[ "${status}" != "0" ]]; then
    echo "[error] Concept generation failed with exit code ${status}. See ${LOG_FILE}" >&2
    exit "${status}"
fi

echo
echo "[done] Summary Markdown: ${RESULTS_ROOT}/concept_generation_summary.md"
echo "[done] Overall Markdown: ${RESULTS_ROOT}/concept_generation_overall.md"
echo "[done] Summary CSV: ${RESULTS_ROOT}/concept_generation_summary.csv"
echo "[done] Overall CSV: ${RESULTS_ROOT}/concept_generation_overall.csv"
echo "[done] Per-class CSV: ${RESULTS_ROOT}/concept_generation_per_class.csv"
