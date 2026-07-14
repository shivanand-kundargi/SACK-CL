#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# One class-incremental run already produces both class-il and task-il logs in this repo.
RESULTS_PATH="${RESULTS_PATH:-results/sack_cifar100_schedule_variants_standard/${RUN_TAG}}"
SEEDS_ENV="${SEEDS:-0}"
METHODS_ENV="${METHODS:-lwf,icarl,der,derpp}"
VARIANTS_ENV="${VARIANTS:-baseline,w_to_u,u_to_w,wbar_to_u,u_to_wbar,u_to_random,random_to_u}"
NON_VERBOSE="${NON_VERBOSE:-0}"
NOTES="${NOTES:-sack_cifar100_schedule_variants_standard_${RUN_TAG}}"
DRY_RUN="${DRY_RUN:-0}"
ANALYZE_AFTER="${ANALYZE_AFTER:-0}"
BACKBONE="${BACKBONE:-}"
ALLOW_EXISTING_RESULTS="${ALLOW_EXISTING_RESULTS:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"
SKIP_COMPLETED_RUNS="${SKIP_COMPLETED_RUNS:-1}"
DETACH_RUNS="${DETACH_RUNS:-1}"
HEARTBEAT_SECS="${HEARTBEAT_SECS:-60}"
STALL_TIMEOUT_SECS="${STALL_TIMEOUT_SECS:-900}"
RUN_TIMEOUT_SECS="${RUN_TIMEOUT_SECS:-21600}"
MAX_RETRIES="${MAX_RETRIES:-1}"

CURRENT_RUN_ACTIVE=0
CURRENT_RUN_METHOD=""
CURRENT_RUN_VARIANT=""
CURRENT_RUN_SEED=""
CURRENT_RUN_LOG_FILE=""
CURRENT_RUN_PID=""
CURRENT_RUN_PID_FILE=""
REUSING_RESULTS_ROOT=0

IFS=',' read -r -a SEED_LIST <<< "${SEEDS_ENV}"
IFS=',' read -r -a METHOD_LIST <<< "${METHODS_ENV}"
IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS_ENV}"

RESULTS_ROOT="${ROOT_DIR}/data/${RESULTS_PATH}"
MANIFEST_PATH="${RESULTS_ROOT}/sweep_manifest.txt"
COMMAND_LOG_PATH="${RESULTS_ROOT}/launch_commands.sh"
RUN_LOG_DIR="${RESULTS_ROOT}/run_logs"
RUN_PID_DIR="${RESULTS_ROOT}/run_pids"
LAUNCHER_LOG_PATH="${RESULTS_ROOT}/launcher.log"

WANDB_FLAGS=()
if [[ -n "${WANDB_ENTITY:-}" && -n "${WANDB_PROJECT:-}" ]]; then
    WANDB_FLAGS+=("--wandb_entity=${WANDB_ENTITY}" "--wandb_project=${WANDB_PROJECT}")
fi

OPTIONAL_FLAGS=()
if [[ -n "${NUM_WORKERS:-}" ]]; then
    OPTIONAL_FLAGS+=("--num_workers=${NUM_WORKERS}")
fi
if [[ -n "${SAVECHECK:-}" ]]; then
    OPTIONAL_FLAGS+=("--savecheck=${SAVECHECK}")
fi
if [[ -n "${SACK_SIMILARITY_PERCENTILE:-}" ]]; then
    OPTIONAL_FLAGS+=("--sack_similarity_percentile=${SACK_SIMILARITY_PERCENTILE}")
fi
if [[ -n "${ORDER_CHANGE:-}" ]]; then
    OPTIONAL_FLAGS+=("--order_change=${ORDER_CHANGE}")
fi

normalize_method() {
    local value
    value="$(echo "$1" | tr '[:upper:]' '[:lower:]' | xargs)"
    case "${value}" in
        codaprompt|coda_prompt|coda-prompt)
            echo "coda-prompt"
            ;;
        *)
            echo "${value}"
            ;;
    esac
}

normalize_variant() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | xargs
}

method_dataset() {
    case "$1" in
        coda-prompt)
            echo "seq-cifar100-224"
            ;;
        *)
            echo "seq-cifar100"
            ;;
    esac
}

method_extra_flags() {
    case "$1" in
        lwf)
            echo "--lr=0.003"
            ;;
        icarl|der|derpp)
            echo "--buffer_size=2000"
            ;;
        coda-prompt)
            echo ""
            ;;
        *)
            echo ""
            ;;
    esac
}

method_model_config_flags() {
    case "$1" in
        lwf)
            echo ""
            ;;
        icarl|der|derpp|coda-prompt)
            echo "--model_config=best"
            ;;
        *)
            echo ""
            ;;
    esac
}

method_backbone_flags() {
    if [[ -z "${BACKBONE}" ]]; then
        echo ""
        return
    fi
    case "$1" in
        coda-prompt)
            echo ""
            ;;
        *)
            echo "--backbone=${BACKBONE}"
            ;;
    esac
}

launcher_timestamp() {
    date '+%Y-%m-%d %H:%M:%S %Z'
}

launcher_note() {
    local line
    line="[$(launcher_timestamp)] $*"
    echo "${line}"
    echo "${line}" >> "${LAUNCHER_LOG_PATH}"
}

ensure_fresh_results_root() {
    if [[ -d "${RESULTS_ROOT}" ]] && [[ "${ALLOW_EXISTING_RESULTS}" != "1" ]]; then
        if find "${RESULTS_ROOT}" -mindepth 1 -print -quit | grep -q .; then
            echo "[error] Refusing to reuse non-empty results root: ${RESULTS_ROOT}" >&2
            echo "[error] Set a new RUN_TAG/RESULTS_PATH for a fresh sweep, or ALLOW_EXISTING_RESULTS=1 to override." >&2
            exit 1
        fi
    elif [[ -d "${RESULTS_ROOT}" ]]; then
        REUSING_RESULTS_ROOT=1
        echo "[info] Reusing existing results root: ${RESULTS_ROOT}"
    fi
    mkdir -p "${RESULTS_ROOT}"
}

run_log_indicates_completed() {
    local run_log_file="$1"
    [[ -f "${run_log_file}" ]] || return 1
    rg -Fq "Logging results and arguments in " "${run_log_file}"
}

read_run_pid() {
    local run_pid_file="$1"
    [[ -f "${run_pid_file}" ]] || return 1
    tr -dc '0-9' < "${run_pid_file}"
}

run_pid_is_active() {
    local run_pid_file="$1"
    local pid
    pid="$(read_run_pid "${run_pid_file}")" || return 1
    [[ -n "${pid}" ]] || return 1
    kill -0 "${pid}" 2>/dev/null
}

terminate_run_pid() {
    local pid="$1"
    local grace_secs="${2:-10}"
    local waited=0

    [[ -n "${pid}" ]] || return 0
    if ! kill -0 "${pid}" 2>/dev/null; then
        return 0
    fi

    kill -TERM "${pid}" 2>/dev/null || true
    while kill -0 "${pid}" 2>/dev/null; do
        if (( waited >= grace_secs )); then
            kill -KILL "${pid}" 2>/dev/null || true
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
}

run_with_heartbeat() {
    local method="$1"
    local variant="$2"
    local seed="$3"
    local attempt="$4"
    local total_attempts="$5"
    local run_log_file="$6"
    local run_pid_file="$7"
    shift 7
    local cmd=("$@")
    local child_pid=""
    local start_ts=""
    local now_ts=""
    local log_mtime=""
    local elapsed_secs=""
    local idle_secs=""
    local exit_code=0

    {
        echo
        echo "=== launcher attempt ${attempt}/${total_attempts} @ $(launcher_timestamp) ==="
        printf 'COMMAND:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
    } >> "${run_log_file}"

    nohup "${cmd[@]}" >> "${run_log_file}" 2>&1 &
    child_pid="$!"
    echo "${child_pid}" > "${run_pid_file}"

    CURRENT_RUN_PID="${child_pid}"
    CURRENT_RUN_PID_FILE="${run_pid_file}"

    launcher_note "[launch] method=${method} variant=${variant} seed=${seed} attempt=${attempt}/${total_attempts} pid=${child_pid}"

    start_ts="$(date +%s)"
    while kill -0 "${child_pid}" 2>/dev/null; do
        sleep "${HEARTBEAT_SECS}"
        if ! kill -0 "${child_pid}" 2>/dev/null; then
            break
        fi

        now_ts="$(date +%s)"
        log_mtime="$(stat -c %Y "${run_log_file}" 2>/dev/null || echo "${start_ts}")"
        elapsed_secs=$((now_ts - start_ts))
        idle_secs=$((now_ts - log_mtime))

        launcher_note "[heartbeat] method=${method} variant=${variant} seed=${seed} pid=${child_pid} elapsed=${elapsed_secs}s idle=${idle_secs}s"

        if (( RUN_TIMEOUT_SECS > 0 && elapsed_secs >= RUN_TIMEOUT_SECS )); then
            launcher_note "[timeout] method=${method} variant=${variant} seed=${seed} exceeded RUN_TIMEOUT_SECS=${RUN_TIMEOUT_SECS}; terminating pid=${child_pid}"
            terminate_run_pid "${child_pid}" 10
            if wait "${child_pid}"; then
                :
            else
                :
            fi
            rm -f "${run_pid_file}"
            CURRENT_RUN_PID=""
            CURRENT_RUN_PID_FILE=""
            return 124
        fi

        if (( STALL_TIMEOUT_SECS > 0 && idle_secs >= STALL_TIMEOUT_SECS )); then
            launcher_note "[stall] method=${method} variant=${variant} seed=${seed} had no log update for ${idle_secs}s; terminating pid=${child_pid}"
            terminate_run_pid "${child_pid}" 10
            if wait "${child_pid}"; then
                :
            else
                :
            fi
            rm -f "${run_pid_file}"
            CURRENT_RUN_PID=""
            CURRENT_RUN_PID_FILE=""
            return 125
        fi
    done

    if wait "${child_pid}"; then
        exit_code=0
    else
        exit_code="$?"
    fi
    rm -f "${run_pid_file}"
    CURRENT_RUN_PID=""
    CURRENT_RUN_PID_FILE=""

    return "${exit_code}"
}

write_run_metadata() {
    local git_head="unknown"

    mkdir -p "${RUN_LOG_DIR}" "${RUN_PID_DIR}"
    touch "${LAUNCHER_LOG_PATH}"

    if [[ "${REUSING_RESULTS_ROOT}" == "1" ]]; then
        return
    fi

    if git -C "${ROOT_DIR}" rev-parse HEAD >/dev/null 2>&1; then
        git_head="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    fi

    {
        echo "run_tag=${RUN_TAG}"
        echo "created_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "results_path=${RESULTS_PATH}"
        echo "results_root=${RESULTS_ROOT}"
        echo "notes=${NOTES}"
        echo "python_bin=${PYTHON_BIN}"
        echo "backbone=${BACKBONE:-dataset-default}"
        echo "seeds=${SEEDS_ENV}"
        echo "methods=${METHODS_ENV}"
        echo "variants=${VARIANTS_ENV}"
        echo "non_verbose=${NON_VERBOSE}"
        echo "analyze_after=${ANALYZE_AFTER}"
        echo "git_head=${git_head}"
    } > "${MANIFEST_PATH}"

    {
        printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n\n'
        printf '# Commands are printed live during execution; this file is intentionally static\n'
    } > "${COMMAND_LOG_PATH}"
    mkdir -p "${RUN_LOG_DIR}"
}

mark_current_run_interrupted() {
    CURRENT_RUN_ACTIVE=0
}

handle_interrupt() {
    if [[ "${CURRENT_RUN_ACTIVE}" == "1" ]]; then
        launcher_note "[interrupt] launcher interrupted while method=${CURRENT_RUN_METHOD} variant=${CURRENT_RUN_VARIANT} seed=${CURRENT_RUN_SEED} was active"
        if [[ -n "${CURRENT_RUN_PID}" ]]; then
            launcher_note "[interrupt] detached child pid=${CURRENT_RUN_PID} was left running intentionally"
        fi
    fi
    mark_current_run_interrupted
    exit 130
}

handle_exit() {
    CURRENT_RUN_ACTIVE=0
}

trap handle_interrupt INT TERM
trap handle_exit EXIT

ensure_fresh_results_root
write_run_metadata

if [[ "${REUSING_RESULTS_ROOT}" == "1" ]]; then
    echo "[info] Results root: ${RESULTS_ROOT} (reused)"
else
    echo "[info] Fresh results root: ${RESULTS_ROOT}"
fi
echo "[info] Manifest: ${MANIFEST_PATH}"
echo "[info] Command log: ${COMMAND_LOG_PATH}"
echo "[info] Per-run stdout/stderr logs: ${RUN_LOG_DIR}"
echo "[info] Launcher log: ${LAUNCHER_LOG_PATH}"
if [[ "${DETACH_RUNS}" == "1" ]]; then
    echo "[info] Child runs are detached from the terminal with heartbeat polling."
elif [[ "${STREAM_LOGS}" == "1" ]]; then
    echo "[info] Progress is printed live and also saved under run_logs/."
else
    echo "[info] Progress is saved under run_logs/."
fi
echo "[info] Skip completed runs: ${SKIP_COMPLETED_RUNS}"
echo "[info] Heartbeat every ${HEARTBEAT_SECS}s | stall timeout ${STALL_TIMEOUT_SECS}s | run timeout ${RUN_TIMEOUT_SECS}s | max retries ${MAX_RETRIES}"

for raw_method in "${METHOD_LIST[@]}"; do
    method="$(normalize_method "${raw_method}")"
    dataset="$(method_dataset "${method}")"

    case "${method}" in
        lwf|icarl|der|derpp|coda-prompt)
            ;;
        *)
            echo "[error] Unsupported method: ${raw_method}" >&2
            exit 1
            ;;
    esac

    extra_flags_string="$(method_extra_flags "${method}")"
    extra_flags=()
    if [[ -n "${extra_flags_string}" ]]; then
        read -r -a extra_flags <<< "${extra_flags_string}"
    fi

    model_config_flags_string="$(method_model_config_flags "${method}")"
    model_config_flags=()
    if [[ -n "${model_config_flags_string}" ]]; then
        read -r -a model_config_flags <<< "${model_config_flags_string}"
    fi

    backbone_flags_string="$(method_backbone_flags "${method}")"
    backbone_flags=()
    if [[ -n "${backbone_flags_string}" ]]; then
        read -r -a backbone_flags <<< "${backbone_flags_string}"
    elif [[ "${method}" == "coda-prompt" && -n "${BACKBONE}" ]]; then
        echo "[warn] coda-prompt uses its custom ViT backbone in this repo; skipping BACKBONE=${BACKBONE}" >&2
    fi

    for raw_variant in "${VARIANT_LIST[@]}"; do
        variant="$(normalize_variant "${raw_variant}")"
        case "${variant}" in
            baseline|w_to_u|u_to_w|wbar_to_u|u_to_wbar|u_to_random|u_to_random_fixed|random_to_u)
                ;;
            *)
                echo "[error] Unsupported variant: ${raw_variant}" >&2
                exit 1
                ;;
        esac

        for seed in "${SEED_LIST[@]}"; do
            run_name="cifar100-${method}-${variant}-seed${seed}"
            run_log_file="${RUN_LOG_DIR}/${run_name}.log"
            run_pid_file="${RUN_PID_DIR}/${run_name}.pid"
            if run_pid_is_active "${run_pid_file}"; then
                active_pid="$(read_run_pid "${run_pid_file}")"
                launcher_note "[active] method=${method} variant=${variant} seed=${seed} is already running with pid=${active_pid}. Exiting launcher so we do not start later jobs in parallel."
                exit 0
            fi
            rm -f "${run_pid_file}"
            if [[ "${SKIP_COMPLETED_RUNS}" == "1" ]] && run_log_indicates_completed "${run_log_file}"; then
                echo "[skip] method=${method} variant=${variant} seed=${seed} already completed according to ${run_log_file}"
                continue
            fi
            sack_flag="--sack=0"
            variant_flags=()
            if [[ "${variant}" != "baseline" ]]; then
                sack_flag="--sack=1"
                variant_flags+=("--sack_schedule_variant=${variant}")
            fi

            cmd=(
                "${PYTHON_BIN}" "-u" "${ROOT_DIR}/main.py"
                "--dataset=${dataset}"
                "--model=${method}"
                "--seed=${seed}"
                "${sack_flag}"
                "--sack_scores_type=0"
                "--results_path=${RESULTS_PATH}"
                "--enable_other_metrics=True"
                "--permute_classes=True"
                "--notes=${NOTES}"
                "--non_verbose=${NON_VERBOSE}"
            )
            if [[ ${#model_config_flags[@]} -gt 0 ]]; then
                cmd+=("${model_config_flags[@]}")
            fi
            if [[ ${#variant_flags[@]} -gt 0 ]]; then
                cmd+=("${variant_flags[@]}")
            fi

            if [[ ${#WANDB_FLAGS[@]} -gt 0 ]]; then
                cmd+=("${WANDB_FLAGS[@]}" "--wandb_name=${run_name}")
            fi
            if [[ ${#OPTIONAL_FLAGS[@]} -gt 0 ]]; then
                cmd+=("${OPTIONAL_FLAGS[@]}")
            fi
            if [[ ${#extra_flags[@]} -gt 0 ]]; then
                cmd+=("${extra_flags[@]}")
            fi
            if [[ ${#backbone_flags[@]} -gt 0 ]]; then
                cmd+=("${backbone_flags[@]}")
            fi

            echo "[run] method=${method} variant=${variant} seed=${seed} dataset=${dataset}"
            printf '  %q' "${cmd[@]}"
            printf '\n'
            echo "  log: ${run_log_file}"
            CURRENT_RUN_ACTIVE=1
            CURRENT_RUN_METHOD="${method}"
            CURRENT_RUN_VARIANT="${variant}"
            CURRENT_RUN_SEED="${seed}"
            CURRENT_RUN_LOG_FILE="${run_log_file}"

            if [[ "${DRY_RUN}" != "1" ]]; then
                total_attempts=$((MAX_RETRIES + 1))
                attempt=1
                while true; do
                    if [[ "${DETACH_RUNS}" == "1" ]]; then
                        if run_with_heartbeat "${method}" "${variant}" "${seed}" "${attempt}" "${total_attempts}" "${run_log_file}" "${run_pid_file}" "${cmd[@]}"; then
                            exit_code=0
                        else
                            exit_code="$?"
                        fi
                    elif [[ "${STREAM_LOGS}" == "1" ]]; then
                        set +o pipefail
                        "${cmd[@]}" 2>&1 | tee "${run_log_file}"
                        exit_code="${PIPESTATUS[0]}"
                        set -o pipefail
                    else
                        if "${cmd[@]}" >"${run_log_file}" 2>&1; then
                            exit_code="0"
                        else
                            exit_code="$?"
                        fi
                    fi

                    if [[ "${exit_code}" == "0" ]]; then
                        echo "[done] method=${method} variant=${variant} seed=${seed} exit_code=0"
                        break
                    fi

                    if (( attempt >= total_attempts )); then
                        CURRENT_RUN_ACTIVE=0
                        echo "[error] method=${method} variant=${variant} seed=${seed} failed after ${attempt} attempt(s). See ${run_log_file}" >&2
                        exit "${exit_code}"
                    fi

                    launcher_note "[retry] method=${method} variant=${variant} seed=${seed} attempt=${attempt}/${total_attempts} exit_code=${exit_code}; retrying"
                    attempt=$((attempt + 1))
                    sleep 5
                done
            fi
            CURRENT_RUN_ACTIVE=0
            CURRENT_RUN_METHOD=""
            CURRENT_RUN_VARIANT=""
            CURRENT_RUN_SEED=""
            CURRENT_RUN_LOG_FILE=""
            CURRENT_RUN_PID=""
            CURRENT_RUN_PID_FILE=""
        done
    done
done

if [[ "${ANALYZE_AFTER}" == "1" ]]; then
    "${PYTHON_BIN}" "${ROOT_DIR}/SACK_Scripts/analyze_sack_cifar100_schedule_variants.py" \
        --results-root "${RESULTS_ROOT}"
fi
