#!/usr/bin/env bash
# Shared Blackwell runtime. Source this from a tracked experiment script.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source this file from an experiment script; do not execute it directly." >&2
    exit 2
fi

BLACKWELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BLACKWELL_DIR}/../.." && pwd)"
BLACKWELL_SCRATCH_ROOT="${BLACKWELL_SCRATCH_ROOT:-/scratch/${USER:?USER is not set}}"
BLACKWELL_VENV="${BLACKWELL_VENV:-${BLACKWELL_SCRATCH_ROOT}/venvs/final-proj}"
BLACKWELL_DATA_ROOT="${BLACKWELL_DATA_ROOT:-${BLACKWELL_SCRATCH_ROOT}/data}"
BLACKWELL_OUTPUT_ROOT="${BLACKWELL_OUTPUT_ROOT:-${BLACKWELL_SCRATCH_ROOT}/runs}"

export REPO_ROOT BLACKWELL_SCRATCH_ROOT BLACKWELL_VENV
export BLACKWELL_DATA_ROOT BLACKWELL_OUTPUT_ROOT
export HF_HOME="${HF_HOME:-${BLACKWELL_SCRATCH_ROOT}/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export NLTK_DATA="${NLTK_DATA:-${BLACKWELL_SCRATCH_ROOT}/nltk_data}"
export PYTHONUNBUFFERED=1

blackwell_run() {
    if [[ $# -lt 2 ]]; then
        echo "Usage: blackwell_run RUN_NAME COMMAND [ARG ...]" >&2
        return 2
    fi
    local run_name="$1"
    shift

    if [[ ! "$run_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "ERROR: RUN_NAME may contain only letters, digits, dot, underscore and dash." >&2
        return 2
    fi
    if [[ ! "${GPU_ID:-}" =~ ^[0-7]$ ]]; then
        echo "ERROR: set GPU_ID to the single physical GPU (0-7) allocated in blackwell_gpus." >&2
        echo "Example: GPU_ID=3 ./Disentanglement/blackwell/experiments/my_run.sh" >&2
        return 2
    fi
    if [[ ! -x "${BLACKWELL_VENV}/bin/python" ]]; then
        echo "ERROR: environment not found at ${BLACKWELL_VENV}; run blackwell/setup.sh first." >&2
        return 1
    fi
    if [[ ! -d "${BLACKWELL_DATA_ROOT}" ]]; then
        echo "ERROR: data root not found: ${BLACKWELL_DATA_ROOT}" >&2
        return 1
    fi

    # CUDA is not initialized until after the manually allocated physical ID is
    # hidden behind logical cuda:0. The trainer is currently single-GPU.
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$NLTK_DATA"
    # shellcheck disable=SC1091
    source "${BLACKWELL_VENV}/bin/activate"

    local run_dir="${BLACKWELL_OUTPUT_ROOT}/${run_name}"
    local log_dir="${run_dir}/launcher_logs"
    local stamp log_file metadata_file commit dirty
    stamp="$(date +%Y%m%d_%H%M%S)"
    log_file="${log_dir}/${stamp}.log"
    metadata_file="${log_dir}/${stamp}.metadata"
    mkdir -p "$log_dir"
    export BLACKWELL_RUN_DIR="$run_dir"

    commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
    dirty="$(git -C "$REPO_ROOT" status --porcelain)"
    {
        printf 'run_name=%s\n' "$run_name"
        printf 'started=%s\n' "$(date --iso-8601=seconds)"
        printf 'host=%s\n' "$(hostname)"
        printf 'user=%s\n' "$USER"
        printf 'git_commit=%s\n' "$commit"
        printf 'git_worktree=%s\n' "$([[ -z "$dirty" ]] && echo clean || echo dirty)"
        printf 'physical_gpu=%s\n' "$GPU_ID"
        printf 'data_root=%s\n' "$BLACKWELL_DATA_ROOT"
        printf 'run_dir=%s\n' "$run_dir"
        printf 'command='
        printf '%q ' "$@"
        printf '\n'
    } | tee "$metadata_file"

    "${BLACKWELL_VENV}/bin/python" - <<'PY' | tee -a "$metadata_file"
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot use the assigned GPU")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"ERROR: expected exactly one visible GPU, found {torch.cuda.device_count()}")
p = torch.cuda.get_device_properties(0)
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"logical_gpu=0")
print(f"gpu_name={p.name}")
print(f"gpu_memory_gib={p.total_memory / 2**30:.1f}")
x = torch.randn(256, 256, device="cuda")
float((x @ x).mean())
print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
print("cuda_smoke_test=passed")
PY

    echo "Full log: $log_file"
    local status
    set +e
    (
        cd "$REPO_ROOT"
        "$@"
    ) 2>&1 | tee "$log_file"
    status=${PIPESTATUS[0]}
    set -e
    {
        printf 'finished=%s\n' "$(date --iso-8601=seconds)"
        printf 'exit_status=%s\n' "$status"
    } | tee -a "$metadata_file"
    return "$status"
}

