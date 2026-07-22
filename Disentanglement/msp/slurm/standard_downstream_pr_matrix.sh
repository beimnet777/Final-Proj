#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=std_pr_matrix
#SBATCH --array=0-9%4
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.err

# SUPERB-style LibriSpeech PR comparison.  The raw SPEAR reference and every
# checkpoint route use the same official tokenizer, projected-linear CTC head,
# data splits, optimizer, schedule, seed, and validation selection.  Eight
# epochs are used (shorter than the original 10-epoch local reference).

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module purge 2>/dev/null || true
  module load rhel8/default-amp 2>/dev/null || true
fi

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
REPO_ROOT="${REPO_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj}"
DIS_DIR="${REPO_ROOT}/Disentanglement"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-42}"
LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-${REPO_ROOT}/Probing/data/LibriSpeech}"
LEXICON_PATH="${LEXICON_PATH:-${REPO_ROOT}/Probing/data/librispeech-lexicon.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Probing/pr/standard_downstream}"

declare -a RUN_NAMES=(
  "libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42"
  "msp_hardqfreeze4000_strongclean_dann12000_advpe010_s42"
  "msp_hardqfreeze4000_sc_optfix_nobal_aux64_gn0004_grlp025_dann12000_12k_s42"
)
declare -a RUN_KINDS=("libri" "msp" "msp")
declare -a SOURCES=("z_t" "z_L" "z_P")

if (( TASK_ID < 0 || TASK_ID > 9 )); then
  echo "Unknown TASK_ID=${TASK_ID}; expected 0..9." >&2
  exit 2
fi

declare -a MODEL_ARGS=()
if (( TASK_ID == 0 )); then
  LABEL="raw_spear"
  MODEL_ARGS=(--model_family spear)
else
  OFFSET=$((TASK_ID - 1))
  RUN_INDEX=$((OFFSET / 3))
  SOURCE_INDEX=$((OFFSET % 3))
  RUN_NAME="${RUN_NAMES[${RUN_INDEX}]}"
  SOURCE="${SOURCES[${SOURCE_INDEX}]}"
  if [[ "${RUN_KINDS[${RUN_INDEX}]}" == "libri" ]]; then
    CHECKPOINT="${REPO_ROOT}/checkpoints/blackwell/${RUN_NAME}/final.pt"
  else
    CHECKPOINT="${DIS_DIR}/msp/checkpoints/${RUN_NAME}/final.pt"
  fi
  LABEL="${RUN_NAME}_${SOURCE}"
  MODEL_ARGS=(
    --model_family disentanglement
    --checkpoint_path "${CHECKPOINT}"
    --representation_source "${SOURCE}"
  )
fi

RUN_DIR="${OUTPUT_ROOT}/runs/${LABEL}_seed${SEED}"
PROBE_CKPT_DIR="${OUTPUT_ROOT}/checkpoints/${LABEL}_seed${SEED}"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUN_DIR}" "${PROBE_CKPT_DIR}" "${DIS_DIR}/msp/logs"
  cd "${REPO_ROOT}"
  [[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 3; }
  [[ -d "${LIBRISPEECH_ROOT}/train-clean-100" ]] || {
    echo "Missing LibriSpeech train-clean-100: ${LIBRISPEECH_ROOT}" >&2; exit 4;
  }
  [[ -f "${LEXICON_PATH}" ]] || { echo "Missing lexicon: ${LEXICON_PATH}" >&2; exit 5; }
  if (( TASK_ID > 0 )); then
    [[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 6; }
  fi
fi

MAX_EXAMPLES=0
EPOCHS=8
if [[ "${SMOKE:-0}" == "1" ]]; then
  MAX_EXAMPLES=4
  EPOCHS=1
fi

COMMAND=("${PYTHON}" -u Probing/pr/pr_run.py
  --probe final
  --epochs "${EPOCHS}"
  --batch_size 16
  --eval_batch_size 16
  --lr 5e-4
  --warmup_steps 500
  --local_data
  --librispeech_root "${LIBRISPEECH_ROOT}"
  --data_cache_dir "${REPO_ROOT}/Probing/data"
  --lexicon_path "${LEXICON_PATH}"
  --num_workers 8
  --max_examples "${MAX_EXAMPLES}"
  --seed "${SEED}"
  --checkpoint_dir "${PROBE_CKPT_DIR}"
  --runs_dir "${RUN_DIR}"
  "${MODEL_ARGS[@]}")

echo "=== Standard downstream PR ==="
echo "task_id : ${TASK_ID}"
echo "label   : ${LABEL}"
echo "epochs  : ${EPOCHS}"
printf '+ %q ' "${COMMAND[@]}"; printf '\n'
if [[ "${DRY_RUN}" != "1" ]]; then
  "${COMMAND[@]}"
fi
