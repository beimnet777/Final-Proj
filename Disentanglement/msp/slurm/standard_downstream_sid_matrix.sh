#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=25:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=std_sid_matrix
#SBATCH --array=0-8%4
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.err

# SUPERB-style VoxCeleb1 closed-set SID for the three checkpoints × three
# representations.  The raw SPEAR reference is intentionally not duplicated;
# its existing full four-epoch result is the comparison point.  This array keeps
# the exact 4-epoch, 8-second train-crop, projected mean-pool linear protocol.

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
VOXCELEB1_ROOT="${VOXCELEB1_ROOT:-/rds/user/${USER}/hpc-work/data/VoxCeleb1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/Probing/sid/standard_downstream}"

declare -a RUN_NAMES=(
  "libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42"
  "msp_hardqfreeze4000_strongclean_dann12000_advpe010_s42"
  "msp_hardqfreeze4000_sc_optfix_nobal_aux64_gn0004_grlp025_dann12000_12k_s42"
)
declare -a RUN_KINDS=("libri" "msp" "msp")
declare -a SOURCES=("z_t" "z_L" "z_P")

if (( TASK_ID < 0 || TASK_ID > 8 )); then
  echo "Unknown TASK_ID=${TASK_ID}; expected 0..8." >&2
  exit 2
fi

RUN_INDEX=$((TASK_ID / 3))
SOURCE_INDEX=$((TASK_ID % 3))
RUN_NAME="${RUN_NAMES[${RUN_INDEX}]}"
SOURCE="${SOURCES[${SOURCE_INDEX}]}"
if [[ "${RUN_KINDS[${RUN_INDEX}]}" == "libri" ]]; then
  CHECKPOINT="${REPO_ROOT}/checkpoints/blackwell/${RUN_NAME}/final.pt"
else
  CHECKPOINT="${DIS_DIR}/msp/checkpoints/${RUN_NAME}/final.pt"
fi
LABEL="${RUN_NAME}_${SOURCE}"

RUN_DIR="${OUTPUT_ROOT}/runs/${LABEL}_seed${SEED}"
PROBE_CKPT_DIR="${OUTPUT_ROOT}/checkpoints/${LABEL}_seed${SEED}"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUN_DIR}" "${PROBE_CKPT_DIR}" "${DIS_DIR}/msp/logs"
  cd "${REPO_ROOT}"
  [[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 3; }
  [[ -d "${VOXCELEB1_ROOT}/dev/wav" ]] || {
    echo "Missing VoxCeleb1 dev/wav: ${VOXCELEB1_ROOT}" >&2; exit 4;
  }
  [[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 5; }
fi

MAX_EXAMPLES=0
EPOCHS=4
if [[ "${SMOKE:-0}" == "1" ]]; then
  MAX_EXAMPLES=4
  EPOCHS=1
fi

COMMAND=("${PYTHON}" -u Probing/sid/sid_run.py
  --probe final
  --voxceleb1_root "${VOXCELEB1_ROOT}"
  --model_family disentanglement
  --checkpoint_path "${CHECKPOINT}"
  --representation_source "${SOURCE}"
  --epochs "${EPOCHS}"
  --batch_size 8
  --eval_batch_size 8
  --lr 4e-4
  --warmup_steps 100
  --num_workers 0
  --train_max_duration_s 8.0
  --max_examples "${MAX_EXAMPLES}"
  --seed "${SEED}"
  --checkpoint_dir "${PROBE_CKPT_DIR}"
  --runs_dir "${RUN_DIR}")

echo "=== Standard downstream SID ==="
echo "task_id    : ${TASK_ID}"
echo "label      : ${LABEL}"
echo "checkpoint : ${CHECKPOINT}"
echo "epochs     : ${EPOCHS}"
printf '+ %q ' "${COMMAND[@]}"; printf '\n'
if [[ "${DRY_RUN}" != "1" ]]; then
  "${COMMAND[@]}"
fi
