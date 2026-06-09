#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=probe_superb_b5
#SBATCH --array=0-4
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/probes/superb_best5/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/probes/superb_best5/%x_%A_%a.err

set -euo pipefail

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1

mkdir -p "${DIS_DIR}/logs/probes/superb_best5"
cd "${DIS_DIR}"

MODELS=(
    "sid1_weakgrl"
    "dual_weak_ub"
    "ste"
    "ub"
    "beta_003"
)

CKPTS=(
    "${DIS_DIR}/checkpoints/sid1_weakgrl/stage2_best.pt"
    "${DIS_DIR}/checkpoints/dual_weak_ub/stage2_best.pt"
    "${DIS_DIR}/checkpoints/ste/stage2_best.pt"
    "${DIS_DIR}/checkpoints/ub/stage2_best.pt"
    "${DIS_DIR}/checkpoints/beta_003/stage2_best.pt"
)

MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"
STAGE2_CKPT="${CKPTS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="superb_probe_${MODEL_NAME}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
PR_MAX_EXAMPLES="${PR_MAX_EXAMPLES:-0}"

echo "=== SUPERB-aligned best-5 probe ==="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Array task   : ${SLURM_ARRAY_TASK_ID}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started      : $(date)"
echo "run_name     : ${RUN_NAME}"
echo "model        : ${MODEL_NAME}"
echo "stage2_ckpt  : ${STAGE2_CKPT}"
echo "probe_steps  : ${PROBE_STEPS}"
echo "pr_max_examples: ${PR_MAX_EXAMPLES}"

if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "ERROR: missing checkpoint: ${STAGE2_CKPT}" >&2
    exit 2
fi

"${PYTHON}" -u probe_runner.py \
    --stage1_ckpt "${DIS_DIR}/checkpoints/best.pt" \
    --stage2_ckpt "${STAGE2_CKPT}" \
    --run_name "${RUN_NAME}" \
    --probe_steps "${PROBE_STEPS}" \
    --pr_max_examples "${PR_MAX_EXAMPLES}"

echo "Finished     : $(date)"
