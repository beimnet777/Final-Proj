#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=diag_probe_b5
#SBATCH --array=0-4%1
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/probes/diag_best5_seeded/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/probes/diag_best5_seeded/%x_%A_%a.err

set -euo pipefail

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"

mkdir -p "${DIS_DIR}/logs/probes/diag_best5_seeded"
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
RUN_NAME="diag_probe_${MODEL_NAME}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
PR_MAX_EXAMPLES="${PR_MAX_EXAMPLES:-0}"
SEED="${SEED:-42}"
SOURCES="${SOURCES:-z_t,z_L,z_P}"
TASKS="${TASKS:-pr,sid}"
PR_PROBE_LR="${PR_PROBE_LR:-5e-4}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PROBE_WARMUP_STEPS="${PROBE_WARMUP_STEPS:-0}"

echo "=== Disentanglement diagnostic best-5 probe ==="
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Array task        : ${SLURM_ARRAY_TASK_ID}"
echo "Node              : $(hostname)"
echo "GPU               : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started           : $(date)"
echo "run_name          : ${RUN_NAME}"
echo "model             : ${MODEL_NAME}"
echo "stage2_ckpt       : ${STAGE2_CKPT}"
echo "sources           : ${SOURCES}"
echo "tasks             : ${TASKS}"
echo "seed              : ${SEED}"
echo "probe_steps       : ${PROBE_STEPS}"
echo "pr_probe_lr       : ${PR_PROBE_LR}"
echo "sid_probe_lr      : ${SID_PROBE_LR}"
echo "warmup_steps      : ${PROBE_WARMUP_STEPS}"
echo "pr_max_examples   : ${PR_MAX_EXAMPLES}"

if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "ERROR: missing checkpoint: ${STAGE2_CKPT}" >&2
    exit 2
fi

"${PYTHON}" -u diag_probe/run.py \
    --stage1_ckpt "${DIS_DIR}/checkpoints/best.pt" \
    --stage2_ckpt "${STAGE2_CKPT}" \
    --run_name "${RUN_NAME}" \
    --sources "${SOURCES}" \
    --tasks "${TASKS}" \
    --probe_steps "${PROBE_STEPS}" \
    --seed "${SEED}" \
    --pr_max_examples "${PR_MAX_EXAMPLES}" \
    --pr_probe_lr "${PR_PROBE_LR}" \
    --sid_probe_lr "${SID_PROBE_LR}" \
    --probe_warmup_steps "${PROBE_WARMUP_STEPS}"

echo "Finished          : $(date)"
