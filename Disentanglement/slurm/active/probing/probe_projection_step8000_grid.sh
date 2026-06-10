#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=proj_p8000_grid
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/probes/projection/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/probes/projection/%x_%j.err

# Parameterized re-probe of the projection step-8000 checkpoint to disambiguate
# the PER~0.97 puzzle along two independent axes:
#   PR_LABEL_SET = superb | dis   (transfer 74-phone eval vs native 41-phone eval)
#   STANDARDIZE  = 0 | 1          (per-dim z-score z_L/z_P before probing)
# Combine with the existing (superb,0) and queued (superb,1) runs to form a 2x2.

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

mkdir -p "${DIS_DIR}/logs/probes/projection"
cd "${DIS_DIR}"

MODEL_NAME="${MODEL_NAME:-projection_dual_grl_gp002_d128}"
STAGE1_CKPT="${STAGE1_CKPT:-${DIS_DIR}/checkpoints/best.pt}"
STAGE2_CKPT="${STAGE2_CKPT:-${DIS_DIR}/checkpoints/${MODEL_NAME}/stage2_step8000.pt}"

PR_LABEL_SET="${PR_LABEL_SET:-dis}"
STANDARDIZE="${STANDARDIZE:-1}"
SOURCES="${SOURCES:-z_t,z_L,z_P}"
TASKS="${TASKS:-pr,sid}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
PR_PROBE_LR="${PR_PROBE_LR:-5e-4}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PROBE_WARMUP_STEPS="${PROBE_WARMUP_STEPS:-0}"
PR_MAX_EXAMPLES="${PR_MAX_EXAMPLES:-0}"
SEED="${SEED:-42}"

STD_TAG="nostd"; STD_FLAG=()
if [[ "${STANDARDIZE}" == "1" ]]; then STD_TAG="std"; STD_FLAG=(--standardize_sources); fi
RUN_NAME="${RUN_NAME:-diag_probe_${MODEL_NAME}_step8000_${PR_LABEL_SET}_${STD_TAG}}"

echo "=== Projection step-8000 grid probe ==="
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node              : $(hostname)"
echo "GPU               : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started           : $(date)"
echo "model             : ${MODEL_NAME}"
echo "run_name          : ${RUN_NAME}"
echo "stage2_ckpt       : ${STAGE2_CKPT}"
echo "sources           : ${SOURCES}"
echo "pr_label_set      : ${PR_LABEL_SET}"
echo "standardize       : ${STANDARDIZE}"
echo "seed              : ${SEED}"

if [[ ! -f "${STAGE1_CKPT}" ]]; then echo "ERROR: missing stage1 ckpt: ${STAGE1_CKPT}" >&2; exit 2; fi
if [[ ! -f "${STAGE2_CKPT}" ]]; then echo "ERROR: missing stage2 ckpt: ${STAGE2_CKPT}" >&2; exit 3; fi

"${PYTHON}" -u diag_probe/run.py \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --stage2_ckpt "${STAGE2_CKPT}" \
    --run_name "${RUN_NAME}" \
    --sources "${SOURCES}" \
    --tasks "${TASKS}" \
    --probe_steps "${PROBE_STEPS}" \
    --seed "${SEED}" \
    --pr_max_examples "${PR_MAX_EXAMPLES}" \
    --pr_label_set "${PR_LABEL_SET}" \
    --pr_probe_lr "${PR_PROBE_LR}" \
    --sid_probe_lr "${SID_PROBE_LR}" \
    --probe_warmup_steps "${PROBE_WARMUP_STEPS}" \
    "${STD_FLAG[@]}"

echo "Finished          : $(date)"
