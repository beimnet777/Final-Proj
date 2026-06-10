#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=proj_p8000_std
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/probes/projection/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/probes/projection/%x_%j.err

# Controlled re-probe of the projection step-8000 checkpoint WITH per-dim
# standardization of z_L/z_P.  The plain probe (probe_projection_step8000_diag.sh)
# gave PER ~ 0.97 on both views, which may be a feature-scale artifact: proj
# weights grew to ||W_P|| ~ 48, so z_L/z_P have large magnitude -> saturated
# probe softmax -> CTC collapses to blank.  Only difference vs the plain probe is
# --standardize_sources (same ckpt / lr / steps / seed), so the comparison is
# clean.  z_t is included as a control: it is NOT standardized, so it must still
# reproduce the ~0.06 PER / 1.0 SID baseline, confirming the pipeline is healthy.

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
RUN_NAME="${RUN_NAME:-diag_probe_${MODEL_NAME}_step8000_std}"

SOURCES="${SOURCES:-z_t,z_L,z_P}"
TASKS="${TASKS:-pr,sid}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
PR_PROBE_LR="${PR_PROBE_LR:-5e-4}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PROBE_WARMUP_STEPS="${PROBE_WARMUP_STEPS:-0}"
PR_MAX_EXAMPLES="${PR_MAX_EXAMPLES:-0}"
SEED="${SEED:-42}"

echo "=== Projection step-8000 probe (STANDARDIZED z_L/z_P) ==="
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node              : $(hostname)"
echo "GPU               : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started           : $(date)"
echo "model             : ${MODEL_NAME}"
echo "run_name          : ${RUN_NAME}"
echo "stage1_ckpt       : ${STAGE1_CKPT}"
echo "stage2_ckpt       : ${STAGE2_CKPT}"
echo "sources           : ${SOURCES}"
echo "tasks             : ${TASKS}"
echo "seed              : ${SEED}"
echo "probe_steps       : ${PROBE_STEPS}"
echo "pr_probe_lr       : ${PR_PROBE_LR}"
echo "sid_probe_lr      : ${SID_PROBE_LR}"
echo "warmup_steps      : ${PROBE_WARMUP_STEPS}"
echo "pr_max_examples   : ${PR_MAX_EXAMPLES}"
echo "standardize       : true (z_L,z_P only)"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
    echo "ERROR: missing stage1 checkpoint: ${STAGE1_CKPT}" >&2
    exit 2
fi

if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "ERROR: missing stage2 checkpoint: ${STAGE2_CKPT}" >&2
    exit 3
fi

"${PYTHON}" -u diag_probe/run.py \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --stage2_ckpt "${STAGE2_CKPT}" \
    --run_name "${RUN_NAME}" \
    --sources "${SOURCES}" \
    --tasks "${TASKS}" \
    --probe_steps "${PROBE_STEPS}" \
    --seed "${SEED}" \
    --pr_max_examples "${PR_MAX_EXAMPLES}" \
    --pr_probe_lr "${PR_PROBE_LR}" \
    --sid_probe_lr "${SID_PROBE_LR}" \
    --probe_warmup_steps "${PROBE_WARMUP_STEPS}" \
    --standardize_sources

echo "Finished          : $(date)"
