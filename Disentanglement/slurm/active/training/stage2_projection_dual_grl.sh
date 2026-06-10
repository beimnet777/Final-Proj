#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=proj_dual
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection/%x_%j.err

# Projection disentanglement:
#   z_L = true compressed projection of z_t
#   z_P = independent true compressed projection of z_t
#   dual GRL: speaker adversary on z_L, phoneme adversary on z_P

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

mkdir -p "${DIS_DIR}/logs/train/stage2/projection"
mkdir -p "${DIS_DIR}/logs/probes/projection"
cd "${DIS_DIR}"

RUN_NAME="${RUN_NAME:-projection_dual_grl_gp002_d128}"
STAGE1_CKPT="${DIS_DIR}/checkpoints/best.pt"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"
PROBE_RUN_NAME="diag_probe_${RUN_NAME}"

STAGE2_STEPS="${STAGE2_STEPS:-8000}"
PROJECTION_DIM="${PROJECTION_DIM:-128}"
SEED="${SEED:-42}"
ALPHA="${ALPHA:-0.02}"
BETA="${BETA:-0.03}"
GRL_WEIGHT="${GRL_WEIGHT:-0.01}"
GRL_P_WEIGHT="${GRL_P_WEIGHT:-0.002}"

PROBE_STEPS="${PROBE_STEPS:-2000}"
TASKS="${TASKS:-pr,sid}"
SOURCES="${SOURCES:-z_t,z_L,z_P}"
PR_PROBE_LR="${PR_PROBE_LR:-5e-4}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PROBE_WARMUP_STEPS="${PROBE_WARMUP_STEPS:-0}"
PR_MAX_EXAMPLES="${PR_MAX_EXAMPLES:-0}"

echo "=== Stage 2 + diagnostic probe: projection dual GRL ==="
echo "Job ID             : ${SLURM_JOB_ID}"
echo "Node               : $(hostname)"
echo "GPU                : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started            : $(date)"
echo "run_name           : ${RUN_NAME}"
echo "stage1_ckpt        : ${STAGE1_CKPT}"
echo "stage2_ckpt        : ${STAGE2_CKPT}"
echo "projection         : true"
echo "projection_dim     : ${PROJECTION_DIM}"
echo "alpha              : ${ALPHA}"
echo "beta               : ${BETA}"
echo "grl_weight         : ${GRL_WEIGHT}"
echo "grl_p_weight       : ${GRL_P_WEIGHT}"
echo "rho                : 0"
echo "seed               : ${SEED}"
echo "stage2_steps       : ${STAGE2_STEPS}"
echo "probe_run_name     : ${PROBE_RUN_NAME}"
echo "probe_sources      : ${SOURCES}"
echo "probe_tasks        : ${TASKS}"
echo "probe_steps        : ${PROBE_STEPS}"
echo "pr_probe_lr        : ${PR_PROBE_LR}"
echo "sid_probe_lr       : ${SID_PROBE_LR}"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
    echo "ERROR: missing stage1 checkpoint: ${STAGE1_CKPT}" >&2
    exit 2
fi

"${PYTHON}" -u run.py \
    --stage              2 \
    --stage1_ckpt        "${STAGE1_CKPT}" \
    --stage2_steps       "${STAGE2_STEPS}" \
    --warmup_steps       500 \
    --alpha              "${ALPHA}" \
    --beta               "${BETA}" \
    --grl_weight         "${GRL_WEIGHT}" \
    --grl_delay_steps    0 \
    --grl_phoneme_weight "${GRL_P_WEIGHT}" \
    --rho                0 \
    --lr                 3e-5 \
    --lr_min             1e-6 \
    --lr_routing         5e-6 \
    --lr_heads           1e-4 \
    --max_train_examples 0 \
    --max_val_examples   500 \
    --grad_log_every     500 \
    --checkpoint_dir     "${CKPT_DIR}" \
    --runs_dir           "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir            "${DIS_DIR}/logs" \
    --seed               "${SEED}" \
    --projection_disentanglement \
    --projection_dim     "${PROJECTION_DIM}"

if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "ERROR: stage2 training finished but checkpoint is missing: ${STAGE2_CKPT}" >&2
    exit 3
fi

echo "=== Diagnostic probe: ${PROBE_RUN_NAME} ==="
echo "Probe started      : $(date)"

"${PYTHON}" -u diag_probe/run.py \
    --stage1_ckpt "${STAGE1_CKPT}" \
    --stage2_ckpt "${STAGE2_CKPT}" \
    --run_name "${PROBE_RUN_NAME}" \
    --sources "${SOURCES}" \
    --tasks "${TASKS}" \
    --probe_steps "${PROBE_STEPS}" \
    --seed "${SEED}" \
    --pr_max_examples "${PR_MAX_EXAMPLES}" \
    --pr_probe_lr "${PR_PROBE_LR}" \
    --sid_probe_lr "${SID_PROBE_LR}" \
    --probe_warmup_steps "${PROBE_WARMUP_STEPS}"

echo "Finished           : $(date)"
