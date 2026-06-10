#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=strong_grl_t2
#SBATCH --array=0-1%1
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/strong_grl_top2/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/strong_grl_top2/%x_%A_%a.err

# Top-2 rerun with stronger adversarial heads:
#   - z_L speaker GRL: diagnostic-SID-style projector + mean pool + classifier
#   - z_P phone GRL: diagnostic-PR-style projector + frame classifier
#
# Array tasks:
#   0: strong_grl_sid1_weakgrl
#   1: strong_grl_dual_weak_ub

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

mkdir -p "${DIS_DIR}/logs/train/stage2/strong_grl_top2"
mkdir -p "${DIS_DIR}/logs/probes/strong_grl_top2"
cd "${DIS_DIR}"

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        BASE_NAME="sid1_weakgrl"
        RUN_NAME="strong_grl_sid1_weakgrl"
        ALPHA="0.02"
        BETA="0.01"
        GRL_WEIGHT="0.01"
        RHO="0.001"
        EXTRA_ARGS=()
        ;;
    1)
        BASE_NAME="dual_weak_ub"
        RUN_NAME="strong_grl_dual_weak_ub"
        ALPHA="0.02"
        BETA="0.01"
        GRL_WEIGHT="0.01"
        RHO="0.001"
        EXTRA_ARGS=(--grl_phoneme_weight 0.01 --ub_weight 0.01)
        ;;
    *)
        echo "ERROR: unsupported array task ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

STAGE1_CKPT="${DIS_DIR}/checkpoints/best.pt"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"
PROBE_RUN_NAME="diag_probe_${RUN_NAME}"

STAGE2_STEPS="${STAGE2_STEPS:-8000}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
SEED="${SEED:-42}"
TASKS="${TASKS:-pr,sid}"
SOURCES="${SOURCES:-z_t,z_L,z_P}"
PR_LABEL_SET="${PR_LABEL_SET:-superb}"
PR_PROBE_LR="${PR_PROBE_LR:-5e-4}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PROBE_WARMUP_STEPS="${PROBE_WARMUP_STEPS:-0}"
PR_MAX_EXAMPLES="${PR_MAX_EXAMPLES:-0}"

echo "=== Stage 2 + diagnostic probe: stronger GRL top-2 ==="
echo "Job ID             : ${SLURM_JOB_ID}"
echo "Array task         : ${SLURM_ARRAY_TASK_ID}"
echo "Node               : $(hostname)"
echo "GPU                : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started            : $(date)"
echo "base_name          : ${BASE_NAME}"
echo "run_name           : ${RUN_NAME}"
echo "stage1_ckpt        : ${STAGE1_CKPT}"
echo "stage2_ckpt        : ${STAGE2_CKPT}"
echo "strong_grl_heads   : true"
echo "alpha              : ${ALPHA}"
echo "beta               : ${BETA}"
echo "grl_weight         : ${GRL_WEIGHT}"
echo "rho                : ${RHO}"
echo "seed               : ${SEED}"
echo "stage2_steps       : ${STAGE2_STEPS}"
echo "probe_run_name     : ${PROBE_RUN_NAME}"
echo "probe_sources      : ${SOURCES}"
echo "probe_tasks        : ${TASKS}"
echo "probe_steps        : ${PROBE_STEPS}"
echo "pr_label_set       : ${PR_LABEL_SET}"
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
    --rho                "${RHO}" \
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
    "${EXTRA_ARGS[@]}"

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
    --pr_label_set "${PR_LABEL_SET}" \
    --pr_probe_lr "${PR_PROBE_LR}" \
    --sid_probe_lr "${SID_PROBE_LR}" \
    --probe_warmup_steps "${PROBE_WARMUP_STEPS}"

echo "Finished           : $(date)"
