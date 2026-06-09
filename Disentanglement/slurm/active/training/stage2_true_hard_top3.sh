#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=true_hard_t3
#SBATCH --array=0-2%1
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/true_hard/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/true_hard/%x_%A_%a.err

# True hard-routing top-3 rerun:
#   1. Train stage 2 with one-hot ST-Gumbel routing enabled.
#   2. Probe the freshly trained checkpoint with corrected diagnostic probing.
#
# Array tasks:
#   0: true_hard_sid1_weakgrl
#   1: true_hard_dual_weak_ub
#   2: true_hard_ste

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

mkdir -p "${DIS_DIR}/logs/train/stage2/true_hard"
mkdir -p "${DIS_DIR}/logs/probes/true_hard_top3"
cd "${DIS_DIR}"

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        BASE_NAME="sid1_weakgrl"
        RUN_NAME="true_hard_sid1_weakgrl"
        ALPHA="0.02"
        BETA="0.01"
        GRL_WEIGHT="0.01"
        EXTRA_ARGS=()
        ;;
    1)
        BASE_NAME="dual_weak_ub"
        RUN_NAME="true_hard_dual_weak_ub"
        ALPHA="0.02"
        BETA="0.01"
        GRL_WEIGHT="0.01"
        EXTRA_ARGS=(--grl_phoneme_weight 0.01 --ub_weight 0.01)
        ;;
    2)
        BASE_NAME="ste"
        RUN_NAME="true_hard_ste"
        ALPHA="0.02"
        BETA="0.01"
        GRL_WEIGHT="0.01"
        EXTRA_ARGS=(--ste_routing)
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
PR_PROBE_LR="${PR_PROBE_LR:-5e-4}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PROBE_WARMUP_STEPS="${PROBE_WARMUP_STEPS:-0}"
PR_MAX_EXAMPLES="${PR_MAX_EXAMPLES:-0}"

echo "=== Stage 2 + diagnostic probe: true hard routing top-3 ==="
echo "Job ID             : ${SLURM_JOB_ID}"
echo "Array task         : ${SLURM_ARRAY_TASK_ID}"
echo "Node               : $(hostname)"
echo "GPU                : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started            : $(date)"
echo "base_name          : ${BASE_NAME}"
echo "run_name           : ${RUN_NAME}"
echo "stage1_ckpt        : ${STAGE1_CKPT}"
echo "stage2_ckpt        : ${STAGE2_CKPT}"
echo "hard_gumbel        : true"
echo "rho                : 0"
echo "alpha              : ${ALPHA}"
echo "beta               : ${BETA}"
echo "grl_weight         : ${GRL_WEIGHT}"
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
    --hard_gumbel_routing \
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
    --pr_probe_lr "${PR_PROBE_LR}" \
    --sid_probe_lr "${SID_PROBE_LR}" \
    --probe_warmup_steps "${PROBE_WARMUP_STEPS}"

echo "Finished           : $(date)"
