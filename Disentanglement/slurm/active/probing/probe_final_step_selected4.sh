#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=probe_final4
#SBATCH --array=0-3%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/final_step_selected4/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/final_step_selected4/%x_%A_%a.err

# Probe the final saved step for selected runs whose stage2_best.pt may not be
# the right checkpoint for judging z_L speaker leakage.
#
# Default probe is z_L -> SID only, no early stopping, because this is the
# specific failure mode under investigation. Override SOURCES/TASKS from sbatch
# if a broader diagnostic is needed.

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

mkdir -p "${DIS_DIR}/logs/diag/final_step_selected4"
cd "${DIS_DIR}"

RUN_NAMES=(
    "job2_dense_gradnorm"
    "job2_dense_gradnorm_cheng_recon_pr_sid"
    "job2_dense_clip_gp02"
    "job2_statsgrl_clip_gp02"
)
LABELS=(
    "dense_statgrl_job2_dense_gradnorm"
    "cheng_recon"
    "job2_dense_clip"
    "job2_statsgrl_clip"
)
FINAL_STEPS=(12000 12000 12000 12000)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#RUN_NAMES[@]} )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
    exit 2
fi

RUN_NAME="${RUN_NAMES[${TASK_ID}]}"
LABEL="${LABELS[${TASK_ID}]}"
FINAL_STEP="${FINAL_STEPS[${TASK_ID}]}"
CKPT="${DIS_DIR}/checkpoints/${RUN_NAME}/stage2_step${FINAL_STEP}.pt"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing final-step checkpoint ${CKPT}" >&2; exit 3; }

SOURCES="${SOURCES:-z_L}"
TASKS="${TASKS:-sid}"
PROBE_SEED="${PROBE_SEED:-42}"
PROBE_STEPS="${PROBE_STEPS:-10000}"
PROBE_PATIENCE="${PROBE_PATIENCE:-0}"

BLOCKS=(--fixed_blocks --per_block_topk
        --K_L 3072 --K_P 1024 --K_U 1024
        --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== Final-step diagnostic probe ==="
echo "started          : $(date)"
echo "gpu              : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "label            : ${LABEL}"
echo "run_name         : ${RUN_NAME}"
echo "ckpt             : ${CKPT}"
echo "sources          : ${SOURCES}"
echo "tasks            : ${TASKS}"
echo "sid_probe_arch   : stats"
echo "probe_seed       : ${PROBE_SEED}"
echo "probe_steps      : ${PROBE_STEPS}"
echo "probe_patience   : ${PROBE_PATIENCE}"

${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${CKPT}" \
    --stage1_ckpt "${CKPT}" \
    --run_name "diag_finalstep_${LABEL}_step${FINAL_STEP}_seed${PROBE_SEED}" \
    "${BLOCKS[@]}" --spear_layernorm \
    --sources "${SOURCES}" --tasks "${TASKS}" --sid_probe_arch stats \
    --probe_steps "${PROBE_STEPS}" --probe_val_every 250 \
    --probe_patience "${PROBE_PATIENCE}" \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 \
    --probe_warmup_steps 0 --seed "${PROBE_SEED}"

echo "finished         : $(date)"
