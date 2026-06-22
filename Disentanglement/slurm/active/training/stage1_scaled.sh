#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=stage1_scaled
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage1/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage1/%x_%j.err

# Scaled stage-1 SAE toward the Gao monosemanticity regime, on local train-clean-360.
#   - bigger dictionary + sparser:  K ~16-20k, topk 64  (k/d_model ~5%, Gao's sweet spot)
#   - Gao dead-latent revival: AuxK + geometric-median bias + per-step decoder renorm
#   - LOCAL flac (no HF/CDN) — train-clean-360 (921 speakers) for the diversity that
#     forces phoneme/speaker atoms to factor (251 speakers was too few).
# Env overrides let this serve as both a smoke (small MAX_TRAIN/STEPS) and the full run.

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
mkdir -p "${DIS_DIR}/logs/train/stage1"
cd "${DIS_DIR}"

K="${K:-16384}"
TOPK="${TOPK:-64}"
AUX_K="${AUX_K:-512}"
STEPS="${STEPS:-20000}"
DEAD_THRESH="${DEAD_THRESH:-256}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train-clean-360}"
MAX_TRAIN="${MAX_TRAIN:-0}"      # 0 = full split; smoke sets small
MAX_VAL="${MAX_VAL:-500}"
RUN_NAME="${RUN_NAME:-scaled_K${K}_t${TOPK}}"
SEED="${SEED:-42}"

echo "=== Scaled stage-1 SAE: ${RUN_NAME} ===  $(date)"
echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "K=${K} topk=${TOPK} aux_k=${AUX_K} dead_thresh=${DEAD_THRESH} steps=${STEPS}"
echo "data: LOCAL ${TRAIN_SPLIT}  max_train=${MAX_TRAIN}"

${PYTHON} -u run.py \
    --stage 1 \
    --local_data --train_split_dir "${TRAIN_SPLIT}" \
    --K "${K}" --topk "${TOPK}" \
    --aux_k "${AUX_K}" --aux_k_coef 0.03125 --dead_steps_threshold "${DEAD_THRESH}" \
    --geom_median_bias --renorm_decoder \
    --spear_layernorm \
    --total_steps "${STEPS}" --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 \
    --max_train_examples "${MAX_TRAIN}" --max_val_examples "${MAX_VAL}" \
    --batch_size 16 --grad_log_every 500 \
    --checkpoint_dir "${DIS_DIR}/checkpoints/${RUN_NAME}" \
    --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" \
    --seed "${SEED}"

echo "Finished $(date)"
