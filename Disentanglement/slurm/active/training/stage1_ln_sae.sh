#!/bin/bash
#SBATCH --job-name=ln_sae
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage1/ln_sae_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage1/ln_sae_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU

# Stage-1 SAE retrain on SUPERB-comparable h_t: each SPEAR layer is LayerNorm'd
# (no affine) before averaging.  Required because --spear_layernorm changes h_t,
# so the existing SAE (checkpoints/best.pt) no longer matches the target.
# Writes to a SEPARATE dir (checkpoints/ln_sae) — does not touch existing runs.

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1

mkdir -p "${DIS_DIR}/logs/train/stage1"
cd "${DIS_DIR}"

echo "=== Stage 1 SAE — LayerNorm'd SPEAR layers (SUPERB-comparable h_t) ==="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"

${PYTHON} -u run.py \
    --stage              1    \
    --spear_layernorm         \
    --max_train_examples 0    \
    --max_val_examples   500  \
    --total_steps        6000 \
    --batch_size         16   \
    --K                  5120 \
    --topk               256  \
    --lr                 1e-4 \
    --lr_min             1e-6 \
    --warmup_steps       500  \
    --checkpoint_dir     "${DIS_DIR}/checkpoints/ln_sae" \
    --runs_dir           "${DIS_DIR}/runs/ln_sae"        \
    --log_dir            "${DIS_DIR}/logs"

echo "Finished : $(date)"
