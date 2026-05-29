#!/bin/bash
#SBATCH --job-name=dis_stage2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/stage2_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/stage2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU

# Stage 2: routing + SID + CTC + GRL, no Barlow Twins

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1

STAGE1_CKPT="${STAGE1_CKPT:-${DIS_DIR}/checkpoints/stage1_best.pt}"

mkdir -p "${DIS_DIR}/logs"

cd "${DIS_DIR}"

echo "=== Stage 2: routing + SID + CTC + GRL ==="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Stage-1 ckpt : ${STAGE1_CKPT}"
echo "Started      : $(date)"

${PYTHON} -u run.py \
    --stage 2 \
    --stage1_ckpt        "${STAGE1_CKPT}"          \
    --max_train_examples 15000                      \
    --max_val_examples   500                        \
    --stage2_steps       40000                      \
    --batch_size         16                         \
    --K                  5120                       \
    --topk               256                        \
    --alpha              1.0                        \
    --beta               0.2                        \
    --rho                0.0001                     \
    --checkpoint_dir     "${DIS_DIR}/checkpoints"   \
    --runs_dir           "${DIS_DIR}/runs"           \
    --log_dir            "${DIS_DIR}/logs"

echo "Finished : $(date)"
