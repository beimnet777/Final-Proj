#!/bin/bash
#SBATCH --job-name=dis_stage2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/stage2_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/stage2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU

# Stage 2: full disentanglement — calibrated weights from calib_29865795
#   α=0.02  β=0.05  grl=0.04  ρ=0.001  — 8000 steps (~4.5 epochs)

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1

mkdir -p "${DIS_DIR}/logs"
cd "${DIS_DIR}"

echo "=== Stage 2: full disentanglement ==="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"

${PYTHON} -u run.py \
    --stage            2                                            \
    --stage1_ckpt      "${DIS_DIR}/checkpoints/best.pt"     \
    --stage2_steps     8000                                         \
    --warmup_steps     500                                          \
    --alpha            0.02                                         \
    --beta             0.003                                         \
    --grl_weight       0.04                                         \
    --rho              0.001                                        \
    --lr               3e-5                                         \
    --lr_min           1e-6                                         \
    --lr_routing       5e-6                                         \
    --lr_heads         1e-4                                         \
    --max_train_examples 0                                          \
    --max_val_examples   500                                        \
    --grad_log_every   500                                          \
    --checkpoint_dir   "${DIS_DIR}/checkpoints"                     \
    --runs_dir         "${DIS_DIR}/runs"                            \
    --log_dir          "${DIS_DIR}/logs"

echo "Finished : $(date)"
