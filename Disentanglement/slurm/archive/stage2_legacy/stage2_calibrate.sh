#!/bin/bash
#SBATCH --job-name=dis_calib
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/calib_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/calib_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU

# Calibration run: 500 steps with unit weights, dense grad norm logging.
# Read the grad_norms output to set alpha, beta, grl_weight for the full run.

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1

mkdir -p "${DIS_DIR}/logs"
cd "${DIS_DIR}"

echo "=== Stage 2 calibration — unit weights, 500 steps ==="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"

${PYTHON} -u run.py \
    --stage            2                                              \
    --stage1_ckpt      "${DIS_DIR}/checkpoints/stage1_best.pt"       \
    --stage2_steps     500                                            \
    --warmup_steps     50                                             \
    --grad_log_every   50                                             \
    --alpha            1.0                                            \
    --beta             1.0                                            \
    --grl_weight       1.0                                            \
    --rho              0.001                                          \
    --lr               3e-5                                           \
    --lr_routing       1e-4                                           \
    --lr_heads         1e-4                                           \
    --max_train_examples 0                                            \
    --max_val_examples   200                                          \
    --checkpoint_dir   "${DIS_DIR}/checkpoints"                       \
    --runs_dir         "${DIS_DIR}/runs"                              \
    --log_dir          "${DIS_DIR}/logs"

echo "Finished : $(date)"
echo ""
echo "=== Read the [grad_norms] lines above to calibrate alpha/beta/grl_weight ==="
