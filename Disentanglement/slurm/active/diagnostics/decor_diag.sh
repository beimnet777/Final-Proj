#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diagnostics/decor/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diagnostics/decor/%x_%j.err
#
# Recompute SAE feature decorrelation post-hoc for baseline vs decor_only.

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1

mkdir -p "${DIS_DIR}/logs/diagnostics/decor"
cd "${DIS_DIR}"

echo "=== decor diagnostic ==="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started : $(date)"

${PYTHON} -u tools/diagnostics/decor_diagnostic.py \
    --ckpt baseline=${DIS_DIR}/checkpoints/best.pt \
    --ckpt decor_only=${DIS_DIR}/checkpoints/decor_only/stage1_best.pt

echo "Finished : $(date)"
