#!/bin/bash
#SBATCH --job-name=dis_s1_B
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/stage1_runB_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/stage1_runB_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU

# Run B: topk=128 (2.5%), delta=0.001, 15k utterances, 50k steps

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1

mkdir -p "${DIS_DIR}/logs"

cd "${DIS_DIR}"

echo "=== Stage 1 Run B: topk=128, delta=0.001, 15k utts, 50k steps ==="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"

${PYTHON} -u run.py \
    --stage 1 \
    --max_train_examples 15000 \
    --max_val_examples   500   \
    --stage1_steps       50000 \
    --batch_size         16    \
    --topk               128   \
    --delta              0.001 \
    --checkpoint_dir     "${DIS_DIR}/checkpoints/runB" \
    --runs_dir           "${DIS_DIR}/runs/runB"        \
    --log_dir            "${DIS_DIR}/logs"

echo "Finished : $(date)"
