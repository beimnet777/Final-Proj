#!/bin/bash
#SBATCH --job-name=dis_old
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/old/logs/run_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/old/logs/run_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU

# Old+Gao: Gao et al. SAE + (D,4K) routed decoder + single-stage training
# delta=1e-5, rho=0.0001, topk=256, 15k utterances, 50k steps

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
OLD_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/old
export PYTHONUNBUFFERED=1

mkdir -p "${OLD_DIR}/logs"

cd "${OLD_DIR}"

echo "=== Old+Gao Disentanglement — single stage ==="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"

${PYTHON} -u run.py \
    --max_train_examples 15000 \
    --max_val_examples   500   \
    --total_steps        50000 \
    --batch_size         16    \
    --topk               256   \
    --delta              1e-5  \
    --rho                1e-4  \
    --beta               0.2   \
    --checkpoint_dir     "${OLD_DIR}/checkpoints" \
    --runs_dir           "${OLD_DIR}/runs"         \
    --log_dir            "${OLD_DIR}/logs"

echo "Finished : $(date)"
