#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=grad_rank
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/grad_conflict/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/grad_conflict/%x_%j.err
set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null; module load rhel8/default-amp 2>/dev/null
PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"
cd "${DIS_DIR}"
COMMON="--K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32 --topk 256 --n_batches 8"

echo "############## POOLED (attention) speaker head -- grl_attn_w2.0 ##############"
${PYTHON} -u diag_probe/grad_rank.py --ckpt checkpoints/grl_attn_w2.0/stage2_best.pt --grl_attention_pool ${COMMON}

echo ""
echo "############## DENSE (per-frame, k31) speaker head -- grl_dense_k31_w2.0 ##############"
${PYTHON} -u diag_probe/grad_rank.py --ckpt checkpoints/grl_dense_k31_w2.0/stage2_best.pt --grl_dense_context --grl_context_kernel 31 ${COMMON}
