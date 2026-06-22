#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=probe_zl_sid
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/zl_sid/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/zl_sid/%x_%j.err
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
BLOCKS="--fixed_blocks --per_block_topk --K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32 --topk 256 --spear_layernorm"

for pair in "case3_attn_w2.0:checkpoints/grl_attn_w2.0/stage2_best.pt" \
            "dense_k31_w2.0:checkpoints/grl_dense_k31_w2.0/stage2_best.pt"; do
  NAME="${pair%%:*}"; CK="${pair##*:}"
  echo; echo "######## z_L->SID  ${NAME}  ($CK) ########"; date
  ${PYTHON} -u diag_probe/run.py \
      --stage2_ckpt "$CK" --stage1_ckpt "$CK" \
      --run_name "zLsid_${NAME}" \
      ${BLOCKS} \
      --sources "z_L" --tasks "sid" --sid_probe_arch stats \
      --probe_steps 10000 --probe_val_every 250 --probe_patience 8 \
      --sid_probe_lr 1e-3 --probe_warmup_steps 0 --seed 42
done
echo "Finished: $(date)"
