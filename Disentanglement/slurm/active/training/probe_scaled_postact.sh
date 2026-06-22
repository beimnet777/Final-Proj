#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=probe_scpostact
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/scaled_postact_probe/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/scaled_postact_probe/%x_%j.err

# Probe scaled_postact (K_L/P/U 9830/3277/3277, topk 40/16/8) — the run whose
# training grl stayed high.  z_L,z_P,z_U each for PR + SID (stats probe, high
# ceiling).  This is the TRUTH the training grl can't give us.

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null; module load rhel8/default-amp 2>/dev/null
PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"
mkdir -p "${DIS_DIR}/logs/diag/scaled_postact_probe"
cd "${DIS_DIR}"
CKPT="${DIS_DIR}/checkpoints/scaled_postact/stage2_best.pt"
BLOCKS=(--fixed_blocks --per_block_topk --K_L 9830 --K_P 3277 --K_U 3277 --topk_L 40 --topk_P 16 --topk_U 8)

[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}" >&2; exit 2; }
echo "=== probe scaled_postact  z_L,z_P,z_U x {pr,sid} ===  $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${CKPT}" --stage1_ckpt "${CKPT}" \
    --run_name "diag_scaled_postact" \
    "${BLOCKS[@]}" --topk 64 --spear_layernorm \
    --sources "z_L,z_P,z_U" --tasks "pr,sid" --sid_probe_arch stats \
    --probe_steps 8000 --probe_val_every 250 --probe_patience 8 \
    --max_train_examples 12000 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 --probe_warmup_steps 0 \
    --seed 42
echo; echo "Finished $(date)"
