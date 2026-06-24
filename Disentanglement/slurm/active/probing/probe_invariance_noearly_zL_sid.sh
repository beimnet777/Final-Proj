#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=probe_inv_noes
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/invariance_noearly/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/invariance_noearly/%x_%j.err

# Probe-only recovery for invariance_only_w4_noramp.
# Purpose: verify whether the old z_L SID ~= 0.01 result was genuine leakage
# removal or an early-stopping artefact. This probes ONLY z_L -> SID for the
# full 10k updates with early stopping disabled.

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"

mkdir -p "${DIS_DIR}/logs/diag/invariance_noearly"
cd "${DIS_DIR}"

RUN_NAME="invariance_only_w4_noramp"
CKPT="${DIS_DIR}/checkpoints/${RUN_NAME}/stage2_best.pt"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing checkpoint ${CKPT}" >&2; exit 2; }

BLOCKS=(--fixed_blocks --per_block_topk \
        --K_L 3072 --K_P 1024 --K_U 1024 \
        --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== No-early-stop z_L SID diagnostic: ${RUN_NAME} ==="
echo "started          : $(date)"
echo "gpu              : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "ckpt             : ${CKPT}"
echo "sources          : z_L"
echo "task             : sid"
echo "sid_probe_arch   : stats"
echo "seed             : 42"
echo "probe_steps      : 10000"
echo "probe_val_every  : 250"
echo "probe_patience   : 0  # disabled; run all probe steps"
echo "sid_probe_lr     : 1e-3"

${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${CKPT}" \
    --stage1_ckpt "${CKPT}" \
    --run_name "diag_${RUN_NAME}_zL_sid_noearly_seed42" \
    "${BLOCKS[@]}" --spear_layernorm \
    --sources "z_L" --tasks "sid" --sid_probe_arch stats \
    --probe_steps 10000 --probe_val_every 250 --probe_patience 0 \
    --pr_max_examples 0 --sid_probe_lr 1e-3 \
    --probe_warmup_steps 0 --seed 42

echo "finished         : $(date)"
