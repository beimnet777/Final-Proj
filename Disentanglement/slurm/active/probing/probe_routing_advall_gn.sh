#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=probe_advall_gn
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/routing_advall_probe/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/routing_advall_probe/%x_%j.err

# Frozen-feature probe of routing_advall_gn (train-only run had NO probe).
# Reads z_L / z_P (+ z_t ceiling) for PR (SUPERB 74-phone) and SID (stats probe).
# Routing flags MUST match training so the forward carves z_L/z_P identically:
#   topk=128 (NOT auto-detected — config default is 256), hard Gumbel, tau_end 0.5,
#   spear_layernorm.  K and n_routes are auto-detected from the checkpoint.
# SID uses the STATS probe (the honest readout; a linear/mean-pool probe can be
# fooled — see the instance-norm reprobe).  Speaker chance = 1/251.

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
mkdir -p "${DIS_DIR}/logs/diag/routing_advall_probe"
cd "${DIS_DIR}"

STAGE1_CKPT="${DIS_DIR}/checkpoints/ln_sae/stage1_best.pt"
STAGE2_CKPT="${DIS_DIR}/checkpoints/routing_advall_gn/stage2_best.pt"
[[ -f "${STAGE1_CKPT}" ]] || { echo "ERROR: missing ${STAGE1_CKPT}" >&2; exit 2; }
[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: missing ${STAGE2_CKPT}" >&2; exit 2; }

echo "=== Probe routing_advall_gn (z_t,z_L,z_P : PR + SID-stats) ===  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

${PYTHON} -u diag_probe/run.py \
    --stage1_ckpt        "${STAGE1_CKPT}" \
    --stage2_ckpt        "${STAGE2_CKPT}" \
    --run_name           diag_probe_routing_advall_gn \
    --spear_layernorm \
    --topk               128 \
    --hard_gumbel_routing \
    --gumbel_tau_end     0.5 \
    --sources            "z_L,z_P" \
    --tasks              "pr,sid" \
    --sid_probe_arch     stats \
    --probe_steps        4000 \
    --probe_val_every    250 \
    --probe_patience     5 \
    --pr_probe_lr        5e-4 \
    --sid_probe_lr       1e-3 \
    --probe_warmup_steps 0 \
    --pr_max_examples    0 \
    --seed               42

echo "Finished probe: $(date)"
