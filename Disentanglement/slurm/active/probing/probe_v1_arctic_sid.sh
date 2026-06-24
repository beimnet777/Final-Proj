#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=probe_v1_arctic
#SBATCH --array=0-1
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/v1_arctic_sid/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/v1_arctic_sid/%x_%A_%a.err

# Matched-distribution z_L SID probe for dual_inv_v1_soft_nogrl.
#
# Probes v1's stage2_step12000.pt against ARCTIC 18-speaker pool (the same
# voices the cross-speaker pair-α was trained on). If invariance generalised
# on its training distribution, accuracy should be near chance (1/18 ≈ 5.6%).
# If it's high, the mechanism failed even in-distribution.
#
# Two seeds (42, 7) — controls per-speaker random-split variability and probe
# init noise. Each task is independent; --array=0-1 runs both in parallel.

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

mkdir -p "${DIS_DIR}/logs/diag/v1_arctic_sid"
cd "${DIS_DIR}"

RUN_NAME="dual_inv_v1_soft_nogrl"
CKPT="${DIS_DIR}/checkpoints/${RUN_NAME}/stage2_step12000.pt"
[[ -f "${CKPT}" ]] || { echo "ERROR: ${CKPT} missing — needs the explicit step-12000 ckpt, not stage2_best.pt" >&2; exit 3; }

ARCTIC_ROOT="${DIS_DIR}/../Probing/data/CMU_ARCTIC"

SEEDS=(42 7)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[${TASK_ID}]}"

echo "=== Matched-distribution ARCTIC SID probe — v1 soft_nogrl ==="
echo "started        : $(date)"
echo "gpu            : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "ckpt           : ${CKPT}"
echo "arctic_root    : ${ARCTIC_ROOT}"
echo "sources        : z_L, z_P"
echo "tasks          : sid"
echo "sid_probe_arch : stats"
echo "arctic_sid_seed: ${SEED}"
echo "probe_seed     : 42  (probe init, same across array tasks)"
echo "probe_patience : 0   (no early stop — honest leakage protocol)"
echo "probe_steps    : 10000"

${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${CKPT}" \
    --stage1_ckpt "${CKPT}" \
    --run_name "diag_v1_soft_nogrl_arctic_sid_split${SEED}" \
    --n_routes 2 --no-hard_gumbel_routing --gumbel_tau_end 1.0 \
    --spear_layernorm \
    --sources "z_L,z_P" --tasks "sid" \
    --sid_dataset arctic --arctic_root "${ARCTIC_ROOT}" --arctic_sid_seed ${SEED} \
    --sid_probe_arch stats \
    --probe_steps 10000 --probe_val_every 250 --probe_patience 0 \
    --sid_probe_lr 1e-3 --probe_warmup_steps 0 --seed 42

echo "finished       : $(date)"
