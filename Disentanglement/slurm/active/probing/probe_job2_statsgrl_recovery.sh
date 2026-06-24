#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=probe_statsgrl
#SBATCH --array=0-1
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/statsgrl_recovery/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/statsgrl_recovery/%x_%A_%a.err

# Probe-only recovery for job2_statsgrl_clip_gp02 ("cha's way").
# Training completed and wrote stage2_best.pt; this script reuses that checkpoint
# and splits SID/PR across array tasks so a PR data/loading failure does not
# block the SID leakage result.

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

mkdir -p "${DIS_DIR}/logs/diag/statsgrl_recovery"
cd "${DIS_DIR}"

RUN_NAME="job2_statsgrl_clip_gp02"
CKPT="${DIS_DIR}/checkpoints/${RUN_NAME}/stage2_best.pt"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing checkpoint ${CKPT}" >&2; exit 2; }

TASKS=(sid pr)
TASK="${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}"

BLOCKS=(--fixed_blocks --per_block_topk \
        --K_L 3072 --K_P 1024 --K_U 1024 \
        --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== Probe-only recovery: ${RUN_NAME}, task=${TASK} ==="
echo "started : $(date)"
echo "gpu     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "ckpt    : ${CKPT}"
echo "sources : z_L,z_P"
echo "task    : ${TASK}"

${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${CKPT}" \
    --stage1_ckpt "${CKPT}" \
    --run_name "diag_${RUN_NAME}_recovery_${TASK}" \
    "${BLOCKS[@]}" --spear_layernorm \
    --sources "z_L,z_P" --tasks "${TASK}" --sid_probe_arch stats \
    --probe_steps 10000 --probe_val_every 250 --probe_patience 8 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 \
    --probe_warmup_steps 0 --seed 42

echo "Finished task=${TASK}: $(date)"
