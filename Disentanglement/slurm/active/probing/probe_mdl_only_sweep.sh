#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=mdl_only_sweep
#SBATCH --array=0-3
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/mdl_only_sweep/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/mdl_only_sweep/%x_%A_%a.err

# MDL-only probe across the 4 high-value checkpoints from the June 23 pending
# sweep.  Skips the 10k-step standard accuracy probe (those numbers already
# live in the original training logs) and runs ONLY the Voita-&-Titov-2020
# prequential MDL codelength, which is probe-budget-free by construction.
#
# Targets (rationale in "Pending-Sweep Analysis - June 23 2026"):
#   0  job2_dense_gradnorm_seed7                  z_L SID = 0.704 outlier  → does MDL agree info is there?
#   1  job2_dense_gradnorm_seed21                 z_L SID = 0.452 mid-case → calibrate train-seed variance
#   2  job2_dense_gradnorm_seed84                 z_L SID = 0.006 "clean"  → triangulate against seed 42
#   3  job2_dense_gradnorm_cheng_recon_pr_sid     z_L SID = 0.600 GradNorm regression → MDL-robust?
#
# Skipped (low value): dense_gn_shuf, gn_nodense, inv_only_nr — those results
# are explained by the val-probe vs. final-probe gap; MDL would just confirm.
#
# NOTE: --array=0-3 with no `%N` throttle → all 4 tasks scheduled CONCURRENTLY.
# Wall clock = ~1 task's runtime (~1.5 h), total GPU-h ~6.  probe_seed=42 is
# held constant to match the existing cross-checkpoint comparisons; the
# parallel probe_seed_sweep_job2.sh handles probe-seed variance separately.

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
mkdir -p "${DIS_DIR}/logs/diag/mdl_only_sweep"
cd "${DIS_DIR}"

RUN_NAMES=(
  job2_dense_gradnorm_seed7
  job2_dense_gradnorm_seed21
  job2_dense_gradnorm_seed84
  job2_dense_gradnorm_cheng_recon_pr_sid
)
RUN_NAME="${RUN_NAMES[${SLURM_ARRAY_TASK_ID:-0}]}"
CKPT="${DIS_DIR}/checkpoints/${RUN_NAME}/stage2_best.pt"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}" >&2; exit 2; }

BLOCKS=(--fixed_blocks --per_block_topk \
        --K_L 3072 --K_P 1024 --K_U 1024 \
        --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== MDL-only probe on ${RUN_NAME} ===  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${CKPT}" --stage1_ckpt "${CKPT}" \
    --run_name "mdl_${RUN_NAME}" \
    "${BLOCKS[@]}" --spear_layernorm \
    --sources "z_L,z_P" --tasks "pr,sid" --sid_probe_arch stats \
    --mdl_only --mdl_steps_per_block 500 --mdl_max_train_examples 4000 \
    --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 \
    --seed 42

echo "Finished MDL-only probe ${RUN_NAME}: $(date)"
