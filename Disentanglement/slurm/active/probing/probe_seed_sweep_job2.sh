#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=probe_seed_sweep
#SBATCH --array=0-2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/probe_seed_sweep/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/probe_seed_sweep/%x_%A_%a.err

# Probe-seed sweep on the existing Job 2 checkpoint.
# Holds the trained model fixed and varies ONLY the diagnostic-probe seed in
#   {7, 42, 123}.
# NOTE: --array=0-2 with no `%N` throttle → all 3 tasks scheduled CONCURRENTLY.
# Wall clock = 1 job's runtime (~6 h max), not 3 × 6 h.  Each task takes one
# A100 GPU.
# Resolves the probe-seed × train-seed × probe-budget contamination flagged in
# "Pending-Sweep Analysis (June 23 2026)": at train_seed=42, the shuffled-speaker
# control gave the same z_L SID as the trained baseline, so we cannot tell
# whether 0.010 is a property of the representation or of probe_seed=42.
# Expected (large-effect) finding: if ANY probe seed recovers z_L SID > 0.05,
# the single-number Job 2 headline must be replaced by a (train_seed, probe_seed)
# distribution.  Cost: 3 diagnostic probes, no training, run in parallel.

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
mkdir -p "${DIS_DIR}/logs/diag/probe_seed_sweep"
cd "${DIS_DIR}"

RUN_NAME="job2_dense_gradnorm"
CKPT="${DIS_DIR}/checkpoints/${RUN_NAME}/stage2_best.pt"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}" >&2; exit 2; }

# Probe seeds: 42 = the baseline ("known anomalous low"), 7 + 123 = two diverse
# fresh seeds (7 is also a train-side seed → train_seed=42 × probe_seed=7 lets
# us cross-check the train_seed=7 × probe_seed=42 cell that scored z_L SID=0.704).
PROBE_SEEDS=(7 42 123)
PROBE_SEED="${PROBE_SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"

BLOCKS=(--fixed_blocks --per_block_topk \
        --K_L 3072 --K_P 1024 --K_U 1024 \
        --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== Probe-seed sweep on ${RUN_NAME}: probe_seed=${PROBE_SEED} ===  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${CKPT}" --stage1_ckpt "${CKPT}" \
    --run_name "diag_${RUN_NAME}_probe_seed${PROBE_SEED}" \
    "${BLOCKS[@]}" --spear_layernorm \
    --sources "z_L,z_P" --tasks "pr,sid" --sid_probe_arch stats \
    --probe_steps 10000 --probe_val_every 250 --probe_patience 0 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 \
    --mdl_probe --mdl_steps_per_block 1250 --mdl_max_train_examples 4000 \
    --seed "${PROBE_SEED}"

echo "Finished probe_seed=${PROBE_SEED}: $(date)"
