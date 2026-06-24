#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=inv_zL_sid_ss
#SBATCH --array=0-3%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/invariance_zL_sid_seed_sweep/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/invariance_zL_sid_seed_sweep/%x_%A_%a.err

# Probe-only seed sweep for the two invariance checkpoints.
# This holds the trained representation fixed and varies ONLY the diagnostic
# probe seed. It probes z_L -> SID only, with early stopping disabled, so PR
# cannot consume RNG state before the SID probe.
#
# Seed 42 has already been run for both checkpoints, so this script runs only
# two additional seeds. Combined result per model: seeds {42, 7, 123}.

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

mkdir -p "${DIS_DIR}/logs/diag/invariance_zL_sid_seed_sweep"
cd "${DIS_DIR}"

MODELS=(
    "invariance_only_w4_noramp"
    "job1_inv_dense"
)
MODEL_LABELS=(
    "invariance_only"
    "invariance_dense"
)
PROBE_SEEDS=(7 123)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_IDX=$((TASK_ID / ${#PROBE_SEEDS[@]}))
SEED_IDX=$((TASK_ID % ${#PROBE_SEEDS[@]}))

if (( MODEL_IDX < 0 || MODEL_IDX >= ${#MODELS[@]} )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
    exit 2
fi

RUN_NAME="${MODELS[${MODEL_IDX}]}"
MODEL_LABEL="${MODEL_LABELS[${MODEL_IDX}]}"
PROBE_SEED="${PROBE_SEEDS[${SEED_IDX}]}"
CKPT="${DIS_DIR}/checkpoints/${RUN_NAME}/stage2_best.pt"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing checkpoint ${CKPT}" >&2; exit 3; }

BLOCKS=(--fixed_blocks --per_block_topk
        --K_L 3072 --K_P 1024 --K_U 1024
        --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== Invariance z_L SID probe-seed sweep ==="
echo "started          : $(date)"
echo "gpu              : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "model            : ${RUN_NAME}"
echo "model_label      : ${MODEL_LABEL}"
echo "ckpt             : ${CKPT}"
echo "sources          : z_L"
echo "task             : sid"
echo "sid_probe_arch   : stats"
echo "probe_seed       : ${PROBE_SEED}"
echo "probe_steps      : 10000"
echo "probe_val_every  : 250"
echo "probe_patience   : 0  # disabled; run all probe steps"
echo "sid_probe_lr     : 1e-3"

${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${CKPT}" \
    --stage1_ckpt "${CKPT}" \
    --run_name "diag_${RUN_NAME}_zL_sid_noearly_seed${PROBE_SEED}" \
    "${BLOCKS[@]}" --spear_layernorm \
    --sources "z_L" --tasks "sid" --sid_probe_arch stats \
    --probe_steps 10000 --probe_val_every 250 --probe_patience 0 \
    --pr_max_examples 0 --sid_probe_lr 1e-3 \
    --probe_warmup_steps 0 --seed "${PROBE_SEED}"

echo "finished         : $(date)"
