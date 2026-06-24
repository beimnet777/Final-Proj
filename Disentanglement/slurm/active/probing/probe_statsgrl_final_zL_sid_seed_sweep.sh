#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=stat_zL_ss
#SBATCH --array=0-1%1
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/statsgrl_final_zL_sid_seed_sweep/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/statsgrl_final_zL_sid_seed_sweep/%x_%A_%a.err

# Probe-seed sweep for the final checkpoint of job2_statsgrl_clip_gp02.
#
# Purpose:
#   Verify that the excellent final-checkpoint z_L speaker removal is not a
#   probe-seed accident before treating stats-GRL as the best non-invariance run.
#
# Important:
#   Seed 42 has already been run on this same final checkpoint:
#     logs/diag/final_step_selected4/probe_final4_31011803_3.out
#   This script runs two additional seeds only. Combined seed set: {42, 7, 123}.
#
# Probe:
#   source: z_L only
#   task:   SID only
#   head:   stats probe = projector -> ReLU -> masked mean+std pool -> linear
#   early stopping disabled (patience=0), so low leakage cannot be explained by
#   a probe that stopped too early.

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

mkdir -p "${DIS_DIR}/logs/diag/statsgrl_final_zL_sid_seed_sweep"
cd "${DIS_DIR}"

RUN_NAME="job2_statsgrl_clip_gp02"
FINAL_STEP=12000
CKPT="${DIS_DIR}/checkpoints/${RUN_NAME}/stage2_step${FINAL_STEP}.pt"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing final-step checkpoint ${CKPT}" >&2; exit 2; }

PROBE_SEEDS=(7 123)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#PROBE_SEEDS[@]} )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
    exit 3
fi
PROBE_SEED="${PROBE_SEEDS[${TASK_ID}]}"

BLOCKS=(--fixed_blocks --per_block_topk
        --K_L 3072 --K_P 1024 --K_U 1024
        --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== Stats-GRL final z_L SID probe-seed sweep ==="
echo "started          : $(date)"
echo "gpu              : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "run_name         : ${RUN_NAME}"
echo "ckpt             : ${CKPT}"
echo "known seed 42    : logs/diag/final_step_selected4/probe_final4_31011803_3.out"
echo "probe_seed       : ${PROBE_SEED}"
echo "sources          : z_L"
echo "task             : sid"
echo "sid_probe_arch   : stats"
echo "probe_steps      : 10000"
echo "probe_val_every  : 250"
echo "probe_patience   : 0  # disabled; run all probe steps"
echo "sid_probe_lr     : 1e-3"

${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${CKPT}" \
    --stage1_ckpt "${CKPT}" \
    --run_name "diag_${RUN_NAME}_final${FINAL_STEP}_zL_sid_noearly_seed${PROBE_SEED}" \
    "${BLOCKS[@]}" --spear_layernorm \
    --sources "z_L" --tasks "sid" --sid_probe_arch stats \
    --probe_steps 10000 --probe_val_every 250 --probe_patience 0 \
    --pr_max_examples 0 --sid_probe_lr 1e-3 \
    --probe_warmup_steps 0 --seed "${PROBE_SEED}"

echo "finished         : $(date)"
