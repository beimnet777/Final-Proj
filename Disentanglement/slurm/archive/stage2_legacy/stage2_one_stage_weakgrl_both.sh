#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:30:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=s2_one_stage_wg
#SBATCH --array=0-1
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/one_stage/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/one_stage/%x_%A_%a.err
#
# Direct submission:
#   sbatch slurm/stage2_one_stage_weakgrl_both.sh
#
# Array task 0: one_stage_weakgrl_x1  α=0.02 β=0.01 grl=0.01 ρ=0.001
# Array task 1: one_stage_weakgrl_x2  α=0.04 β=0.02 grl=0.02 ρ=0.001
#
# Each array task trains from scratch, then probes its own stage2_best.pt.

set -euo pipefail

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1

mkdir -p "${DIS_DIR}/logs/train/stage2/one_stage" "${DIS_DIR}/logs/probes/diagnostic_historical"
cd "${DIS_DIR}"

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        RUN_NAME=one_stage_weakgrl_x1
        ALPHA=0.02
        BETA=0.01
        GRL_WEIGHT=0.01
        ;;
    1)
        RUN_NAME=one_stage_weakgrl_x2
        ALPHA=0.04
        BETA=0.02
        GRL_WEIGHT=0.02
        ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

RHO=0.001
CKPT="${DIS_DIR}/checkpoints/${RUN_NAME}/stage2_best.pt"

echo "=== One-stage weak-GRL: ${RUN_NAME} ==="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Array task : ${SLURM_ARRAY_TASK_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"
echo "α=${ALPHA}  β=${BETA}  grl=${GRL_WEIGHT}  ρ=${RHO}"

${PYTHON} -u run.py \
    --stage             2                                              \
    --stage2_from_scratch                                             \
    --stage1_ckpt       "${DIS_DIR}/checkpoints/best.pt"              \
    --stage2_steps      8000                                           \
    --warmup_steps      500                                            \
    --alpha             "${ALPHA}"                                     \
    --beta              "${BETA}"                                      \
    --grl_weight        "${GRL_WEIGHT}"                                \
    --grl_delay_steps   0                                              \
    --rho               "${RHO}"                                       \
    --lr                3e-5                                           \
    --lr_min            1e-6                                           \
    --lr_routing        5e-6                                           \
    --lr_heads          1e-4                                           \
    --max_train_examples 0                                             \
    --max_val_examples   500                                           \
    --grad_log_every    500                                            \
    --checkpoint_dir    "${DIS_DIR}/checkpoints/${RUN_NAME}"           \
    --runs_dir          "${DIS_DIR}/runs/${RUN_NAME}"                  \
    --log_dir           "${DIS_DIR}/logs"

echo "=== Probe: ${RUN_NAME} ==="
echo "stage2_ckpt=${CKPT}"

${PYTHON} -u probe_runner.py \
    --stage1_ckpt  "${DIS_DIR}/checkpoints/best.pt" \
    --stage2_ckpt  "${CKPT}" \
    --run_name     "probe_${RUN_NAME}" \
    --probe_steps  2000

echo "Finished : $(date)"
