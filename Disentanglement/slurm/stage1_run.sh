#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#
# Generic stage-1 worker — called by stage1_sweep.sh.
# Parameters injected via --export: RUN_NAME EXTRA_ARGS

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1

mkdir -p "${DIS_DIR}/logs/stage1"
cd "${DIS_DIR}"

echo "=== Stage 1: ${RUN_NAME} ==="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started : $(date)"
echo "extra   : ${EXTRA_ARGS:-none}"

${PYTHON} -u run.py \
    --stage             1                                        \
    --total_steps       6000                                     \
    --batch_size        16                                       \
    --lr                1e-4                                     \
    --lr_min            1e-6                                     \
    --warmup_steps      500                                      \
    --max_train_examples 0                                       \
    --max_val_examples   500                                     \
    --checkpoint_dir    "${DIS_DIR}/checkpoints/${RUN_NAME}"     \
    --runs_dir          "${DIS_DIR}/runs/${RUN_NAME}"            \
    --log_dir           "${DIS_DIR}/logs/stage1"                 \
    ${EXTRA_ARGS:-}

echo "Finished : $(date)"
