#!/bin/bash
#
# Step 2 of 2: Train the final-layer linear probe on pre-extracted SPEAR
# features.  The 596M-param encoder is NOT loaded — only the 29-dim linear
# classifier trains.  10 epochs should complete in < 30 min.
#
# Requires: feature_cache/{train,val,test}.pt produced by extract_features_final.slurm
#

#SBATCH -J spear_probe_final_cached
#SBATCH -A MLMI-bbg25-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /home/${USER}/.bashrc || true

set -euo pipefail
source /rds/project/rds-xyBFuSj0hm0/MLMI6.L2026/miniconda3/bin/activate
conda activate mlmi4

WORKDIR="$(cd "${SLURM_SUBMIT_DIR}" && [[ "$(basename "$PWD")" == scripts ]] && cd .. ; pwd)"
cd "${WORKDIR}"
mkdir -p logs
export OMP_NUM_THREADS=1
SEED="${SEED:-42}"
export PYTHONHASHSEED="${SEED}"

JOBID="${SLURM_JOB_ID}"
EXPERIMENT_NAME="spear_probe_final_cached_librispeech100h"
LOG_FILE="logs/out.${EXPERIMENT_NAME}.${JOBID}"

echo "JobID:        ${JOBID}"
echo "Experiment:   ${EXPERIMENT_NAME}"
echo "Time:         $(date)"
echo "Node:         $(hostname)"
echo "Working dir:  ${WORKDIR}"
echo "Log file:     ${LOG_FILE}"
echo "Seed:         ${SEED}"
echo "Running command: python -u run.py --probe final --feature_cache_dir ./feature_cache ..."

python -u run.py \
  --probe              final \
  --feature_cache_dir  ./feature_cache \
  --train_hours        100 \
  --batch_size         64 \
  --eval_batch_size    128 \
  --epochs             10 \
  --lr                 5e-4 \
  --warmup_steps       500 \
  --seed               "${SEED}" \
  --runs_dir           ./runs \
  2>&1 | tee "${LOG_FILE}"
