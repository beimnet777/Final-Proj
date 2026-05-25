#!/bin/bash
#
# SPEAR-XL probing: 2-layer BLSTM (1024 units/dir) + SpecAugment + CTC
# Trained on full LibriSpeech train-clean-100 (~100 h). Encoder frozen.
#

#SBATCH -J spear_probe_lstm
#SBATCH -A MLMI-bbg25-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
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

JOBID="${SLURM_JOB_ID}"
EXPERIMENT_NAME="spear_probe_lstm_librispeech100h"
LOG_FILE="logs/out.${EXPERIMENT_NAME}.${JOBID}"

echo "JobID:        ${JOBID}"
echo "Experiment:   ${EXPERIMENT_NAME}"
echo "Time:         $(date)"
echo "Node:         $(hostname)"
echo "Working dir:  ${WORKDIR}"
echo "Log file:     ${LOG_FILE}"

CMD=(
  python -u run.py
  --probe        lstm
  --train_hours  100
  --batch_size   8
  --eval_batch_size 16
  --epochs       10
  --lr           5e-4
  --warmup_steps 500
  --lstm_hidden  1024
  --lstm_layers  2
  --runs_dir     ./runs
)

echo "Running command: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
