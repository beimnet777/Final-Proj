#!/bin/bash
#
# SPEAR-XL probing: final-layer-only linear probe, trained on full LibriSpeech
# train-clean-100 (~100 h). Encoder frozen; only the linear classifier learns.
#

#SBATCH -J spear_probe_final
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

# Resolve the project root (Probing/) from this script's location, so the job
# behaves the same whether you sbatch from Probing/ or from scripts/.
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${WORKDIR}"
mkdir -p logs
export OMP_NUM_THREADS=1

JOBID="${SLURM_JOB_ID}"
EXPERIMENT_NAME="spear_probe_final_librispeech100h"
LOG_FILE="logs/out.${EXPERIMENT_NAME}.${JOBID}"

CMD=(
  python -u run.py
  --probe final
  --train_hours 100
  --batch_size 16
  --eval_batch_size 32
  --epochs 10
  --lr 5e-4
  --warmup_steps 500
  --runs_dir ./runs
)

echo "JobID: ${JOBID}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "Log file: ${LOG_FILE}"
echo "Running command: ${CMD[*]}"

"${CMD[@]}" > "${LOG_FILE}" 2>&1
