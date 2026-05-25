#!/bin/bash
#
# SPEAR-XL ER probing: weighted-layer-sum mean-pool linear probe on IEMOCAP.
# Runs all 5 cross-validation folds in a single job.
# Submit from anywhere inside er/ or er/scripts/:
#   sbatch er/scripts/er_weighted.slurm
#

#SBATCH -J er_weighted
#SBATCH -A MLMI-bbg25-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
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

# Resolve working directory to er/ regardless of where sbatch was called from.
WORKDIR="/rds/user/${USER}/hpc-work/Thesis/Final-Proj/Probing/er"
cd "${WORKDIR}"
mkdir -p logs
export OMP_NUM_THREADS=1

JOBID="${SLURM_JOB_ID}"
EXPERIMENT_NAME="spear_er_weighted_iemocap"
LOG_FILE="logs/out.${EXPERIMENT_NAME}.${JOBID}"

# Path to the extracted IEMOCAP_full_release directory.
# Request access at: https://sail.usc.edu/iemocap/
IEMOCAP_ROOT="/rds/project/rds-xyBFuSj0hm0/dataset/IEMOCAP_full_release"
if [[ ! -d "${IEMOCAP_ROOT}" ]]; then
  echo "ERROR: IEMOCAP_ROOT not found: ${IEMOCAP_ROOT}" >&2
  exit 1
fi

CMD=(
  python -u er_run.py
  --probe weighted
  --iemocap_root "${IEMOCAP_ROOT}"
  --epochs 20
  --batch_size 16
  --eval_batch_size 16
  --lr 1e-4
  --warmup_steps 100
  --checkpoint_dir "${WORKDIR}/checkpoints/${JOBID}"
  --runs_dir "${WORKDIR}/runs/${JOBID}"
)

echo "JobID: ${JOBID}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "Log file: ${LOG_FILE}"
echo "Running command: ${CMD[*]}"

"${CMD[@]}" > "${LOG_FILE}" 2>&1
