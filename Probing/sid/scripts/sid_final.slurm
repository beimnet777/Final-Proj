#!/bin/bash
#
# SPEAR-XL SID probing: final-layer mean-pool linear probe on VoxCeleb1.
# Submit from anywhere inside sid/ or sid/scripts/:
#   sbatch sid/scripts/sid_final.slurm
#

#SBATCH -J sid_final
#SBATCH -A MLMI-bbg25-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=25:00:00
#SBATCH --mail-type=END
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /home/${USER}/.bashrc || true

set -euo pipefail
source /rds/project/rds-xyBFuSj0hm0/MLMI6.L2026/miniconda3/bin/activate
conda activate mlmi4

# Resolve working directory to sid/ regardless of where sbatch was called from.
WORKDIR="/rds/user/${USER}/hpc-work/Thesis/Final-Proj/Probing/sid"
cd "${WORKDIR}"
mkdir -p logs
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

JOBID="${SLURM_JOB_ID}"
EXPERIMENT_NAME="spear_sid_final_voxceleb1"
LOG_FILE="logs/out.${EXPERIMENT_NAME}.${JOBID}"

VOXCELEB1_ROOT="/rds/user/${USER}/hpc-work/data/VoxCeleb1"
if [[ ! -d "${VOXCELEB1_ROOT}/dev" ]]; then
  echo "ERROR: VoxCeleb1 dev split not found under: ${VOXCELEB1_ROOT}" >&2
  exit 1
fi

CMD=(
  python -u sid_run.py
  --probe final
  --voxceleb1_root "${VOXCELEB1_ROOT}"
  --epochs 4
  --batch_size 8
  --eval_batch_size 8
  --lr 4e-4
  --warmup_steps 100
  --num_workers 0
  --train_max_duration_s 8.0
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
