#!/bin/bash
#
# Step 1 of 2: Extract final-layer SPEAR features for the 100h train split,
# val, and test.  Saves fp16 tensors to ./feature_cache/.
#
# Run time: ~30–60 min (one pass over ~100h of audio through SPEAR-XL).
# Cache size: ~7 GB (train) + ~0.5 GB (val) + ~0.3 GB (test) ≈ 8 GB total.
#

#SBATCH -J spear_extract_final
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

JOBID="${SLURM_JOB_ID}"
LOG_FILE="logs/out.spear_extract_final.${JOBID}"

echo "JobID:        ${JOBID}"
echo "Time:         $(date)"
echo "Node:         $(hostname)"
echo "Working dir:  ${WORKDIR}"
echo "Log file:     ${LOG_FILE}"

python -u cache_features.py \
  --cache_dir    ./feature_cache \
  --train_hours  100 \
  --layer_idx    -1 \
  --batch_size   4 \
  2>&1 | tee "${LOG_FILE}"

echo "Extraction done: $(date)"
echo "Cache contents:"
du -sh feature_cache/
ls -lh feature_cache/
