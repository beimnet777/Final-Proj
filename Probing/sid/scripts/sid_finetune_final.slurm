#!/bin/bash
#
# One-epoch fine-tune of the saved SID final-layer probe from best checkpoint.
# Submit from anywhere:  sbatch sid/scripts/sid_finetune_final.slurm
#

#SBATCH -J sid_ft_final
#SBATCH -A MLMI-bbg25-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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

WORKDIR="/rds/user/${USER}/hpc-work/Thesis/Final-Proj/Probing/sid"
cd "${WORKDIR}"
mkdir -p logs
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

JOBID="${SLURM_JOB_ID}"
EXPERIMENT_NAME="spear_sid_finetune_final_voxceleb1"
LOG_FILE="logs/out.${EXPERIMENT_NAME}.${JOBID}"

VOXCELEB1_ROOT="/rds/user/${USER}/hpc-work/data/VoxCeleb1"
CHECKPOINT="${WORKDIR}/checkpoints/sid_probe_final_best.pt"

CMD=(
  python -u sid_finetune.py
  --probe final
  --checkpoint "${CHECKPOINT}"
  --voxceleb1_root "${VOXCELEB1_ROOT}"
  --extra_epochs 1
  --lr 1e-5
  --batch_size 8
  --eval_batch_size 16
  --checkpoint_dir "${WORKDIR}/checkpoints/${JOBID}"
  --runs_dir "${WORKDIR}/runs/${JOBID}"
  --num_workers 0
)

echo "JobID: ${JOBID}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Time: $(date)"

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
