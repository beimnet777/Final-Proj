#!/bin/bash
#
# SPEAR-XL PR probing: fixed uniform layer average CTC probe on LibriSpeech 100h.
# Submit from anywhere inside pr/ or pr/scripts/:
#   sbatch pr/scripts/pr_fixed_weighted.slurm
#

#SBATCH -J pr_fixed_weighted
#SBATCH -A MLMI-bbg25-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /home/${USER}/.bashrc || true

set -euo pipefail
source /rds/project/rds-xyBFuSj0hm0/MLMI6.L2026/miniconda3/bin/activate
conda activate mlmi4

WORKDIR="/rds/user/${USER}/hpc-work/Thesis/Final-Proj/Probing/pr"
cd "${WORKDIR}"
mkdir -p logs
export OMP_NUM_THREADS=1

JOBID="${SLURM_JOB_ID}"
EXPERIMENT_NAME="spear_pr_fixed_weighted_librispeech100"
LOG_FILE="logs/out.${EXPERIMENT_NAME}.${JOBID}"

DATA_CACHE="$(dirname "${WORKDIR}")/data"
LEXICON_PATH="${DATA_CACHE}/librispeech-lexicon.txt"
if [[ ! -f "${LEXICON_PATH}" ]]; then
    echo "Downloading LibriSpeech lexicon..."
    mkdir -p "${DATA_CACHE}"
    wget -q https://www.openslr.org/resources/11/librispeech-lexicon.txt \
         -O "${LEXICON_PATH}"
fi

CMD=(
  python -u pr_run.py
  --probe fixed_weighted
  --epochs 10
  --batch_size 16
  --eval_batch_size 16
  --lr 5e-4
  --warmup_steps 500
  --data_cache_dir "${DATA_CACHE}"
  --lexicon_path "${LEXICON_PATH}"
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
