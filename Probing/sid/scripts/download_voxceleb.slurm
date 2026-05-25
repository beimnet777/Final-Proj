#!/bin/bash
#
# Download + extract VoxCeleb1 wav files from HuggingFace on a CPU compute node.
#
# Two-phase approach (handles clusters where compute nodes lack internet):
#   Phase 1 (download):  run manually on login node first if needed
#                          python sid/download_data.py --no_extract --out_dir $OUT_DIR
#   Phase 2 (extract):   this script does the extraction on a compute node
#
# If compute nodes DO have internet access, set DOWNLOAD=1 below and this
# script will do both download and extraction in one job.
#
# Submit from anywhere inside sid/ or sid/scripts/:
#   sbatch sid/scripts/download_voxceleb.slurm
#

#SBATCH -J vox1_download
#SBATCH -A MLMI-bbg25-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
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

OUT_DIR="/rds/user/${USER}/hpc-work/data/VoxCeleb1"

# Set DOWNLOAD=1 if compute nodes have internet access on this cluster.
# Set DOWNLOAD=0 if you already ran: python sid/download_data.py --no_extract
DOWNLOAD=0

echo "JobID: ${SLURM_JOB_ID}"
echo "Node:  $(hostname)"
echo "Time:  $(date)"
echo "Out:   ${OUT_DIR}"

if [[ "${DOWNLOAD}" == "1" ]]; then
    echo "=== Phase 1+2: download + extract ==="
    python -u download_data.py --split all --out_dir "${OUT_DIR}"
else
    echo "=== Phase 2: extract only (zips already in ${OUT_DIR}/.hf_cache) ==="
    python -u download_data.py --split all --out_dir "${OUT_DIR}" --no_download
fi

echo "Done: $(date)"
