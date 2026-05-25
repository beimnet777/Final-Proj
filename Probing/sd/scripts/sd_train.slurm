#!/bin/bash
#
# SPEAR-XL spoof detection: weighted-layer proj+pool+MLP probe on ASVspoof 2019 LA.
# Submit from anywhere inside sd/ or sd/scripts/:
#   sbatch sd/scripts/sd_train.slurm
#

#SBATCH -J sd_weighted
#SBATCH -A MLMI-bbg25-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END
#SBATCH -p ampere

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /home/${USER}/.bashrc || true

set -euo pipefail
source /rds/project/rds-xyBFuSj0hm0/MLMI6.L2026/miniconda3/bin/activate
conda activate mlmi4

# Resolve working directory to sd/ regardless of where sbatch was called from.
WORKDIR="/rds/user/${USER}/hpc-work/Thesis/Final-Proj/Probing/sd"
cd "${WORKDIR}"
mkdir -p logs
export OMP_NUM_THREADS=1

JOBID="${SLURM_JOB_ID}"
EXPERIMENT_NAME="spear_sd_weighted_asv19la"
LOG_FILE="logs/out.${EXPERIMENT_NAME}.${JOBID}"

# ── Required: ASVspoof 2019 LA ───────────────────────────────────────────────
# Download with:  bash sd/download_data.sh
ASV19_LA_ROOT="/rds/user/${USER}/hpc-work/data/ASVspoof2019/LA"
if [[ ! -d "${ASV19_LA_ROOT}" ]]; then
    echo "ERROR: ASV19_LA_ROOT not found: ${ASV19_LA_ROOT}" >&2
    echo "Run:   bash sd/download_data.sh" >&2
    exit 1
fi

# ── Optional: cross-dataset test sets ─────────────────────────────────────────
# Any dataset whose path is absent is silently skipped by sd_run.py —
# the job will NOT fail if these directories do not exist.
# Uncomment a block once the corresponding data is on disk.

# ASV21_KEYS_DIR="/rds/user/${USER}/hpc-work/data/ASVspoof2021_keys"
# ASV21_LA_ROOT="/rds/user/${USER}/hpc-work/data/ASVspoof2021_LA"
# ASV21_DF_ROOT="/rds/user/${USER}/hpc-work/data/ASVspoof2021_DF"
# ITW_ROOT="/rds/user/${USER}/hpc-work/data/ITW"

# Build the command — required args first, then add optional eval args
# if the corresponding directories have been set above.
CMD=(
    python -u sd_run.py
    --probe weighted
    --asv19_la_root "${ASV19_LA_ROOT}"
    --epochs 10
    --batch_size 8
    --eval_batch_size 16
    --lr 1e-4
    --warmup_steps 500
    --num_workers 0
    --checkpoint_dir "${WORKDIR}/checkpoints/${JOBID}"
    --runs_dir "${WORKDIR}/runs/${JOBID}"
)

# Add optional cross-dataset eval args (uncomment each group as data becomes available)
# [[ -n "${ASV21_LA_ROOT:-}" && -n "${ASV21_KEYS_DIR:-}" ]] && \
#   CMD+=(--asv21_la_root "${ASV21_LA_ROOT}" --asv21_la_keys "${ASV21_KEYS_DIR}/keys_2021_LA_eval.txt")
# [[ -n "${ASV21_DF_ROOT:-}" && -n "${ASV21_KEYS_DIR:-}" ]] && \
#   CMD+=(--asv21_df_root "${ASV21_DF_ROOT}" --asv21_df_keys "${ASV21_KEYS_DIR}/keys_2021_DF_eval.txt")
# [[ -n "${ITW_ROOT:-}" ]] && \
#   CMD+=(--itw_root "${ITW_ROOT}")

echo "JobID: ${JOBID}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "Log file: ${LOG_FILE}"
echo "Running command: ${CMD[*]}"

"${CMD[@]}" > "${LOG_FILE}" 2>&1
