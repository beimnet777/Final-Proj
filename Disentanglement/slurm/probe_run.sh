#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#
# Probe worker — called by probe_sweep.sh.
# Parameters injected via --export: RUN_NAME STAGE2_CKPT (may be empty)

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1

mkdir -p "${DIS_DIR}/logs"
cd "${DIS_DIR}"

echo "=== Probe: ${RUN_NAME} ==="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started : $(date)"
echo "stage2_ckpt=${STAGE2_CKPT:-none}"

EXTRA=""
if [[ -n "${STAGE2_CKPT}" ]]; then
    EXTRA="--stage2_ckpt ${STAGE2_CKPT}"
fi

${PYTHON} -u probe_runner.py \
    --stage1_ckpt  "${STAGE1_CKPT:-${DIS_DIR}/checkpoints/best.pt}" \
    --run_name     "${RUN_NAME}" \
    --probe_steps  2000 \
    ${EXTRA} \
    ${PROBE_EXTRA:-}

echo "Finished : $(date)"
