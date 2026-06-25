#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.err

# Standalone MSP-Podcast disentanglement: content + speaker + prosody + emotion in
# one dataset, per-batch, with PCGrad over the cooperative tasks.

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"

mkdir -p "${DIS_DIR}/msp/logs"
cd "${DIS_DIR}"

RUN_NAME="${RUN_NAME:-msp_v1}"
STEPS="${STEPS:-12000}"
EXTRA_ARGS=("$@")          # e.g. --no_pcgrad / --soft_routing / --grl_emotion_weight 0.8

echo "=== MSP standalone disentanglement ==="
echo "started   : $(date)"
echo "gpu       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "run_name  : ${RUN_NAME}"
echo "steps     : ${STEPS}"
echo "extra     : ${EXTRA_ARGS[*]:-(none)}"

${PYTHON} -u -m msp.run \
    --run_name "${RUN_NAME}" \
    --steps "${STEPS}" \
    --num_workers 8 \
    "${EXTRA_ARGS[@]}"

echo "finished  : $(date)"
