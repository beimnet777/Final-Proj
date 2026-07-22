#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_emopost
#SBATCH --array=0-1%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.err

# Controlled emotion-cleanup sweep based on the corrected no-balance run.
# Both cases use emotion GRL=0.10 while routing is learned (steps 0-4k), then
# change only the emotion-GRL loss weight after quota-freezing:
#   task 0: 0.20
#   task 1: 0.40
# All other optimisation, routing, task and dead-unit settings are inherited
# unchanged from msp_corrected_strongclean_12k.sh.
#
# Submit:
#   sbatch Disentanglement/msp/slurm/sweep_msp_postfreeze_emotion_grl_12k.sh
# Print a member without executing it:
#   SLURM_ARRAY_TASK_ID=0 DRY_RUN=1 bash \
#     Disentanglement/msp/slurm/sweep_msp_postfreeze_emotion_grl_12k.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj}"
COMMON_SCRIPT="${REPO_ROOT}/Disentanglement/msp/slurm/msp_corrected_strongclean_12k.sh"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

case "${TASK_ID}" in
  0)
    export MSP_RUN_NAME=msp_hardqfreeze4000_sc_optfix_nobal_emopost020_aux64_gn0004_grlp025_dann12000_12k_s42
    export MSP_POSTFREEZE_EMOTION_GRL_WEIGHT=0.20
    ;;
  1)
    export MSP_RUN_NAME=msp_hardqfreeze4000_sc_optfix_nobal_emopost040_aux64_gn0004_grlp025_dann12000_12k_s42
    export MSP_POSTFREEZE_EMOTION_GRL_WEIGHT=0.40
    ;;
  *)
    echo "Unknown task ${TASK_ID}; expected 0 or 1" >&2
    exit 2
    ;;
esac

export MSP_PCGRAD_BALANCE=none
export MSP_ADVERSARY_BALANCE=none
export MSP_EMOTION_GRL_WEIGHT=0.10

if [[ ! -f "${COMMON_SCRIPT}" ]]; then
  echo "Missing common MSP recipe: ${COMMON_SCRIPT}" >&2
  exit 3
fi

echo "=== MSP post-freeze emotion-GRL sweep ==="
echo "task       : ${TASK_ID}"
echo "run_name   : ${MSP_RUN_NAME}"
echo "emotion GRL: pre=${MSP_EMOTION_GRL_WEIGHT} post=${MSP_POSTFREEZE_EMOTION_GRL_WEIGHT}"

exec bash "${COMMON_SCRIPT}"
