#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_baladv12k
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.err

# Corrected MSP optimizer experiment:
#   - unit-balance cooperative SAE gradients before PCGrad;
#   - unit-balance factor adversaries, then restore the original weighted
#     adversary-bundle norm;
#   - otherwise reuse the corrected 4k quota-freeze / 12k strong-clean recipe;
#   - independently probe final.pt only for PR/SID/emotion/prosody.
#
# Submit:
#   sbatch Disentanglement/msp/slurm/msp_balanced_adversaries_12k.sh
# Smoke-test command construction:
#   DRY_RUN=1 bash Disentanglement/msp/slurm/msp_balanced_adversaries_12k.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj}"
DIS_DIR="${DIS_DIR:-${REPO_ROOT}/Disentanglement}"
PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
RUN_NAME=msp_hardqfreeze4000_sc_optfix_balcoop_baladv_aux64_gn0004_grlp025_dann12000_12k_s42
COMMON_SCRIPT="${DIS_DIR}/msp/slurm/msp_corrected_strongclean_12k.sh"

export MSP_RUN_NAME="${RUN_NAME}"
export MSP_PCGRAD_BALANCE=unit
export MSP_ADVERSARY_BALANCE=unit_preserve_bundle

[[ -f "${COMMON_SCRIPT}" ]] || { echo "Missing common recipe: ${COMMON_SCRIPT}" >&2; exit 3; }

bash "${COMMON_SCRIPT}"

CHECKPOINT="${DIS_DIR}/msp/checkpoints/${RUN_NAME}/final.pt"
MANIFEST="${DIS_DIR}/data/msp_subset"
AUDIO_ROOT="${DIS_DIR}/data/msp_audio"
TRANSCRIPTS="/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0/Transcripts.zip"
LEXICON_PATH="${REPO_ROOT}/Probing/data/librispeech-lexicon.txt"
OUTPUT_ROOT="${DIS_DIR}/msp/probe_results/selected_checkpoints"
OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}_final12k_full_5k_seed42.json"

mkdir -p "${OUTPUT_ROOT}"
if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${CHECKPOINT}" ]] || { echo "Missing final checkpoint: ${CHECKPOINT}" >&2; exit 4; }
  [[ -f "${LEXICON_PATH}" ]] || { echo "Missing lexicon: ${LEXICON_PATH}" >&2; exit 5; }
fi

run_or_print() {
  echo
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

cd "${DIS_DIR}"
echo "=== Probe corrected balanced-adversary final.pt ==="
echo "started    : $(date)"
echo "checkpoint : ${CHECKPOINT}"
echo "output     : ${OUTPUT}"

run_or_print "${PYTHON}" -u -m msp.probe \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${MANIFEST}" \
  --audio_root "${AUDIO_ROOT}" \
  --transcripts "${TRANSCRIPTS}" \
  --lexicon_path "${LEXICON_PATH}" \
  --steps 5000 \
  --val_every 500 \
  --batch_size 16 \
  --eval_batch 32 \
  --num_workers 8 \
  --lr 5e-4 \
  --seed 42 \
  --sources z_t,z_L,z_P \
  --tasks pr,sid,emotion,prosody \
  --pr_probe_arch linear \
  --pr_probe_proj_dim 256 \
  --output "${OUTPUT}"

echo "finished   : $(date)"
