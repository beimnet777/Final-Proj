#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=spear_mel_bridge
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.err

# Train one checkpoint-independent SPEAR -> BigVGAN-compatible log-mel bridge.
# No SAE route, swap, test utterance, or speaker label is used for fitting.
#
# Submit:
#   sbatch SAEUnitAnalysis/slurm/train_audio_bridge_blackwell.sh
#
# Dry-run:
#   DRY_RUN=1 bash SAEUnitAnalysis/slurm/train_audio_bridge_blackwell.sh

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module purge 2>/dev/null || true
  module load rhel8/default-amp 2>/dev/null || true
fi

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
REPO_ROOT="${REPO_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/blackwell/libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42/final.pt}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/sae_analysis/librispeech_bundle_12k_mfa}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/spear_bigvgan_v2_24khz_100band}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
HIDDEN_DIM="${HIDDEN_DIM:-384}"
RESIDUAL_LAYERS="${RESIDUAL_LAYERS:-6}"
MAX_TRAIN_UTTERANCES="${MAX_TRAIN_UTTERANCES:-0}"
MAX_VALIDATION_UTTERANCES="${MAX_VALIDATION_UTTERANCES:-1000}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

cd "${REPO_ROOT}"

args=(
  -m SAEUnitAnalysis.train_audio_bridge
  --checkpoint "${CHECKPOINT}"
  --data "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --device cuda
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --hidden-dim "${HIDDEN_DIM}"
  --residual-layers "${RESIDUAL_LAYERS}"
  --max-train-utterances "${MAX_TRAIN_UTTERANCES}"
  --max-validation-utterances "${MAX_VALIDATION_UTTERANCES}"
  --mel-sample-rate 24000
  --n-fft 1024
  --win-length 1024
  --hop-length 256
  --n-mels 100
  --f-min 0
  --f-max 12000
  --seed "${SEED}"
)

echo "+ ${PYTHON} ${args[*]}"
if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
  [[ -f "${DATA_ROOT}/dataset.yaml" ]] || { echo "Missing analysis bundle: ${DATA_ROOT}" >&2; exit 2; }
  mkdir -p "${OUTPUT_DIR}"
  "${PYTHON}" "${args[@]}"
fi
