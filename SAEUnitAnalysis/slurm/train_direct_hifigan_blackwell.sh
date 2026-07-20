#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=spear_direct_hifigan
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.err

# Direct waveform path:
#   frozen SPEAR h (50 Hz, 1280-D) -> HiFi-GAN -> 16 kHz waveform
#
# The first invocation creates one reusable float16 SPEAR cache containing only
# train+validation. Later invocations reuse it. Training resumes automatically
# from last.pt, so a 24-hour allocation can safely be continued. Once max_steps
# is reached, the job renders ten held-out registered pairs with recipient/donor
# references, original-SPEAR reconstruction, SAE baseline, P swap and L swap.
#
# Submit:
#   sbatch SAEUnitAnalysis/slurm/train_direct_hifigan_blackwell.sh
#
# Validate paths and command without extracting/training:
#   DRY_RUN=1 bash SAEUnitAnalysis/slurm/train_direct_hifigan_blackwell.sh

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
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/spear_direct_cache_train10k_val1k}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/spear_direct_hifigan_knnvc_init}"
PAIR_MANIFEST="${PAIR_MANIFEST:-${REPO_ROOT}/SAEUnitAnalysis/configs/direct_hifigan_demo_pairs.csv}"
DEMO_OUTPUT_DIR="${DEMO_OUTPUT_DIR:-${OUTPUT_DIR}/final_demo_10_pairs}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/pretrained}"
PRETRAINED_GENERATOR="${PRETRAINED_GENERATOR:-${PRETRAINED_DIR}/knnvc_regular_g_02500000.pt}"
PRETRAINED_URL="${PRETRAINED_URL:-https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt}"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
MAX_TRAIN_UTTERANCES="${MAX_TRAIN_UTTERANCES:-0}"
MAX_VALIDATION_UTTERANCES="${MAX_VALIDATION_UTTERANCES:-1000}"
MAX_STEPS="${MAX_STEPS:-250000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEGMENT_FRAMES="${SEGMENT_FRAMES:-24}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
ADVERSARIAL_START_STEP="${ADVERSARIAL_START_STEP:-5000}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-5000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
RENDER_FINAL_DEMO="${RENDER_FINAL_DEMO:-1}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/torch_home}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

cache_args=(
  -m SAEUnitAnalysis.cache_spear_audio_features
  --checkpoint "${CHECKPOINT}"
  --data "${DATA_ROOT}"
  --output-dir "${CACHE_DIR}"
  --device cuda
  --batch-size "${CACHE_BATCH_SIZE}"
  --max-train-utterances "${MAX_TRAIN_UTTERANCES}"
  --max-validation-utterances "${MAX_VALIDATION_UTTERANCES}"
  --seed "${SEED}"
)

train_args=(
  -m SAEUnitAnalysis.train_direct_hifigan
  --cache "${CACHE_DIR}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --device cuda
  --model-size full
  --max-steps "${MAX_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --segment-frames "${SEGMENT_FRAMES}"
  --learning-rate "${LEARNING_RATE}"
  --adversarial-start-step "${ADVERSARIAL_START_STEP}"
  --validation-interval "${VALIDATION_INTERVAL}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --validation-batches "${VALIDATION_BATCHES}"
  --num-workers "${NUM_WORKERS}"
  --pretrained-generator "${PRETRAINED_GENERATOR}"
  --keep-periodic 3
  --seed "${SEED}"
)

if [[ -f "${OUTPUT_DIR}/last.pt" ]]; then
  train_args+=(--resume "${OUTPUT_DIR}/last.pt")
fi

demo_args=(
  -m SAEUnitAnalysis.render_direct_hifigan_demo
  --checkpoint "${CHECKPOINT}"
  --data "${DATA_ROOT}"
  --direct-hifigan "${OUTPUT_DIR}/best.pt"
  --pair-manifest "${PAIR_MANIFEST}"
  --output-dir "${DEMO_OUTPUT_DIR}"
  --device cuda
  --pairs 10
  --batch-size 4
  --length-tolerance 0.10
)

echo "+ cd ${REPO_ROOT}"
echo "+ ${PYTHON} ${cache_args[*]}"
echo "+ ${PYTHON} ${train_args[*]}"
if [[ "${RENDER_FINAL_DEMO}" == "1" ]]; then
  echo "+ ${PYTHON} ${demo_args[*]}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

[[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/dataset.yaml" ]] || { echo "Missing data bundle: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/utterances.csv" ]] || { echo "Missing utterance manifest: ${DATA_ROOT}/utterances.csv" >&2; exit 2; }
if [[ "${RENDER_FINAL_DEMO}" == "1" ]]; then
  [[ -f "${PAIR_MANIFEST}" ]] || { echo "Missing demo pair registry: ${PAIR_MANIFEST}" >&2; exit 2; }
fi
mkdir -p "${PRETRAINED_DIR}" "${CACHE_DIR}" "${OUTPUT_DIR}" "${REPO_ROOT}/Disentanglement/blackwell/logs"

if [[ ! -s "${PRETRAINED_GENERATOR}" ]]; then
  temporary="${PRETRAINED_GENERATOR}.partial"
  rm -f "${temporary}"
  echo "[direct-hifigan] downloading published kNN-VC generator warm start"
  curl -L --fail --retry 3 --output "${temporary}" "${PRETRAINED_URL}"
  mv "${temporary}" "${PRETRAINED_GENERATOR}"
fi

cd "${REPO_ROOT}"
"${PYTHON}" "${cache_args[@]}"
"${PYTHON}" "${train_args[@]}"
if [[ "${RENDER_FINAL_DEMO}" == "1" ]]; then
  [[ -f "${OUTPUT_DIR}/best.pt" ]] || { echo "Training completed without best.pt" >&2; exit 2; }
  "${PYTHON}" "${demo_args[@]}"
fi
