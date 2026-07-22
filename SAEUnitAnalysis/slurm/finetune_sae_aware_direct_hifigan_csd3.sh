#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=saeaware_hifigan
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.err

# Continue one mature direct HiFi-GAN checkpoint on a 50/50 mixture of:
#   original frozen-SPEAR h -> waveform
#   fixed-SAE reconstruction h_hat -> the same waveform
# SPEAR and the SAE remain frozen. No swapped condition is paired with source
# audio, because doing so would teach the vocoder to ignore the intervention.
#
# The base checkpoint is deliberately a required script argument:
#   sbatch SAEUnitAnalysis/slurm/finetune_sae_aware_direct_hifigan_csd3.sh \
#     /rds/user/bbg25/hpc-work/Thesis/Final-Proj/SAEUnitAnalysis/audio_models/spear_direct_hifigan_trainclean100_full/step_00100000.pt
#
# The output directory is separate from the original vocoder run and resumes
# from its own last.pt if this allocation is interrupted.

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module purge 2>/dev/null || true
  module load rhel8/default-amp 2>/dev/null || true
fi

REPO_ROOT="${REPO_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj}"
PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
BASE_VOCODER_CHECKPOINT="${1:-${BASE_VOCODER_CHECKPOINT:-}}"
[[ -n "${BASE_VOCODER_CHECKPOINT}" ]] || {
  echo "Usage: sbatch $0 /absolute/path/to/step_XXXXXXXX.pt" >&2
  exit 2
}

SAE_CHECKPOINT="${SAE_CHECKPOINT:-${REPO_ROOT}/checkpoints/blackwell/libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42/final.pt}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/librispeech_csd3_audio_bundle_full}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/spear_direct_cache_trainclean100_full}"
PAIR_MANIFEST="${PAIR_MANIFEST:-${REPO_ROOT}/SAEUnitAnalysis/configs/direct_hifigan_demo_pairs.csv}"
BASE_TAG="$(basename "${BASE_VOCODER_CHECKPOINT}" .pt)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/spear_direct_hifigan_saeaware_fixed240L16P_mix50_from_${BASE_TAG}_50k}"
DEMO_OUTPUT_DIR="${DEMO_OUTPUT_DIR:-${OUTPUT_DIR}/demo_final_10_pairs}"

ADDITIONAL_STEPS="${ADDITIONAL_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEGMENT_FRAMES="${SEGMENT_FRAMES:-24}"
SAE_FRACTION="${SAE_FRACTION:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-2500}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/torch_home}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

[[ -d "${REPO_ROOT}" ]] || { echo "Missing repository: ${REPO_ROOT}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 2; }
[[ -f "${BASE_VOCODER_CHECKPOINT}" ]] || {
  echo "Missing base vocoder checkpoint: ${BASE_VOCODER_CHECKPOINT}" >&2
  exit 2
}
[[ -f "${SAE_CHECKPOINT}" ]] || { echo "Missing SAE checkpoint: ${SAE_CHECKPOINT}" >&2; exit 2; }
[[ -f "${CACHE_DIR}/manifest.json" ]] || { echo "Missing SPEAR cache: ${CACHE_DIR}" >&2; exit 2; }
[[ -f "${DATA_ROOT}/dataset.yaml" ]] || { echo "Missing data bundle: ${DATA_ROOT}" >&2; exit 2; }
[[ -f "${PAIR_MANIFEST}" ]] || { echo "Missing pair manifest: ${PAIR_MANIFEST}" >&2; exit 2; }
[[ "${OUTPUT_DIR}" != "$(dirname "${BASE_VOCODER_CHECKPOINT}")" ]] || {
  echo "Output directory would overwrite the base vocoder run" >&2
  exit 2
}

cd "${REPO_ROOT}"
"${PYTHON}" - "${BASE_VOCODER_CHECKPOINT}" "${SAE_CHECKPOINT}" "${CACHE_DIR}" <<'PY'
import json
import sys
from pathlib import Path

import torch

from SAEUnitAnalysis.checkpoint import load_checkpoint

base, sae, cache = map(Path, sys.argv[1:])
payload = torch.load(base, map_location="cpu", weights_only=False)
if payload.get("format") != "direct_spear_hifigan_v1":
    raise SystemExit(f"ERROR: unsupported base vocoder checkpoint: {base}")
missing = [key for key in ("generator", "mpd", "msd", "optimizer_g", "optimizer_d")
           if payload.get(key) is None]
if missing:
    raise SystemExit(f"ERROR: base checkpoint is not a full periodic checkpoint: missing={missing}")
manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
resolved = load_checkpoint(sae)
if int(payload["config"]["input_dim"]) != int(manifest["input_dim"]):
    raise SystemExit("ERROR: vocoder/cache input dimensions differ")
if int(resolved.config["D"]) != int(manifest["input_dim"]):
    raise SystemExit("ERROR: SAE/cache input dimensions differ")
for key in ("spear_model_id", "spear_revision", "spear_layernorm"):
    if key in resolved.config and resolved.config[key] != manifest.get(key):
        raise SystemExit(
            f"ERROR: SAE/cache domain mismatch for {key}: "
            f"SAE={resolved.config[key]!r} cache={manifest.get(key)!r}")
print(f"[preflight] base vocoder: step={int(payload.get('step', 0)):,}; full GAN state passed")
print(f"[preflight] fixed SAE/cache domain: D={resolved.config['D']}; passed")
PY

echo "Base vocoder : ${BASE_VOCODER_CHECKPOINT}"
echo "Fixed SAE    : ${SAE_CHECKPOINT}"
echo "Output       : ${OUTPUT_DIR}"
echo "Mixture      : SAE=${SAE_FRACTION}; original=1-SAE"
echo "Continuation : ${ADDITIONAL_STEPS} additional steps"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[preflight] DRY_RUN passed; no GPU training started."
  exit 0
fi

"${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot use the allocated CUDA GPU")
print(f"[preflight] CUDA: {torch.cuda.get_device_name(0)}")
PY

mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" -m SAEUnitAnalysis.train_sae_aware_direct_hifigan \
  --cache "${CACHE_DIR}" \
  --data-root "${DATA_ROOT}" \
  --sae-checkpoint "${SAE_CHECKPOINT}" \
  --base-vocoder-checkpoint "${BASE_VOCODER_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --additional-steps "${ADDITIONAL_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --segment-frames "${SEGMENT_FRAMES}" \
  --sae-fraction "${SAE_FRACTION}" \
  --learning-rate "${LEARNING_RATE}" \
  --validation-interval "${VALIDATION_INTERVAL}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --validation-batches "${VALIDATION_BATCHES}" \
  --num-workers "${NUM_WORKERS}" \
  --keep-periodic 3 \
  --seed "${SEED}"

"${PYTHON}" -m SAEUnitAnalysis.render_direct_hifigan_demo \
  --checkpoint "${SAE_CHECKPOINT}" \
  --data "${DATA_ROOT}" \
  --direct-hifigan "${OUTPUT_DIR}/last.pt" \
  --pair-manifest "${PAIR_MANIFEST}" \
  --output-dir "${DEMO_OUTPUT_DIR}" \
  --device cuda \
  --pairs 10 \
  --batch-size 4 \
  --length-tolerance 0.10

echo "[sae-aware-hifigan] final listening report: ${DEMO_OUTPUT_DIR}/report/index.html"
