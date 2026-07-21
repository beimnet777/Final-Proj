#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=21:00:00
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
# from last.pt, so an interrupted allocation can safely be continued. Once max_steps
# is reached, the job renders ten held-out registered pairs with recipient/donor
# references, original-SPEAR reconstruction, SAE baseline, P swap and L swap.
#
# Submit:
#   sbatch SAEUnitAnalysis/slurm/train_direct_hifigan_blackwell.sh
#
# Validate the real CSD3 paths on a login node without extracting SPEAR
# features or training. This also builds/validates the symlink-only bundle,
# verifies the downloaded warm start, and loads it into the exact full model:
#   DRY_RUN=1 bash SAEUnitAnalysis/slurm/train_direct_hifigan_blackwell.sh

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module purge 2>/dev/null || true
  module load rhel8/default-amp 2>/dev/null || true
fi

REPO_ROOT="${REPO_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj}"
PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"

# CSD3 keeps the extracted LibriSpeech tree under Probing/data. The local
# data/sae_analysis bundle is git-ignored and is not present on the cluster.
# This job creates a lightweight, symlink-only bundle from the existing CSD3
# audio tree; phone alignments are not needed for waveform-vocoder training.
LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-${REPO_ROOT}/Probing/data/LibriSpeech}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/checkpoints/blackwell/libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42/final.pt}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/librispeech_csd3_audio_bundle_full}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/spear_direct_cache_trainclean100_full}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/spear_direct_hifigan_trainclean100_full}"
PAIR_MANIFEST="${PAIR_MANIFEST:-${REPO_ROOT}/SAEUnitAnalysis/configs/direct_hifigan_demo_pairs.csv}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_ROOT}/SAEUnitAnalysis/audio_models/pretrained}"
PRETRAINED_GENERATOR="${PRETRAINED_GENERATOR:-${PRETRAINED_DIR}/knnvc_regular_g_02500000.pt}"
PRETRAINED_URL="${PRETRAINED_URL:-https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt}"
PRETRAINED_SHA256="${PRETRAINED_SHA256:-f98b760e0e5fd0019cffd3a9d22bab5c4c2fe38491532d7680a8ad07eaf3e8dd}"

CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
# Zero means all available utterances. The production job intentionally uses
# the complete train-clean-100 and dev-clean splits to maximize speaker and
# phonetic coverage; reduced subsets must be requested explicitly.
MAX_TRAIN_UTTERANCES="${MAX_TRAIN_UTTERANCES:-0}"
MAX_VALIDATION_UTTERANCES="${MAX_VALIDATION_UTTERANCES:-0}"
EXPECTED_TRAIN_UTTERANCES="${EXPECTED_TRAIN_UTTERANCES:-28539}"
EXPECTED_VALIDATION_UTTERANCES="${EXPECTED_VALIDATION_UTTERANCES:-2703}"
EXPECTED_TEST_UTTERANCES="${EXPECTED_TEST_UTTERANCES:-2620}"
MAX_STEPS="${MAX_STEPS:-100000}"
BEST_SNAPSHOT="${BEST_SNAPSHOT:-${OUTPUT_DIR}/best_after_step${MAX_STEPS}.pt}"
DEMO_OUTPUT_DIR="${DEMO_OUTPUT_DIR:-${OUTPUT_DIR}/demo_after_step${MAX_STEPS}_10_pairs}"
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
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
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
  --direct-hifigan "${BEST_SNAPSHOT}"
  --pair-manifest "${PAIR_MANIFEST}"
  --output-dir "${DEMO_OUTPUT_DIR}"
  --device cuda
  --pairs 10
  --batch-size 4
  --length-tolerance 0.10
)

echo "+ cd ${REPO_ROOT}"
echo "CSD3 LibriSpeech  : ${LIBRISPEECH_ROOT}"
echo "Checkpoint        : ${CHECKPOINT}"
echo "Data bundle       : ${DATA_ROOT}"
echo "Feature cache     : ${CACHE_DIR}"
echo "Training output   : ${OUTPUT_DIR}"
echo "+ ${PYTHON} ${cache_args[*]}"
echo "+ ${PYTHON} ${train_args[*]}"
if [[ "${RENDER_FINAL_DEMO}" == "1" ]]; then
  echo "+ ${PYTHON} ${demo_args[*]}"
fi

[[ -d "${REPO_ROOT}" ]] || { echo "Missing repository: ${REPO_ROOT}" >&2; exit 2; }
[[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
for split in train-clean-100 dev-clean test-clean; do
  [[ -d "${LIBRISPEECH_ROOT}/${split}" ]] || {
    echo "Missing CSD3 LibriSpeech split: ${LIBRISPEECH_ROOT}/${split}" >&2
    exit 2
  }
done
if [[ "${RENDER_FINAL_DEMO}" == "1" ]]; then
  [[ -f "${PAIR_MANIFEST}" ]] || { echo "Missing demo pair registry: ${PAIR_MANIFEST}" >&2; exit 2; }
fi

cd "${REPO_ROOT}"
"${PYTHON}" - "${CHECKPOINT}" "${LIBRISPEECH_ROOT}" "${PAIR_MANIFEST}" "${RENDER_FINAL_DEMO}" <<'PY'
import csv
import sys
from pathlib import Path

import numpy
import pandas
import soundfile
import torch
import torchaudio
import yaml

from SAEUnitAnalysis.checkpoint import load_checkpoint
from SAEUnitAnalysis.direct_hifigan import DirectSpearHiFiGenerator

checkpoint = Path(sys.argv[1]).resolve()
librispeech_root = Path(sys.argv[2]).resolve()
pair_manifest = Path(sys.argv[3]).resolve()
render_demo = sys.argv[4] == "1"

if render_demo:
    with pair_manifest.open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    if len(pairs) != 10:
        raise SystemExit(f"ERROR: expected 10 registered demo pairs, found {len(pairs)}")
    for pair in pairs:
        for role, speaker_key in (("recipient", "recipient_speaker"), ("donor", "donor_speaker")):
            utterance_id = str(pair[role])
            parts = utterance_id.split("-")
            if len(parts) < 3:
                raise SystemExit(f"ERROR: malformed demo utterance ID: {utterance_id}")
            if parts[0] != str(pair[speaker_key]):
                raise SystemExit(
                    f"ERROR: speaker mismatch for demo {role} {utterance_id}: "
                    f"ID={parts[0]} registry={pair[speaker_key]}"
                )
            stem = librispeech_root / "test-clean" / parts[0] / parts[1] / utterance_id
            if not any(stem.with_suffix(suffix).is_file() for suffix in (".flac", ".wav")):
                raise SystemExit(f"ERROR: registered test audio is missing: {stem}.flac")

resolved = load_checkpoint(checkpoint)
if int(resolved.config.get("D", -1)) != 1280:
    raise SystemExit(f"ERROR: expected 1280-D SPEAR input, got {resolved.config.get('D')}")
block_topk = resolved.config.get("block_topk") or resolved.config.get("topk_blocks")
if list(block_topk or []) != [240, 16, 0]:
    raise SystemExit(f"ERROR: expected fixed 240L/16P checkpoint, got block_topk={block_topk}")

print("[preflight] imports: passed")
print(f"[preflight] CSD3 LibriSpeech: train-clean-100/dev-clean/test-clean present")
print(f"[preflight] checkpoint: D=1280 block_topk={list(block_topk)}")
if render_demo:
    print("[preflight] final demo: 10 registered test-clean pairs passed")
PY

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot use the allocated CUDA GPU")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"ERROR: expected one visible GPU, found {torch.cuda.device_count()}")
properties = torch.cuda.get_device_properties(0)
x = torch.randn(256, 256, device="cuda")
float((x @ x).mean())
print(f"[preflight] CUDA: {properties.name}, {properties.total_memory / 2**30:.1f} GiB")
PY
fi

mkdir -p \
  "${PRETRAINED_DIR}" "${DATA_ROOT}" "${CACHE_DIR}" "${OUTPUT_DIR}" \
  "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" \
  "${REPO_ROOT}/Disentanglement/blackwell/logs"

if [[ ! -f "${DATA_ROOT}/dataset.yaml" || ! -f "${DATA_ROOT}/utterances.csv" ]]; then
  echo "[direct-hifigan] creating symlink-only analysis bundle from CSD3 LibriSpeech"
  "${PYTHON}" -m SAEUnitAnalysis.build_librispeech_bundle \
    --librispeech-root "${LIBRISPEECH_ROOT}" \
    --output "${DATA_ROOT}" \
    --max-train 0 \
    --max-validation 0 \
    --max-test 0 \
    --seed "${SEED}"
fi

"${PYTHON}" - \
  "${DATA_ROOT}" "${PAIR_MANIFEST}" "${RENDER_FINAL_DEMO}" \
  "${EXPECTED_TRAIN_UTTERANCES}" "${EXPECTED_VALIDATION_UTTERANCES}" \
  "${EXPECTED_TEST_UTTERANCES}" <<'PY'
import csv
import sys
from pathlib import Path

from SAEUnitAnalysis.bundle import AnalysisBundle

data_root = Path(sys.argv[1]).resolve()
pair_manifest = Path(sys.argv[2]).resolve()
render_demo = sys.argv[3] == "1"
expected = {
    "train": int(sys.argv[4]),
    "validation": int(sys.argv[5]),
    "test": int(sys.argv[6]),
}
bundle = AnalysisBundle(data_root)
counts = {name: len(bundle.split(name)) for name in ("train", "validation", "test")}
if counts != expected:
    raise SystemExit(
        "ERROR: CSD3 bundle is not the complete canonical LibriSpeech selection: "
        f"observed={counts} expected={expected}"
    )

missing_audio = []
split_by_id = {}
speaker_by_id = {}
for _, row in bundle.utterances.iterrows():
    utterance_id = str(row["utterance_id"])
    path = bundle.audio_path(row)
    if not path.is_file() and len(missing_audio) < 10:
        missing_audio.append(str(path))
    split_by_id[utterance_id] = str(row["split"])
    speaker_by_id[utterance_id] = str(row.get("speaker_id", ""))
if missing_audio:
    raise SystemExit("ERROR: generated bundle audio is missing; first paths: " + repr(missing_audio))

if render_demo:
    with pair_manifest.open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    test_split = str(bundle.spec.split_map["test"])
    for pair in pairs:
        for role, speaker_key in (("recipient", "recipient_speaker"), ("donor", "donor_speaker")):
            utterance_id = str(pair[role])
            if utterance_id not in split_by_id:
                raise SystemExit(f"ERROR: demo {role} is absent from generated bundle: {utterance_id}")
            if split_by_id[utterance_id] != test_split:
                raise SystemExit(f"ERROR: demo {role} {utterance_id} is not in test split")
            if speaker_by_id[utterance_id] != str(pair[speaker_key]):
                raise SystemExit(f"ERROR: speaker mismatch for demo {role} {utterance_id}")

print(f"[preflight] full LibriSpeech bundle: {counts}; {len(bundle.utterances)} audio links passed")
PY

verify_pretrained() {
  [[ -s "$1" ]] || return 1
  "${PYTHON}" - "$1" "${PRETRAINED_SHA256}" <<'PY'
import hashlib
import sys

path, expected = sys.argv[1:]
digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
if digest != expected:
    raise SystemExit(f"SHA256 mismatch for {path}: expected {expected}, got {digest}")
PY
}

if ! verify_pretrained "${PRETRAINED_GENERATOR}"; then
  temporary="${PRETRAINED_GENERATOR}.partial"
  rm -f "${temporary}"
  echo "[direct-hifigan] downloading published kNN-VC generator warm start"
  curl -L --fail --retry 3 --output "${temporary}" "${PRETRAINED_URL}"
  verify_pretrained "${temporary}"
  mv "${temporary}" "${PRETRAINED_GENERATOR}"
fi

"${PYTHON}" - "${PRETRAINED_GENERATOR}" "${OUTPUT_DIR}" <<'PY'
import sys
from pathlib import Path

import torch

from SAEUnitAnalysis.direct_hifigan import (
    DirectHiFiGANConfig,
    DirectSpearHiFiGenerator,
    load_direct_hifigan,
    load_pretrained_knnvc_generator,
)

pretrained = Path(sys.argv[1]).resolve()
output_dir = Path(sys.argv[2]).resolve()
generator = DirectSpearHiFiGenerator(DirectHiFiGANConfig(input_dim=1280))
report = load_pretrained_knnvc_generator(generator, pretrained)
expected_incompatible = {"lin_pre.weight"}
actual_incompatible = set(report["incompatible_shapes"])
if actual_incompatible != expected_incompatible:
    raise SystemExit(
        "ERROR: unexpected kNN-VC warm-start incompatibilities: "
        f"expected={sorted(expected_incompatible)} actual={sorted(actual_incompatible)}"
    )
if int(report["loaded_tensors"]) != int(report["total_target_tensors"]) - 1:
    raise SystemExit(
        "ERROR: incomplete kNN-VC warm start: "
        f"loaded={report['loaded_tensors']} total={report['total_target_tensors']}"
    )
with torch.inference_mode():
    generated = generator(torch.zeros(1, 2, 1280))
if tuple(generated.shape) != (1, 1, 640) or not bool(torch.isfinite(generated).all()):
    raise SystemExit(
        f"ERROR: full generator contract failed: shape={tuple(generated.shape)} "
        f"finite={bool(torch.isfinite(generated).all())}"
    )

last = output_dir / "last.pt"
best = output_dir / "best.pt"
if last.exists():
    resumed, payload = load_direct_hifigan(last)
    if int(resumed.config.input_dim) != 1280:
        raise SystemExit(f"ERROR: resume checkpoint has input_dim={resumed.config.input_dim}")
    if not best.is_file():
        raise SystemExit(f"ERROR: resume checkpoint exists but best checkpoint is missing: {best}")
    load_direct_hifigan(best)
    print(f"[preflight] resume: step={int(payload.get('step', 0)):,}; best.pt passed")

print(
    f"[preflight] kNN-VC warm start: SHA-256 passed; "
    f"loaded {report['loaded_tensors']}/{report['total_target_tensors']} tensors; "
    "full generator forward passed"
)
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[preflight] DRY_RUN passed all CPU-side checks; extraction, CUDA, and training were not started."
  exit 0
fi

"${PYTHON}" "${cache_args[@]}"
"${PYTHON}" "${train_args[@]}"
if [[ "${RENDER_FINAL_DEMO}" == "1" ]]; then
  [[ -f "${OUTPUT_DIR}/best.pt" ]] || { echo "Training completed without best.pt" >&2; exit 2; }
  "${PYTHON}" - "${OUTPUT_DIR}/best.pt" "${BEST_SNAPSHOT}" <<'PY'
import shutil
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, destination)
print(f"[direct-hifigan] preserved selected checkpoint: {destination}")
PY
  "${PYTHON}" "${demo_args[@]}"
fi
