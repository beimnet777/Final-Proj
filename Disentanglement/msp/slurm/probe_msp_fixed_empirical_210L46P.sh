#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_fix210_probe
#SBATCH --array=0-2%3
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.err

# Independent probes for the completed MSP fixed-routing control.
#
# Only final.pt is evaluated: the validation-selected best.pt is at step 3k,
# before the adversaries have received the full 12k schedule.
#
# Array members:
#   0: linear PR and prosody probes, 5k complete steps
#   1: linear SID probes, 7.5k complete steps
#   2: speaker-disjoint emotion probes, 5k complete steps
#
# Submit from the repository root:
#   sbatch Disentanglement/msp/slurm/probe_msp_fixed_empirical_210L46P.sh

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module purge 2>/dev/null || true
  module load rhel8/default-amp 2>/dev/null || true
fi

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
REPO_ROOT="${REPO_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj}"
DIS_DIR="${DIS_DIR:-${REPO_ROOT}/Disentanglement}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEED:-42}"
VAL_EVERY=500
DRY_RUN="${DRY_RUN:-0}"

RUN_NAME=msp_fixed2130L2990P_topk210L46P_sc_optfix_aux64_gn0004_grlp025_dann12000_12k_s42
CHECKPOINT="${DIS_DIR}/msp/checkpoints/${RUN_NAME}/final.pt"
MANIFEST="${DIS_DIR}/data/msp_subset"
AUDIO_ROOT="${DIS_DIR}/data/msp_audio"
TRANSCRIPTS="/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0/Transcripts.zip"
LEXICON_PATH="${REPO_ROOT}/Probing/data/librispeech-lexicon.txt"
OUTPUT_ROOT="${DIS_DIR}/msp/probe_results/fixed_empirical_210L46P"

SOURCES=z_t,z_L,z_P
EMOTION_DIAGNOSTICS=0
SPEAKER_DISJOINT=0

case "${TASK_ID}" in
  0)
    TASKS=pr,prosody
    STEPS=5000
    LABEL=final12k_pr_prosody_probe5k
    ;;
  1)
    TASKS=sid
    STEPS=7500
    LABEL=final12k_sid_probe7500
    ;;
  2)
    TASKS=emotion
    STEPS=5000
    LABEL=final12k_emotion_speaker_disjoint_probe5k
    EMOTION_DIAGNOSTICS=1
    SPEAKER_DISJOINT=1
    ;;
  *)
    echo "Unknown task ${TASK_ID}; expected 0 through 2" >&2
    exit 2
    ;;
esac

OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}_${LABEL}_seed${SEED}.json"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"

mkdir -p "${DIS_DIR}/msp/logs" "${OUTPUT_ROOT}"
cd "${DIS_DIR}"

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 3; }
  [[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 4; }
  [[ -f "${LEXICON_PATH}" ]] || { echo "Missing lexicon: ${LEXICON_PATH}" >&2; exit 5; }
  [[ -f "${TRANSCRIPTS}" ]] || { echo "Missing transcripts: ${TRANSCRIPTS}" >&2; exit 6; }
fi

PROBE_COMMAND=(
  "${PYTHON}" -u -m msp.probe
  --checkpoint "${CHECKPOINT}"
  --manifest "${MANIFEST}"
  --audio_root "${AUDIO_ROOT}"
  --transcripts "${TRANSCRIPTS}"
  --lexicon_path "${LEXICON_PATH}"
  --steps "${STEPS}"
  --val_every "${VAL_EVERY}"
  --batch_size 16
  --eval_batch 32
  --num_workers 8
  --lr 5e-4
  --seed "${SEED}"
  --sources "${SOURCES}"
  --tasks "${TASKS}"
  --pr_probe_arch linear
  --pr_probe_proj_dim 256
)
if [[ "${SPEAKER_DISJOINT}" == "1" ]]; then
  PROBE_COMMAND+=(--speaker_disjoint_emotion_split)
fi
if [[ "${EMOTION_DIAGNOSTICS}" == "1" ]]; then
  PROBE_COMMAND+=(--emotion_diagnostics)
fi
PROBE_COMMAND+=(--output "${OUTPUT}")

echo "=== MSP fixed-routing independent probe ==="
echo "started    : $(date)"
echo "task       : ${TASK_ID}"
echo "checkpoint : ${CHECKPOINT}"
echo "sources    : ${SOURCES}"
echo "tasks      : ${TASKS}"
echo "schedule   : ${STEPS} complete steps; val every ${VAL_EVERY}"
echo "output     : ${OUTPUT}"
printf '+ %q ' "${PROBE_COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PROBE_COMMAND[@]}"
fi

echo "finished   : $(date)"
