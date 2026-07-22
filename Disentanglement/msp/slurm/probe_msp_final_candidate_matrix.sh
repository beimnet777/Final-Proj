#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_final_probe
#SBATCH --array=0-8%9
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.err

# Final, non-duplicative MSP checkpoint probe matrix.
#
# New candidate checkpoints each receive:
#   - PR + emotion + prosody for 5,000 complete steps;
#   - SID for 7,500 complete steps.
# Existing final checkpoints receive SID only because their other probes exist.
# The last task diagnoses emotion on a speaker-disjoint split.
#
# Submit from the repository root:
#   sbatch Disentanglement/msp/slurm/probe_msp_final_candidate_matrix.sh
# Print a member without executing it:
#   SLURM_ARRAY_TASK_ID=0 DRY_RUN=1 bash \
#     Disentanglement/msp/slurm/probe_msp_final_candidate_matrix.sh

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

MANIFEST="${DIS_DIR}/data/msp_subset"
AUDIO_ROOT="${DIS_DIR}/data/msp_audio"
TRANSCRIPTS="/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0/Transcripts.zip"
LEXICON_PATH="${REPO_ROOT}/Probing/data/librispeech-lexicon.txt"
CHECKPOINT_ROOT="${DIS_DIR}/msp/checkpoints"
OUTPUT_ROOT="${DIS_DIR}/msp/probe_results/final_candidate_matrix"

TASKS=""
SOURCES=z_t,z_L,z_P
STEPS=0
LABEL=""
EMOTION_DIAGNOSTICS=0
SPEAKER_DISJOINT=0

case "${TASK_ID}" in
  0)
    RUN_NAME=msp_hardqfreeze4000_sc_optfix_nobal_aux64_gn0004_grlp025_dann12000_12k_s42
    CKPT_NAME=step10000.pt
    TASKS=pr,emotion,prosody
    STEPS=5000
    LABEL=step10k_factor_probe5k
    EMOTION_DIAGNOSTICS=1
    ;;
  1)
    RUN_NAME=msp_hardqfreeze4000_sc_optfix_nobal_aux64_gn0004_grlp025_dann12000_12k_s42
    CKPT_NAME=step10000.pt
    TASKS=sid
    STEPS=7500
    LABEL=step10k_sid_probe7500
    ;;
  2)
    RUN_NAME=msp_hardqfreeze4000_strongclean_gn0004_grlp025_dann12000_advpe010_20k_s42
    CKPT_NAME=step10000.pt
    TASKS=pr,emotion,prosody
    STEPS=5000
    LABEL=step10k_factor_probe5k
    EMOTION_DIAGNOSTICS=1
    ;;
  3)
    RUN_NAME=msp_hardqfreeze4000_strongclean_gn0004_grlp025_dann12000_advpe010_20k_s42
    CKPT_NAME=step10000.pt
    TASKS=sid
    STEPS=7500
    LABEL=step10k_sid_probe7500
    ;;
  4)
    RUN_NAME=msp_hardqfreeze4000_strongclean_dann12000_advpe010_s42
    CKPT_NAME=best.pt
    TASKS=pr,emotion,prosody
    STEPS=5000
    LABEL=best5k_factor_probe5k
    EMOTION_DIAGNOSTICS=1
    ;;
  5)
    RUN_NAME=msp_hardqfreeze4000_strongclean_dann12000_advpe010_s42
    CKPT_NAME=best.pt
    TASKS=sid
    STEPS=7500
    LABEL=best5k_sid_probe7500
    ;;
  6)
    RUN_NAME=msp_hardqfreeze4000_sc_optfix_nobal_aux64_gn0004_grlp025_dann12000_12k_s42
    CKPT_NAME=final.pt
    TASKS=sid
    STEPS=7500
    LABEL=final12k_sid_probe7500
    ;;
  7)
    RUN_NAME=msp_hardqfreeze4000_strongclean_dann12000_advpe010_s42
    CKPT_NAME=final.pt
    TASKS=sid
    STEPS=7500
    LABEL=final12k_sid_probe7500
    ;;
  8)
    RUN_NAME=msp_hardqfreeze4000_sc_optfix_nobal_aux64_gn0004_grlp025_dann12000_12k_s42
    CKPT_NAME=final.pt
    TASKS=emotion
    STEPS=5000
    LABEL=final12k_emotion_speaker_disjoint_probe5k
    SPEAKER_DISJOINT=1
    EMOTION_DIAGNOSTICS=1
    ;;
  *)
    echo "Unknown task ${TASK_ID}; expected 0 through 8" >&2
    exit 2
    ;;
esac

CHECKPOINT="${CHECKPOINT_ROOT}/${RUN_NAME}/${CKPT_NAME}"
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

run_or_print() {
  echo
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

echo "=== MSP final candidate independent probe ==="
echo "started    : $(date)"
echo "task       : ${TASK_ID}"
echo "checkpoint : ${CHECKPOINT}"
echo "sources    : ${SOURCES}"
echo "tasks      : ${TASKS}"
echo "schedule   : ${STEPS} complete steps; val every ${VAL_EVERY}"
echo "output     : ${OUTPUT}"

PROBE_COMMAND=("${PYTHON}" -u -m msp.probe \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${MANIFEST}" \
  --audio_root "${AUDIO_ROOT}" \
  --transcripts "${TRANSCRIPTS}" \
  --lexicon_path "${LEXICON_PATH}" \
  --steps "${STEPS}" \
  --val_every "${VAL_EVERY}" \
  --batch_size 16 \
  --eval_batch 32 \
  --num_workers 8 \
  --lr 5e-4 \
  --seed "${SEED}" \
  --sources "${SOURCES}" \
  --tasks "${TASKS}" \
  --pr_probe_arch linear \
  --pr_probe_proj_dim 256)
if [[ "${SPEAKER_DISJOINT}" == "1" ]]; then
  PROBE_COMMAND+=(--speaker_disjoint_emotion_split)
fi
if [[ "${EMOTION_DIAGNOSTICS}" == "1" ]]; then
  PROBE_COMMAND+=(--emotion_diagnostics)
fi
PROBE_COMMAND+=(--output "${OUTPUT}")

run_or_print "${PROBE_COMMAND[@]}"

echo "finished   : $(date)"
