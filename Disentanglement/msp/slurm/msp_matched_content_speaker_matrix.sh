#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=07:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_matched_probe
#SBATCH --array=0-14%6
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.err

# Matched MSP content/speaker comparison for exactly three final checkpoints:
#   - Libri fixed routing;
#   - MSP strong-clean quota-freeze;
#   - MSP corrected no-balance quota-freeze.
#
# Members 0..2 : projected-linear PR, all z_t/z_L/z_P, 7,500 steps.
# Members 3..5 : linear stats-pool SID, all z_t/z_L/z_P, 7,500 steps.
# Members 6..14: LSTM ASR, one source per GPU, 15,000 steps.
# Every trained probe runs the complete schedule. Validation selects the best
# state independently for each source before the held-out test evaluation.

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module purge 2>/dev/null || true
  module load rhel8/default-amp 2>/dev/null || true
fi

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
REPO_ROOT="${REPO_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj}"
DIS_DIR="${DIS_DIR:-${REPO_ROOT}/Disentanglement}"
BLACKWELL_CKPT_ROOT="${BLACKWELL_CKPT_ROOT:-${REPO_ROOT}/checkpoints/blackwell}"
MSP_CKPT_ROOT="${MSP_CKPT_ROOT:-${DIS_DIR}/msp/checkpoints}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

MANIFEST="${MANIFEST:-${DIS_DIR}/data/msp_subset}"
AUDIO_ROOT="${AUDIO_ROOT:-${DIS_DIR}/data/msp_audio}"
TRANSCRIPTS="${TRANSCRIPTS:-/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0/Transcripts.zip}"
LEXICON_PATH="${LEXICON_PATH:-${REPO_ROOT}/Probing/data/librispeech-lexicon.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DIS_DIR}/msp/probe_results/matched_content_speaker}"

PR_STEPS="${PR_STEPS:-7500}"
SID_STEPS="${SID_STEPS:-7500}"
ASR_STEPS="${ASR_STEPS:-15000}"
VAL_EVERY="${VAL_EVERY:-500}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  PR_STEPS=2
  SID_STEPS=2
  ASR_STEPS=2
  VAL_EVERY=1
fi

declare -a RUN_NAMES=(
  "libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42"
  "msp_hardqfreeze4000_strongclean_dann12000_advpe010_s42"
  "msp_hardqfreeze4000_sc_optfix_nobal_aux64_gn0004_grlp025_dann12000_12k_s42"
)
declare -a RUN_KINDS=("libri" "msp" "msp")

if (( TASK_ID < 0 || TASK_ID > 14 )); then
  echo "Unknown TASK_ID=${TASK_ID}; expected 0..14." >&2
  exit 2
fi

if (( TASK_ID < 3 )); then
  RUN_INDEX=${TASK_ID}
  TASKS="pr"
  SOURCES="z_t,z_L,z_P"
  STEPS="${PR_STEPS}"
  LABEL="pr_linear_all_7500"
  EXTRA_ARGS=(
    --pr_probe_arch linear
    --pr_probe_proj_dim 256
    --pr_probe_lr 5e-4
    --pr_probe_warmup_steps 500
  )
elif (( TASK_ID < 6 )); then
  RUN_INDEX=$((TASK_ID - 3))
  TASKS="sid"
  SOURCES="z_t,z_L,z_P"
  STEPS="${SID_STEPS}"
  LABEL="sid_linear_all_7500"
  EXTRA_ARGS=(--lr 5e-4)
else
  OFFSET=$((TASK_ID - 6))
  RUN_INDEX=$((OFFSET / 3))
  SOURCE_INDEX=$((OFFSET % 3))
  declare -a SOURCE_NAMES=("z_t" "z_L" "z_P")
  SOURCES="${SOURCE_NAMES[${SOURCE_INDEX}]}"
  TASKS="asr"
  STEPS="${ASR_STEPS}"
  LABEL="asr_lstm_${SOURCES}_15000"
  EXTRA_ARGS=(
    --asr_probe_arch lstm
    --asr_probe_lr 5e-4
    --asr_probe_warmup_steps 500
    --asr_probe_proj_dim 1024
    --asr_lstm_hidden 1024
    --asr_lstm_layers 2
    --asr_time_mask_param 50
    --asr_freq_mask_param 64
    --asr_probe_dropout 0.1
  )
fi

RUN_NAME="${RUN_NAMES[${RUN_INDEX}]}"
RUN_KIND="${RUN_KINDS[${RUN_INDEX}]}"
if [[ "${RUN_KIND}" == "libri" ]]; then
  CHECKPOINT="${BLACKWELL_CKPT_ROOT}/${RUN_NAME}/final.pt"
else
  CHECKPOINT="${MSP_CKPT_ROOT}/${RUN_NAME}/final.pt"
fi
OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}_${LABEL}_seed${SEED}.json"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${DIS_DIR}/msp/logs" "${OUTPUT_ROOT}"
  cd "${DIS_DIR}"
  [[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 3; }
  [[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 4; }
  [[ -f "${LEXICON_PATH}" ]] || { echo "Missing lexicon: ${LEXICON_PATH}" >&2; exit 5; }
  [[ -e "${TRANSCRIPTS}" ]] || { echo "Missing transcripts: ${TRANSCRIPTS}" >&2; exit 6; }
  [[ -f "${MANIFEST}/manifest.csv" || -f "${MANIFEST}" ]] || {
    echo "Missing MSP manifest: ${MANIFEST}" >&2; exit 7;
  }
  [[ -d "${AUDIO_ROOT}" ]] || { echo "Missing MSP audio root: ${AUDIO_ROOT}" >&2; exit 8; }
fi

COMMAND=("${PYTHON}" -u -m msp.probe
  --checkpoint "${CHECKPOINT}"
  --manifest "${MANIFEST}"
  --audio_root "${AUDIO_ROOT}"
  --transcripts "${TRANSCRIPTS}"
  --lexicon_path "${LEXICON_PATH}"
  --sources "${SOURCES}"
  --tasks "${TASKS}"
  --steps "${STEPS}"
  --val_every "${VAL_EVERY}"
  --batch_size 16
  --eval_batch 32
  --num_workers 8
  --lr 5e-4
  --seed "${SEED}"
  --output "${OUTPUT}"
  "${EXTRA_ARGS[@]}")

echo "=== Matched MSP content/speaker probe ==="
echo "started    : $(date)"
echo "task_id    : ${TASK_ID}"
echo "run        : ${RUN_NAME}"
echo "checkpoint : ${CHECKPOINT}"
echo "sources    : ${SOURCES}"
echo "task       : ${TASKS}"
echo "steps      : ${STEPS}"
echo "output     : ${OUTPUT}"
printf '+ %q ' "${COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "1" ]]; then
  "${COMMAND[@]}"
fi

echo "finished   : $(date)"
