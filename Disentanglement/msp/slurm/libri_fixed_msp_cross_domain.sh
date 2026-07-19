#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=libri_fixed_msp_xdom
#SBATCH --array=0-5%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.err

# Cross-domain MSP probes for the fixed Libri content/speaker checkpoint only.
#
# Tasks:
#   0  MSP PR/PER:          z_t,z_L,z_P, SUPERB-style projected linear CTC probe
#   1  MSP SID:             z_t,z_L,z_P, closed-set linear stats-pool probe
#   2  MSP speaker EER:     z_t,z_L,z_P, frozen mean+std route cosine verification
#   3  MSP ASR WER/CER:     z_t, LSTM character-CTC probe
#   4  MSP ASR WER/CER:     z_L, LSTM character-CTC probe
#   5  MSP ASR WER/CER:     z_P, LSTM character-CTC probe
#
# Submit from the HPC repo root:
#   cd /rds/user/bbg25/hpc-work/Thesis/Final-Proj
#   sbatch Disentanglement/msp/slurm/libri_fixed_msp_cross_domain.sh
#
# Submit a smoke test for one task:
#   SMOKE=1 sbatch --array=0 Disentanglement/msp/slurm/libri_fixed_msp_cross_domain.sh

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module purge 2>/dev/null || true
  module load rhel8/default-amp 2>/dev/null || true
fi

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
REPO_ROOT="${REPO_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj}"
DIS_DIR="${DIS_DIR:-${REPO_ROOT}/Disentanglement}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/blackwell}"
LEXICON_PATH="${LEXICON_PATH:-${REPO_ROOT}/Probing/data/librispeech-lexicon.txt}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

RUN="libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42"
CHECKPOINT="${FIXED_CKPT:-${CKPT_ROOT}/${RUN}/final.pt}"

MANIFEST="${MANIFEST:-${DIS_DIR}/data/msp_subset}"
AUDIO_ROOT="${AUDIO_ROOT:-${DIS_DIR}/data/msp_audio}"
TRANSCRIPTS="${TRANSCRIPTS:-/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0/Transcripts.zip}"

SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PR_STEPS="${PR_STEPS:-5000}"
SID_STEPS="${SID_STEPS:-5000}"
ASR_STEPS="${ASR_STEPS:-10000}"
VAL_EVERY="${VAL_EVERY:-500}"
SV_MAX_PAIRS="${SV_MAX_PAIRS:-20000}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  PR_STEPS=2
  SID_STEPS=2
  ASR_STEPS=2
  VAL_EVERY=1
  SV_MAX_PAIRS=200
fi

LOG_ROOT="${DIS_DIR}/msp/logs/libri_fixed_msp_cross_domain"
JSON_ROOT="${LOG_ROOT}/json"
mkdir -p "${LOG_ROOT}" "${JSON_ROOT}" "${DIS_DIR}/msp/logs"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[error] Missing checkpoint: ${CHECKPOINT}" >&2
  echo "[hint] Set FIXED_CKPT=/path/to/final.pt or sync ${RUN}/final.pt under ${CKPT_ROOT}." >&2
  exit 4
fi
if [[ ! -f "${MANIFEST}/manifest.csv" && ! -f "${MANIFEST}" ]]; then
  echo "[error] Missing MSP manifest: ${MANIFEST}" >&2
  exit 5
fi
if [[ ! -d "${AUDIO_ROOT}" ]]; then
  echo "[error] Missing MSP audio root: ${AUDIO_ROOT}" >&2
  exit 6
fi
if [[ ! -e "${TRANSCRIPTS}" ]]; then
  echo "[error] Missing MSP transcripts: ${TRANSCRIPTS}" >&2
  exit 7
fi
if [[ ! -f "${LEXICON_PATH}" ]]; then
  echo "[error] Missing lexicon: ${LEXICON_PATH}" >&2
  echo "[hint] Set LEXICON_PATH=/path/to/librispeech-lexicon.txt." >&2
  exit 8
fi

cd "${DIS_DIR}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
LABEL=""
SOURCES=""
TASKS=""
STEPS=0
declare -a EXTRA_ARGS=()

case "${TASK_ID}" in
  0)
    LABEL="pr_linear_zt_zL_zP_5k"
    SOURCES="z_t,z_L,z_P"
    TASKS="pr"
    STEPS="${PR_STEPS}"
    EXTRA_ARGS=(--pr_probe_arch linear --pr_probe_proj_dim 256 --lr 5e-4)
    ;;
  1)
    LABEL="sid_linear_zt_zL_zP_5k"
    SOURCES="z_t,z_L,z_P"
    TASKS="sid"
    STEPS="${SID_STEPS}"
    EXTRA_ARGS=(--lr 1e-3)
    ;;
  2)
    LABEL="sv_eer_zt_zL_zP"
    SOURCES="z_t,z_L,z_P"
    TASKS="sv"
    STEPS=0
    EXTRA_ARGS=(--sv_max_pairs "${SV_MAX_PAIRS}" --sv_seed "${SEED}")
    ;;
  3)
    LABEL="asr_lstm_z_t_10k"
    SOURCES="z_t"
    TASKS="asr"
    STEPS="${ASR_STEPS}"
    EXTRA_ARGS=(--asr_probe_arch lstm --asr_probe_lr 5e-4 --asr_probe_warmup_steps 500)
    ;;
  4)
    LABEL="asr_lstm_z_L_10k"
    SOURCES="z_L"
    TASKS="asr"
    STEPS="${ASR_STEPS}"
    EXTRA_ARGS=(--asr_probe_arch lstm --asr_probe_lr 5e-4 --asr_probe_warmup_steps 500)
    ;;
  5)
    LABEL="asr_lstm_z_P_10k"
    SOURCES="z_P"
    TASKS="asr"
    STEPS="${ASR_STEPS}"
    EXTRA_ARGS=(--asr_probe_arch lstm --asr_probe_lr 5e-4 --asr_probe_warmup_steps 500)
    ;;
  *)
    echo "[error] Unknown TASK_ID=${TASK_ID}; expected 0..5." >&2
    exit 2
    ;;
esac

OUT_JSON="${JSON_ROOT}/${RUN}_${LABEL}_seed${SEED}.json"

echo "=== Libri fixed -> MSP cross-domain probe ==="
echo "started    : $(date)"
echo "task_id    : ${TASK_ID}"
echo "run        : ${RUN}"
echo "checkpoint : ${CHECKPOINT}"
echo "sources    : ${SOURCES}"
echo "tasks      : ${TASKS}"
echo "steps      : ${STEPS}"
echo "val_every  : ${VAL_EVERY}"
echo "manifest   : ${MANIFEST}"
echo "audio_root : ${AUDIO_ROOT}"
echo "lexicon    : ${LEXICON_PATH}"
echo "output     : ${OUT_JSON}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

"${PYTHON}" -u -m msp.probe \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${MANIFEST}" \
  --audio_root "${AUDIO_ROOT}" \
  --transcripts "${TRANSCRIPTS}" \
  --lexicon_path "${LEXICON_PATH}" \
  --sources "${SOURCES}" \
  --tasks "${TASKS}" \
  --steps "${STEPS}" \
  --val_every "${VAL_EVERY}" \
  --batch_size 16 \
  --eval_batch 32 \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --output "${OUT_JSON}" \
  "${EXTRA_ARGS[@]}"

echo "finished   : $(date)"
