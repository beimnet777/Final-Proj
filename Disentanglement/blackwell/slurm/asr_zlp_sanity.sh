#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=36:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=libri_asr_zlp_sanity
#SBATCH --array=0-3%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%A_%a.err

# Focused ASR sanity checks for the final Libri result.
#
# Why this exists:
#   - learned 20k z_L ASR looked collapsed despite good z_L PR.
#   - learned 20k z_P ASR became surprisingly good compared with old fixed z_P ASR.
#   - old fixed z_P ASR used early stopping / had NaNs; this script uses full 10k.
#
# Tasks:
#   0) learned 20k: z_L,z_P ASR, seed 42, same settings as final matrix
#   1) learned 20k: z_L,z_P ASR, seed 43, same settings
#   2) learned 20k: z_L,z_P ASR, seed 42, SpecAugment off
#   3) fixed 240L/16P: z_P ASR, seed 42, full 10k no early stopping
#
# Submit:
#   sbatch Disentanglement/blackwell/slurm/asr_zlp_sanity.sh
#
# Submit one task:
#   sbatch --array=0 Disentanglement/blackwell/slurm/asr_zlp_sanity.sh

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
LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-${REPO_ROOT}/Probing/data/LibriSpeech}"
LEXICON_PATH="${LEXICON_PATH:-${REPO_ROOT}/Probing/data/librispeech-lexicon.txt}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"

LOG_ROOT="${DIS_DIR}/blackwell/logs/asr_zlp_sanity"
JSON_ROOT="${LOG_ROOT}/json"
mkdir -p "${LOG_ROOT}" "${JSON_ROOT}"

cd "${REPO_ROOT}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"

RUN=""
LABEL=""
SOURCES=""
ROUTE_KIND=""
SEED=42
TIME_MASK=50
FREQ_MASK=64

case "${TASK_ID}" in
  0)
    RUN="libri_advlearn_hardqfreeze4000_gn0002_dann6800_gp02_aux64_20k_s42"
    LABEL="zL_zP_asr_lstm_10k_repeat_seed42"
    SOURCES="z_L,z_P"
    ROUTE_KIND="learned_hard_binary"
    SEED=42
    ;;
  1)
    RUN="libri_advlearn_hardqfreeze4000_gn0002_dann6800_gp02_aux64_20k_s42"
    LABEL="zL_zP_asr_lstm_10k_repeat_seed43"
    SOURCES="z_L,z_P"
    ROUTE_KIND="learned_hard_binary"
    SEED=43
    ;;
  2)
    RUN="libri_advlearn_hardqfreeze4000_gn0002_dann6800_gp02_aux64_20k_s42"
    LABEL="zL_zP_asr_lstm_10k_noaug_seed42"
    SOURCES="z_L,z_P"
    ROUTE_KIND="learned_hard_binary"
    SEED=42
    TIME_MASK=0
    FREQ_MASK=0
    ;;
  3)
    RUN="libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42"
    LABEL="zP_asr_lstm_10k_full_seed42"
    SOURCES="z_P"
    ROUTE_KIND="fixed_240_16"
    SEED=42
    ;;
  *)
    echo "Unknown TASK_ID=${TASK_ID}; expected 0..3." >&2
    exit 2
    ;;
esac

CKPT="${CKPT_ROOT}/${RUN}/final.pt"
RUN_NAME="diag_${RUN}_${LABEL}"
OUTPUT_JSON="${JSON_ROOT}/${RUN_NAME}.json"

declare -a ROUTE_ARGS=()
case "${ROUTE_KIND}" in
  fixed_240_16)
    ROUTE_ARGS=(
      --fixed_blocks
      --per_block_topk
      --K_L 4096
      --K_P 1024
      --K_U 0
      --topk_L 240
      --topk_P 16
      --topk_U 0
    )
    ;;
  learned_hard_binary)
    ROUTE_ARGS=(
      --n_routes 2
      --hard_gumbel_routing
      --gumbel_tau_end 0.1
    )
    ;;
  *)
    echo "Internal error: unknown ROUTE_KIND=${ROUTE_KIND}" >&2
    exit 3
    ;;
esac

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 4
fi
if [[ ! -d "${LIBRISPEECH_ROOT}" ]]; then
  echo "Missing LibriSpeech root: ${LIBRISPEECH_ROOT}" >&2
  exit 5
fi
if [[ ! -f "${LEXICON_PATH}" ]]; then
  echo "Missing lexicon: ${LEXICON_PATH}" >&2
  exit 6
fi

echo "=== Libri ASR z_L/z_P sanity ==="
echo "started        : $(date)"
echo "task_id        : ${TASK_ID}"
echo "run            : ${RUN}"
echo "label          : ${LABEL}"
echo "checkpoint     : ${CKPT}"
echo "sources        : ${SOURCES}"
echo "route_kind     : ${ROUTE_KIND}"
echo "seed           : ${SEED}"
echo "steps          : 10000"
echo "patience       : 0 (run full steps)"
echo "time/freq mask : ${TIME_MASK}/${FREQ_MASK}"
echo "json           : ${OUTPUT_JSON}"
echo "python         : ${PYTHON}"
echo "gpu            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

"${PYTHON}" -u Disentanglement/diag_probe/run.py \
  --stage2_ckpt "${CKPT}" \
  --stage1_ckpt "${CKPT}" \
  --run_name "${RUN_NAME}" \
  --output_json "${OUTPUT_JSON}" \
  --local_data \
  --librispeech_root "${LIBRISPEECH_ROOT}" \
  --lexicon_path "${LEXICON_PATH}" \
  --spear_layernorm \
  --topk 256 \
  --probe_steps 10000 \
  --probe_val_every 250 \
  --probe_patience 0 \
  --probe_warmup_steps 0 \
  --seed "${SEED}" \
  --num_workers "${NUM_WORKERS}" \
  --sources "${SOURCES}" \
  --tasks asr \
  "${ROUTE_ARGS[@]}" \
  --asr_probe_arch lstm \
  --asr_probe_lr 5e-4 \
  --asr_probe_warmup_steps 500 \
  --asr_probe_proj_dim 1024 \
  --asr_lstm_hidden 1024 \
  --asr_lstm_layers 2 \
  --asr_time_mask_param "${TIME_MASK}" \
  --asr_freq_mask_param "${FREQ_MASK}" \
  --asr_probe_dropout 0.1 \
  --asr_max_examples 0

echo "finished       : $(date)"
