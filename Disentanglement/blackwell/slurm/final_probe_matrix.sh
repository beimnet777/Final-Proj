#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=36:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=libri_final_probe
#SBATCH --array=0-7%1
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%A_%a.err

# Final focused Libri probe matrix.
#
# This script intentionally runs only the agreed missing/rechecked probes:
#   0) recon-only SAE baseline: z_t ASR
#   1) fixed 240L/16P: z_t,z_L PR linear
#   2) fixed 240L/16P: z_L SID linear
#   3) fixed 240L/16P: z_t,z_L,z_P SID stats
#   4) learned quota-freeze 20k: z_t,z_L,z_P ASR
#   5) learned quota-freeze 20k: z_t,z_L,z_P SID stats
#   6) learned quota-freeze 16k: z_t,z_L,z_P ASR
#   7) learned quota-freeze 16k: z_t,z_L,z_P SID stats
#
# Submit all sequentially:
#   sbatch Disentanglement/blackwell/slurm/final_probe_matrix.sh
#
# Submit one task:
#   sbatch --array=4 Disentanglement/blackwell/slurm/final_probe_matrix.sh
#
# Run two at a time if the queue/GPU budget allows:
#   sbatch --array=0-7%2 Disentanglement/blackwell/slurm/final_probe_matrix.sh

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
LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Probing/data/LibriSpeech}"
LEXICON_PATH="${LEXICON_PATH:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Probing/data/librispeech-lexicon.txt}"
NUM_WORKERS="${NUM_WORKERS:-8}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"

LOG_ROOT="${DIS_DIR}/blackwell/logs/final_probe_matrix"
JSON_ROOT="${LOG_ROOT}/json"
mkdir -p "${LOG_ROOT}" "${JSON_ROOT}"

cd "${REPO_ROOT}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"

RUN=""
LABEL=""
SOURCES=""
TASKS=""
STEPS=5000
ROUTE_KIND=""
declare -a TASK_ARGS=()

case "${TASK_ID}" in
  0)
    RUN="libri_sae_recon_topk256_aux64_12k_s42"
    LABEL="zt_asr_lstm_10k_full"
    SOURCES="z_t"
    TASKS="asr"
    STEPS=10000
    ROUTE_KIND="plain"
    TASK_ARGS=(
      --asr_probe_arch lstm
      --asr_probe_lr 5e-4
      --asr_probe_warmup_steps 500
      --asr_probe_proj_dim 1024
      --asr_lstm_hidden 1024
      --asr_lstm_layers 2
      --asr_time_mask_param 50
      --asr_freq_mask_param 64
      --asr_probe_dropout 0.1
      --asr_max_examples 0
    )
    ;;
  1)
    RUN="libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42"
    LABEL="zt_zL_pr_linear_5k_full"
    SOURCES="z_t,z_L"
    TASKS="pr"
    STEPS=5000
    ROUTE_KIND="fixed_240_16"
    TASK_ARGS=(
      --pr_probe_arch linear
      --pr_probe_lr 5e-4
      --pr_probe_warmup_steps 500
      --pr_max_examples 0
      --no-pr_checkpoint_sanity
    )
    ;;
  2)
    RUN="libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42"
    LABEL="zL_sid_linear_5k_full"
    SOURCES="z_L"
    TASKS="sid"
    STEPS=5000
    ROUTE_KIND="fixed_240_16"
    TASK_ARGS=(
      --sid_probe_arch linear
      --sid_probe_lr 1e-3
    )
    ;;
  3)
    RUN="libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42"
    LABEL="zt_zL_zP_sid_stats_5k_full"
    SOURCES="z_t,z_L,z_P"
    TASKS="sid"
    STEPS=5000
    ROUTE_KIND="fixed_240_16"
    TASK_ARGS=(
      --sid_probe_arch stats
      --sid_probe_lr 1e-3
    )
    ;;
  4)
    RUN="libri_advlearn_hardqfreeze4000_gn0002_dann6800_gp02_aux64_20k_s42"
    LABEL="zt_zL_zP_asr_lstm_10k_full"
    SOURCES="z_t,z_L,z_P"
    TASKS="asr"
    STEPS=10000
    ROUTE_KIND="learned_hard_binary"
    TASK_ARGS=(
      --asr_probe_arch lstm
      --asr_probe_lr 5e-4
      --asr_probe_warmup_steps 500
      --asr_probe_proj_dim 1024
      --asr_lstm_hidden 1024
      --asr_lstm_layers 2
      --asr_time_mask_param 50
      --asr_freq_mask_param 64
      --asr_probe_dropout 0.1
      --asr_max_examples 0
    )
    ;;
  5)
    RUN="libri_advlearn_hardqfreeze4000_gn0002_dann6800_gp02_aux64_20k_s42"
    LABEL="zt_zL_zP_sid_stats_5k_full"
    SOURCES="z_t,z_L,z_P"
    TASKS="sid"
    STEPS=5000
    ROUTE_KIND="learned_hard_binary"
    TASK_ARGS=(
      --sid_probe_arch stats
      --sid_probe_lr 1e-3
    )
    ;;
  6)
    RUN="libri_advlearn_hardqfreeze4000_gn0002_dann6800_gp02_aux64_16k_s42"
    LABEL="zt_zL_zP_asr_lstm_10k_full"
    SOURCES="z_t,z_L,z_P"
    TASKS="asr"
    STEPS=10000
    ROUTE_KIND="learned_hard_binary"
    TASK_ARGS=(
      --asr_probe_arch lstm
      --asr_probe_lr 5e-4
      --asr_probe_warmup_steps 500
      --asr_probe_proj_dim 1024
      --asr_lstm_hidden 1024
      --asr_lstm_layers 2
      --asr_time_mask_param 50
      --asr_freq_mask_param 64
      --asr_probe_dropout 0.1
      --asr_max_examples 0
    )
    ;;
  7)
    RUN="libri_advlearn_hardqfreeze4000_gn0002_dann6800_gp02_aux64_16k_s42"
    LABEL="zt_zL_zP_sid_stats_5k_full"
    SOURCES="z_t,z_L,z_P"
    TASKS="sid"
    STEPS=5000
    ROUTE_KIND="learned_hard_binary"
    TASK_ARGS=(
      --sid_probe_arch stats
      --sid_probe_lr 1e-3
    )
    ;;
  *)
    echo "Unknown TASK_ID=${TASK_ID}; expected 0..7." >&2
    exit 2
    ;;
esac

CKPT="${CKPT_ROOT}/${RUN}/final.pt"
RUN_NAME="diag_${RUN}_${LABEL}_seed42"
OUTPUT_JSON="${JSON_ROOT}/${RUN_NAME}.json"

declare -a ROUTE_ARGS=()
case "${ROUTE_KIND}" in
  plain)
    ROUTE_ARGS=()
    ;;
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
  echo "Expected tracked final.pt files under CKPT_ROOT=${CKPT_ROOT}" >&2
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

echo "=== Libri final probe matrix ==="
echo "started     : $(date)"
echo "task_id     : ${TASK_ID}"
echo "run         : ${RUN}"
echo "label       : ${LABEL}"
echo "checkpoint  : ${CKPT}"
echo "sources     : ${SOURCES}"
echo "tasks       : ${TASKS}"
echo "steps       : ${STEPS}"
echo "patience    : 0 (run full steps)"
echo "route_kind  : ${ROUTE_KIND}"
echo "json        : ${OUTPUT_JSON}"
echo "python      : ${PYTHON}"
echo "gpu         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

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
  --probe_steps "${STEPS}" \
  --probe_val_every 250 \
  --probe_patience 0 \
  --probe_warmup_steps 0 \
  --seed 42 \
  --num_workers "${NUM_WORKERS}" \
  --sources "${SOURCES}" \
  --tasks "${TASKS}" \
  "${ROUTE_ARGS[@]}" \
  "${TASK_ARGS[@]}"

echo "finished    : $(date)"
