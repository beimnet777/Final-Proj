#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=libri_fixed_vdead
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.err

# Extension, not replacement, of the best fixed Libri run:
#   libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42
#
# Only intended change:
#   --valid_frame_dead_count
#
# This makes the AuxK/dead-latent counter ignore padded frames when deciding
# whether a latent has fired. Everything else follows the fixed 240L/16P recipe.
#
# Submit:
#   sbatch Disentanglement/blackwell/slurm/fixed_validdead_12k.sh
#
# Dry run:
#   DRY_RUN=1 bash Disentanglement/blackwell/slurm/fixed_validdead_12k.sh

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module purge 2>/dev/null || true
  module load rhel8/default-amp 2>/dev/null || true
fi

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
REPO_ROOT="${REPO_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj}"
DIS_DIR="${DIS_DIR:-${REPO_ROOT}/Disentanglement}"
LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-${REPO_ROOT}/Probing/data/LibriSpeech}"
LEXICON_PATH="${LEXICON_PATH:-${REPO_ROOT}/Probing/data/librispeech-lexicon.txt}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/blackwell_followups}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

RUN_NAME="libri_advfb_prrecover_validdead_gn00015_240L16P_aux64_12k_s${SEED}"
RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
CKPT_DIR="${RUN_DIR}/checkpoints"
TB_DIR="${RUN_DIR}/tensorboard"
TRAINER_LOG_DIR="${RUN_DIR}/trainer_logs"
LOG_ROOT="${DIS_DIR}/blackwell/logs/${RUN_NAME}"
JSON_ROOT="${LOG_ROOT}/json"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -x "${PYTHON}" ]] || { echo "Missing python: ${PYTHON}" >&2; exit 4; }
  [[ -d "${LIBRISPEECH_ROOT}" ]] || { echo "Missing LibriSpeech root: ${LIBRISPEECH_ROOT}" >&2; exit 5; }
  [[ -f "${LEXICON_PATH}" ]] || { echo "Missing lexicon: ${LEXICON_PATH}" >&2; exit 6; }
  mkdir -p "${CKPT_DIR}" "${TB_DIR}" "${TRAINER_LOG_DIR}" "${LOG_ROOT}" "${JSON_ROOT}"
  cd "${REPO_ROOT}"
else
  if [[ -d "${REPO_ROOT}" ]]; then
    cd "${REPO_ROOT}"
  fi
fi

run_or_print() {
  echo
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

final_ckpt() {
  if [[ -f "${CKPT_DIR}/stage2_step12000.pt" ]]; then
    printf '%s\n' "${CKPT_DIR}/stage2_step12000.pt"
  else
    printf '%s\n' "${CKPT_DIR}/final.pt"
  fi
}

train_data_args=(
  --local_data
  --librispeech_root "${LIBRISPEECH_ROOT}"
  --lexicon_path "${LEXICON_PATH}"
  --train_split_dir train-clean-100
  --speaker_stratified_holdout
  --spear_layernorm
)

probe_data_args=(
  --local_data
  --librispeech_root "${LIBRISPEECH_ROOT}"
  --lexicon_path "${LEXICON_PATH}"
  --spear_layernorm
)

fixed_route_args=(
  --topk 256
  --fixed_blocks
  --per_block_topk
  --K_L 4096
  --K_P 1024
  --K_U 0
  --topk_L 240
  --topk_P 16
  --topk_U 0
)

probe_common_args=(
  "${probe_data_args[@]}"
  "${fixed_route_args[@]}"
  --seed "${SEED}"
  --num_workers "${NUM_WORKERS}"
  --probe_val_every 250
  --probe_patience 0
  --probe_warmup_steps 0
  --no-pr_checkpoint_sanity
)

run_probe() {
  local label="$1"
  local tasks="$2"
  shift 2
  local run_name="diag_${RUN_NAME}_${label}_seed${SEED}"
  run_or_print "${PYTHON}" -u Disentanglement/diag_probe/run.py \
    --stage2_ckpt "${CKPT}" \
    --stage1_ckpt "${CKPT}" \
    "${probe_common_args[@]}" \
    --run_name "${run_name}" \
    --output_json "${JSON_ROOT}/${run_name}.json" \
    --sources z_t,z_L,z_P \
    --tasks "${tasks}" \
    "$@"
}

echo "=== Libri fixed valid-frame-dead-count extension ==="
echo "started          : $(date)"
echo "run_name         : ${RUN_NAME}"
echo "repo_root        : ${REPO_ROOT}"
echo "librispeech_root : ${LIBRISPEECH_ROOT}"
echo "lexicon_path     : ${LEXICON_PATH}"
echo "run_dir          : ${RUN_DIR}"
echo "log_root         : ${LOG_ROOT}"
echo "dry_run          : ${DRY_RUN}"
echo "python           : ${PYTHON}"
echo "gpu              : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

run_or_print "${PYTHON}" -u Disentanglement/run.py \
  --stage 2 \
  --stage2_from_scratch \
  "${train_data_args[@]}" \
  --K 5120 \
  "${fixed_route_args[@]}" \
  --grl_linear_mean \
  --grl_grad_norm \
  --grl_grad_norm_target 0.00015 \
  --alpha 0.8 \
  --beta 0.6 \
  --grl_weight 1.0 \
  --grl_phoneme_weight 0.2 \
  --grl_delay_steps 0 \
  --dann_full_discriminator \
  --lr_disc 1e-3 \
  --n_disc_steps 3 \
  --aux_k 64 \
  --aux_k_coef 0.03125 \
  --dead_steps_threshold 256 \
  --valid_frame_dead_count \
  --rho 0.0 \
  --stage2_steps 12000 \
  --stage2_schedule_steps 12000 \
  --warmup_steps 500 \
  --batch_size 16 \
  --eval_batch_size 32 \
  --lr 1e-4 \
  --lr_min 1e-5 \
  --lr_heads 1e-4 \
  --lr_sid_head 0.001 \
  --grad_clip 1.0 \
  --log_every 100 \
  --grad_log_every 500 \
  --ckpt_every 1000 \
  --num_workers 2 \
  --seed "${SEED}" \
  --checkpoint_dir "${CKPT_DIR}" \
  --runs_dir "${TB_DIR}" \
  --log_dir "${TRAINER_LOG_DIR}"

CKPT="$(final_ckpt)"
if [[ "${DRY_RUN}" != "1" && ! -f "${CKPT}" ]]; then
  echo "Missing trained checkpoint: ${CKPT}" >&2
  exit 7
fi
echo "checkpoint       : ${CKPT}"

run_probe "pr_linear_5k" pr \
  --probe_steps 5000 \
  --pr_probe_warmup_steps 500 \
  --pr_probe_arch linear \
  --pr_probe_lr 5e-4 \
  --pr_max_examples 0

run_probe "sid_linear_5k" sid \
  --probe_steps 5000 \
  --pr_probe_warmup_steps 500 \
  --sid_probe_arch linear \
  --sid_probe_lr 1e-3

run_probe "sid_stats_5k" sid \
  --probe_steps 5000 \
  --pr_probe_warmup_steps 500 \
  --sid_probe_arch stats \
  --sid_probe_lr 1e-3

run_probe "asr_lstm_10k" asr \
  --probe_steps 10000 \
  --asr_probe_arch lstm \
  --asr_probe_lr 5e-4 \
  --asr_probe_warmup_steps 500 \
  --asr_probe_proj_dim 1024 \
  --asr_lstm_hidden 1024 \
  --asr_lstm_layers 2 \
  --asr_time_mask_param 50 \
  --asr_freq_mask_param 64 \
  --asr_probe_dropout 0.1 \
  --asr_max_examples 0

echo
echo "finished         : $(date)"
