#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=libri_fixed_aux512
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.err

# Controlled extension of:
#   libri_advfb_prrecover_validdead_gn00015_240L16P_aux64_12k_s42
#
# Optimization change only:
#   legacy AuxK64 -> adaptive AuxK512 (k_eff=min(512,n_dead)).
# The Aux coefficient, model, routing, objectives, schedule and seed are unchanged.
# Valid-frame dead counting remains enabled. New diagnostics report dead L/P/U,
# interval new/revived units, firing breadth, Aux coverage, and Aux gradient norm.
#
# This job intentionally performs training only. Independent probes should be run
# only if the corrected dead/reconstruction trajectory improves over the control.
#
# Submit:
#   sbatch Disentanglement/blackwell/slurm/fixed_validdead_adaptive_aux512_12k.sh
# Dry run:
#   DRY_RUN=1 bash Disentanglement/blackwell/slurm/fixed_validdead_adaptive_aux512_12k.sh

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
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

RUN_NAME="libri_advfb_prrecover_validdead_adaptaux512_gn00015_240L16P_12k_s${SEED}"
RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
CKPT_DIR="${RUN_DIR}/checkpoints"
TB_DIR="${RUN_DIR}/tensorboard"
TRAINER_LOG_DIR="${RUN_DIR}/trainer_logs"

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
  mkdir -p "${CKPT_DIR}" "${TB_DIR}" "${TRAINER_LOG_DIR}"
  cd "${REPO_ROOT}"
elif [[ -d "${REPO_ROOT}" ]]; then
  cd "${REPO_ROOT}"
fi

cmd=(
  "${PYTHON}" -u Disentanglement/run.py
  --stage 2
  --stage2_from_scratch
  --local_data
  --librispeech_root "${LIBRISPEECH_ROOT}"
  --lexicon_path "${LEXICON_PATH}"
  --train_split_dir train-clean-100
  --speaker_stratified_holdout
  --spear_layernorm
  --K 5120
  --topk 256
  --fixed_blocks
  --per_block_topk
  --K_L 4096 --K_P 1024 --K_U 0
  --topk_L 240 --topk_P 16 --topk_U 0
  --grl_linear_mean
  --grl_grad_norm
  --grl_grad_norm_target 0.00015
  --alpha 0.8
  --beta 0.6
  --grl_weight 1.0
  --grl_phoneme_weight 0.2
  --grl_delay_steps 0
  --dann_full_discriminator
  --lr_disc 1e-3
  --n_disc_steps 3
  --aux_k 512
  --aux_k_coef 0.03125
  --aux_k_adaptive
  --dead_steps_threshold 256
  --valid_frame_dead_count
  --rho 0.0
  --stage2_steps 12000
  --stage2_schedule_steps 12000
  --warmup_steps 500
  --batch_size 16
  --eval_batch_size 32
  --lr 1e-4
  --lr_min 1e-5
  --lr_heads 1e-4
  --lr_sid_head 0.001
  --grad_clip 1.0
  --log_every 100
  --grad_log_every 500
  --ckpt_every 1000
  --num_workers 2
  --seed "${SEED}"
  --checkpoint_dir "${CKPT_DIR}"
  --runs_dir "${TB_DIR}"
  --log_dir "${TRAINER_LOG_DIR}"
)

echo "=== Libri fixed valid-dead adaptive-AuxK extension ==="
echo "started          : $(date)"
echo "run_name         : ${RUN_NAME}"
echo "repo_root        : ${REPO_ROOT}"
echo "librispeech_root : ${LIBRISPEECH_ROOT}"
echo "lexicon_path     : ${LEXICON_PATH}"
echo "run_dir          : ${RUN_DIR}"
echo "dry_run          : ${DRY_RUN}"
printf '+ %q ' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "1" ]]; then
  "${cmd[@]}"
fi

echo "finished         : $(date)"
