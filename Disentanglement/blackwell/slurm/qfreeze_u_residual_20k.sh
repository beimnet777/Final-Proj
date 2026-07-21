#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=libri_qf_ures
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.err

# Three-route Libri qfreeze with a purposeful residual U bucket.
# U reconstructs the detached residual left by L+P and is kept free of speaker
# and phone information with weak pre-freeze and normal post-freeze adversaries.
#
# Train (default):
#   sbatch Disentanglement/blackwell/slurm/qfreeze_u_residual_20k.sh
# Probe only after the training health gate passes:
#   sbatch --export=ALL,PHASE=probe Disentanglement/blackwell/slurm/qfreeze_u_residual_20k.sh
# Dry-run either phase:
#   DRY_RUN=1 bash Disentanglement/blackwell/slurm/qfreeze_u_residual_20k.sh
#   PHASE=probe DRY_RUN=1 bash Disentanglement/blackwell/slurm/qfreeze_u_residual_20k.sh

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
PHASE="${PHASE:-train}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-42}"

FREEZE_STEP=4000
TOTAL_STEPS=20000
DANN_RAMP_STEPS=6800
U_RESIDUAL_WEIGHT=1.0
U_GRL_PRE=0.10
U_PHONE_GRL_PRE=0.05
U_GRL_POST=0.40
U_PHONE_GRL_POST=0.20

BASE_RUN="libri_advlearn_hardqfreeze4000_ures_base_dann6800_gn00015_gp02_gu010_gpu005_aux64_20k_s${SEED}"
RUN_NAME="libri_advlearn_hardqfreeze4000_ures1_postgu040_gpu020_gn0002_dann6800_gp02_aux64_20k_s${SEED}"
BASE_DIR="${RUNS_ROOT}/${BASE_RUN}"
RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
BASE_CKPT="${BASE_DIR}/checkpoints/latest-resume.pt"
FINAL_CKPT="${RUN_DIR}/checkpoints/stage2_step${TOTAL_STEPS}.pt"
JSON_ROOT="${DIS_DIR}/blackwell/logs/qfreeze_u_residual_20k/json/${RUN_NAME}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

mkdir -p "${DIS_DIR}/blackwell/logs" "${JSON_ROOT}" "${RUNS_ROOT}"
cd "${REPO_ROOT}"

run_or_print() {
  echo
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

checkpoint_step() {
  "${PYTHON}" - "$1" <<'PY'
import sys
import torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(payload.get("step", -1)))
PY
}

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -x "${PYTHON}" ]] || { echo "Missing Python: ${PYTHON}" >&2; exit 3; }
  [[ -d "${LIBRISPEECH_ROOT}" ]] || { echo "Missing LibriSpeech: ${LIBRISPEECH_ROOT}" >&2; exit 4; }
  [[ -f "${LEXICON_PATH}" ]] || { echo "Missing lexicon: ${LEXICON_PATH}" >&2; exit 5; }
fi

common_train_args=(
  --stage 2 --stage2_from_scratch
  --local_data --librispeech_root "${LIBRISPEECH_ROOT}"
  --lexicon_path "${LEXICON_PATH}" --train_split_dir train-clean-100
  --speaker_stratified_holdout --spear_layernorm
  --K 5120 --topk 256 --aux_k 64 --aux_k_coef 0.03125
  --dead_steps_threshold 256
  --stage2_steps "${TOTAL_STEPS}" --stage2_schedule_steps "${TOTAL_STEPS}"
  --warmup_steps 500 --batch_size 16 --eval_batch_size 32
  --lr 1e-4 --lr_min 1e-5 --lr_heads 1e-4 --lr_sid_head 0.001
  --grad_clip 1.0 --log_every 100 --grad_log_every 500
  --ckpt_every 1000 --num_workers 2 --seed "${SEED}"
  --n_routes 3 --hard_gumbel_routing
  --gumbel_tau_start 1.0 --gumbel_tau_end 0.1
  --routing_init_std 0.5 --routing_spec_weight 0.01 --lr_routing 1e-3
  --grl_linear_mean --grl_grad_norm
  --alpha 0.8 --beta 0.6 --grl_weight 1.0 --grl_phoneme_weight 0.2
  --grl_delay_steps 0 --dann_full_discriminator
  --dann_ramp_steps "${DANN_RAMP_STEPS}" --lr_disc 1e-3 --n_disc_steps 3
  --rho 0.0 --u_residual_recon_weight "${U_RESIDUAL_WEIGHT}"
)

if [[ "${PHASE}" == "train" ]]; then
  mkdir -p "${BASE_DIR}/checkpoints" "${BASE_DIR}/tensorboard" "${BASE_DIR}/trainer_logs"
  mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/tensorboard" "${RUN_DIR}/trainer_logs"

  echo "=== Libri qfreeze residual-U 20k ==="
  echo "started      : $(date)"
  echo "run          : ${RUN_NAME}"
  echo "schedule     : freeze=${FREEZE_STEP} total=${TOTAL_STEPS} DANN=${DANN_RAMP_STEPS}"
  echo "U objective  : residual weight=${U_RESIDUAL_WEIGHT}"
  echo "U adversary  : pre=${U_GRL_PRE}/${U_PHONE_GRL_PRE} post=${U_GRL_POST}/${U_PHONE_GRL_POST}"

  reuse_base=0
  if [[ "${DRY_RUN}" != "1" && -f "${BASE_CKPT}" ]]; then
    [[ "$(checkpoint_step "${BASE_CKPT}")" == "${FREEZE_STEP}" ]] && reuse_base=1
  fi
  if [[ "${reuse_base}" == "1" ]]; then
    echo "[base] reusing exact 4k residual-U checkpoint: ${BASE_CKPT}"
  else
    run_or_print "${PYTHON}" -u Disentanglement/run.py \
      "${common_train_args[@]}" \
      --grl_grad_norm_target 0.00015 \
      --grl_u_weight "${U_GRL_PRE}" \
      --grl_phoneme_u_weight "${U_PHONE_GRL_PRE}" \
      --segment_steps "${FREEZE_STEP}" --resume_every 500 \
      --checkpoint_dir "${BASE_DIR}/checkpoints" \
      --runs_dir "${BASE_DIR}/tensorboard" \
      --log_dir "${BASE_DIR}/trainer_logs"
  fi

  if [[ "${DRY_RUN}" != "1" ]]; then
    [[ -f "${BASE_CKPT}" ]] || { echo "Missing 4k checkpoint: ${BASE_CKPT}" >&2; exit 6; }
    [[ "$(checkpoint_step "${BASE_CKPT}")" == "${FREEZE_STEP}" ]] || {
      echo "Base checkpoint is not step ${FREEZE_STEP}: ${BASE_CKPT}" >&2; exit 7; }
  fi

  run_or_print "${PYTHON}" -u Disentanglement/run.py \
    "${common_train_args[@]}" \
    --grl_grad_norm_target 0.0002 \
    --grl_u_weight "${U_GRL_POST}" \
    --grl_phoneme_u_weight "${U_PHONE_GRL_POST}" \
    --resume "${BASE_CKPT}" \
    --freeze_learned_routing_on_resume --freeze_route_topk_on_resume \
    --route_topk_calib_batches 20 --resume_every 500 \
    --checkpoint_dir "${RUN_DIR}/checkpoints" \
    --runs_dir "${RUN_DIR}/tensorboard" \
    --log_dir "${RUN_DIR}/trainer_logs"

  echo "finished     : $(date)"
  echo "health gate  : inspect recon, ures, dead units, L/P/U activity and leakage before probing"
  exit 0
fi

if [[ "${PHASE}" != "probe" ]]; then
  echo "Unknown PHASE=${PHASE}; expected train or probe" >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" && ! -f "${FINAL_CKPT}" && -f "${RUN_DIR}/checkpoints/final.pt" ]]; then
  FINAL_CKPT="${RUN_DIR}/checkpoints/final.pt"
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${FINAL_CKPT}" ]] || { echo "Missing healthy final checkpoint: ${FINAL_CKPT}" >&2; exit 8; }
fi

probe_common=(
  --stage2_ckpt "${FINAL_CKPT}" --stage1_ckpt "${FINAL_CKPT}"
  --local_data --librispeech_root "${LIBRISPEECH_ROOT}" --lexicon_path "${LEXICON_PATH}"
  --spear_layernorm --topk 256 --n_routes 3 --hard_gumbel_routing --gumbel_tau_end 0.1
  --probe_steps 5000 --probe_val_every 250 --probe_patience 0
  --probe_warmup_steps 0 --pr_probe_warmup_steps 500
  --seed "${SEED}" --num_workers 8 --no-pr_checkpoint_sanity
  --sources z_t,z_L,z_P,z_U
)

PR_RUN="diag_${RUN_NAME}_pr_linear_seed${SEED}"
SID_RUN="diag_${RUN_NAME}_sid_linear_seed${SEED}"
run_or_print "${PYTHON}" -u Disentanglement/diag_probe/run.py \
  "${probe_common[@]}" --run_name "${PR_RUN}" \
  --output_json "${JSON_ROOT}/${PR_RUN}.json" \
  --tasks pr --pr_probe_arch linear --pr_probe_lr 5e-4 --pr_max_examples 0
run_or_print "${PYTHON}" -u Disentanglement/diag_probe/run.py \
  "${probe_common[@]}" --run_name "${SID_RUN}" \
  --output_json "${JSON_ROOT}/${SID_RUN}.json" \
  --tasks sid --sid_probe_arch linear --sid_probe_lr 1e-3

echo "finished probes: $(date)"
