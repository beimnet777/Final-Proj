#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=libri_qf_gp035
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.err

# Focused learned-routing follow-up:
#   - hard learned routing
#   - learn routes until step 4000
#   - freeze learned route membership
#   - calibrate/freeze learned route-local TopK quota
#   - continue to 20000 total steps
#   - same as best qfreeze-20k, except stronger phone/content adversary:
#       grl_phoneme_weight = 0.3  (was 0.20)
#
# After training, run only the directly relevant frozen probes:
#   - PR linear on z_t,z_L,z_P
#   - SID linear on z_t,z_L,z_P
#
# Submit:
#   sbatch Disentanglement/blackwell/slurm/qfreeze_gp035_20k.sh
#
# Dry-run:
#   DRY_RUN=1 bash Disentanglement/blackwell/slurm/qfreeze_gp035_20k.sh

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

FREEZE_STEP="${FREEZE_STEP:-4000}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
DANN_RAMP_STEPS="${DANN_RAMP_STEPS:-6800}"
BASE_GRL_TARGET="${BASE_GRL_TARGET:-0.00015}"
POST_GRL_TARGET="${POST_GRL_TARGET:-0.0002}"
PHONEME_GRL_WEIGHT="${PHONEME_GRL_WEIGHT:-0.3}"
ROUTE_TOPK_CALIB_BATCHES="${ROUTE_TOPK_CALIB_BATCHES:-20}"

PROBE_STEPS="${PROBE_STEPS:-5000}"
PROBE_VAL_EVERY="${PROBE_VAL_EVERY:-250}"
PROBE_PATIENCE="${PROBE_PATIENCE:-0}"

BASE_RUN="libri_advlearn_hardqfreeze${FREEZE_STEP}_base_dann${DANN_RAMP_STEPS}_gn00015_gp035_aux64_20k_s${SEED}"
RUN_NAME="libri_advlearn_hardqfreeze${FREEZE_STEP}_gn0002_dann${DANN_RAMP_STEPS}_gp035_aux64_20k_s${SEED}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

LOG_ROOT="${DIS_DIR}/blackwell/logs/qfreeze_gp035_20k"
JSON_ROOT="${LOG_ROOT}/json/${RUN_NAME}"
mkdir -p "${LOG_ROOT}" "${JSON_ROOT}" "${RUNS_ROOT}"

cd "${REPO_ROOT}"

run_or_print() {
  echo
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

checkpoint_step() {
  local ckpt="$1"
  "${PYTHON}" - "$ckpt" <<'PY'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ckpt.get("step", -1)))
PY
}

final_ckpt_for() {
  local run_name="$1"
  local ckpt_dir="${RUNS_ROOT}/${run_name}/checkpoints"
  local ckpt="${ckpt_dir}/stage2_step${TOTAL_STEPS}.pt"
  if [[ ! -f "${ckpt}" && -f "${ckpt_dir}/final.pt" ]]; then
    ckpt="${ckpt_dir}/final.pt"
  fi
  printf '%s\n' "${ckpt}"
}

common_train_args=(
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
  --aux_k 64
  --aux_k_coef 0.03125
  --dead_steps_threshold 256
  --stage2_steps "${TOTAL_STEPS}"
  --stage2_schedule_steps "${TOTAL_STEPS}"
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
)

learned_common_args=(
  --n_routes 2
  --hard_gumbel_routing
  --gumbel_tau_start 1.0
  --gumbel_tau_end 0.1
  --routing_init_std 0.5
  --routing_spec_weight 0.01
  --grl_linear_mean
  --grl_grad_norm
  --alpha 0.8
  --beta 0.6
  --grl_weight 1.0
  --grl_phoneme_weight "${PHONEME_GRL_WEIGHT}"
  --grl_delay_steps 0
  --dann_full_discriminator
  --dann_ramp_steps "${DANN_RAMP_STEPS}"
  --lr_disc 1e-3
  --n_disc_steps 3
  --rho 0.0
  --lr_routing 1e-3
)

probe_route_args=(
  --n_routes 2
  --hard_gumbel_routing
  --gumbel_tau_end 0.1
)

probe_common_args=(
  --local_data
  --librispeech_root "${LIBRISPEECH_ROOT}"
  --lexicon_path "${LEXICON_PATH}"
  --spear_layernorm
  --topk 256
  "${probe_route_args[@]}"
  --probe_steps "${PROBE_STEPS}"
  --probe_val_every "${PROBE_VAL_EVERY}"
  --probe_patience "${PROBE_PATIENCE}"
  --probe_warmup_steps 0
  --pr_probe_warmup_steps 500
  --seed "${SEED}"
  --num_workers "${NUM_WORKERS}"
  --no-pr_checkpoint_sanity
)

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -x "${PYTHON}" ]] || { echo "Missing python: ${PYTHON}" >&2; exit 4; }
  [[ -d "${LIBRISPEECH_ROOT}" ]] || { echo "Missing LibriSpeech root: ${LIBRISPEECH_ROOT}" >&2; exit 5; }
  [[ -f "${LEXICON_PATH}" ]] || { echo "Missing lexicon: ${LEXICON_PATH}" >&2; exit 6; }
fi

echo "=== Libri qfreeze gp035 20k ==="
echo "started       : $(date)"
echo "run_name      : ${RUN_NAME}"
echo "base_run      : ${BASE_RUN}"
echo "freeze_step   : ${FREEZE_STEP}"
echo "total_steps   : ${TOTAL_STEPS}"
echo "dann_ramp     : ${DANN_RAMP_STEPS}"
echo "speaker gn    : base=${BASE_GRL_TARGET}, post=${POST_GRL_TARGET}"
echo "grl_p weight  : ${PHONEME_GRL_WEIGHT}"
echo "runs_root     : ${RUNS_ROOT}"
echo "json_root     : ${JSON_ROOT}"
echo "dry_run       : ${DRY_RUN}"
echo "gpu           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

BASE_DIR="${RUNS_ROOT}/${BASE_RUN}"
RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
mkdir -p "${BASE_DIR}/checkpoints" "${BASE_DIR}/tensorboard" "${BASE_DIR}/trainer_logs"
mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/tensorboard" "${RUN_DIR}/trainer_logs"

BASE_CKPT="${BASE_DIR}/checkpoints/latest-resume.pt"

if [[ "${DRY_RUN}" != "1" && -f "${BASE_CKPT}" ]]; then
  BASE_STEP="$(checkpoint_step "${BASE_CKPT}")"
else
  BASE_STEP=""
fi

if [[ "${BASE_STEP}" == "${FREEZE_STEP}" ]]; then
  echo "[base] reusing existing freeze checkpoint: ${BASE_CKPT}"
else
  echo "[base] training learned routing to step ${FREEZE_STEP}"
  run_or_print "${PYTHON}" -u Disentanglement/run.py \
    "${common_train_args[@]}" \
    "${learned_common_args[@]}" \
    --grl_grad_norm_target "${BASE_GRL_TARGET}" \
    --segment_steps "${FREEZE_STEP}" \
    --resume_every 500 \
    --checkpoint_dir "${BASE_DIR}/checkpoints" \
    --runs_dir "${BASE_DIR}/tensorboard" \
    --log_dir "${BASE_DIR}/trainer_logs"

  if [[ "${DRY_RUN}" != "1" ]]; then
    [[ -f "${BASE_CKPT}" ]] || { echo "Missing base checkpoint: ${BASE_CKPT}" >&2; exit 7; }
    BASE_STEP="$(checkpoint_step "${BASE_CKPT}")"
    [[ "${BASE_STEP}" == "${FREEZE_STEP}" ]] || {
      echo "Bad base checkpoint step=${BASE_STEP}; expected ${FREEZE_STEP}: ${BASE_CKPT}" >&2
      exit 8
    }
  fi
fi

echo "[branch] freeze route membership + route-local TopK quota; continue to ${TOTAL_STEPS}"
run_or_print "${PYTHON}" -u Disentanglement/run.py \
  "${common_train_args[@]}" \
  "${learned_common_args[@]}" \
  --grl_grad_norm_target "${POST_GRL_TARGET}" \
  --resume "${BASE_CKPT}" \
  --freeze_learned_routing_on_resume \
  --freeze_route_topk_on_resume \
  --route_topk_calib_batches "${ROUTE_TOPK_CALIB_BATCHES}" \
  --resume_every 500 \
  --checkpoint_dir "${RUN_DIR}/checkpoints" \
  --runs_dir "${RUN_DIR}/tensorboard" \
  --log_dir "${RUN_DIR}/trainer_logs"

FINAL_CKPT="$(final_ckpt_for "${RUN_NAME}")"
if [[ "${DRY_RUN}" != "1" && ! -f "${FINAL_CKPT}" ]]; then
  echo "Missing final checkpoint: ${FINAL_CKPT}" >&2
  exit 9
fi

PR_RUN="diag_${RUN_NAME}_zt_zL_zP_pr_linear_seed${SEED}"
SID_RUN="diag_${RUN_NAME}_zt_zL_zP_sid_linear_seed${SEED}"

echo "[probe] PR linear z_t,z_L,z_P"
run_or_print "${PYTHON}" -u Disentanglement/diag_probe/run.py \
  --stage2_ckpt "${FINAL_CKPT}" \
  --stage1_ckpt "${FINAL_CKPT}" \
  --run_name "${PR_RUN}" \
  --output_json "${JSON_ROOT}/${PR_RUN}.json" \
  "${probe_common_args[@]}" \
  --sources "z_t,z_L,z_P" \
  --tasks pr \
  --pr_probe_arch linear \
  --pr_probe_lr 5e-4 \
  --pr_max_examples 0

echo "[probe] SID linear z_t,z_L,z_P"
run_or_print "${PYTHON}" -u Disentanglement/diag_probe/run.py \
  --stage2_ckpt "${FINAL_CKPT}" \
  --stage1_ckpt "${FINAL_CKPT}" \
  --run_name "${SID_RUN}" \
  --output_json "${JSON_ROOT}/${SID_RUN}.json" \
  "${probe_common_args[@]}" \
  --sources "z_t,z_L,z_P" \
  --tasks sid \
  --sid_probe_arch linear \
  --sid_probe_lr 1e-3

echo "finished      : $(date)"
