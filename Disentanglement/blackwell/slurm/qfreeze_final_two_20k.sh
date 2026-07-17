#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=libri_qf_final2
#SBATCH --array=0-1%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%A_%a.err

# Final two Libri learned-qfreeze follow-ups.
#
# Task 0: post-freeze phone cleanup only
#   Train/reuse a gp=0.2 / DANN=6.8k base checkpoint to the 4k freeze point
#   inside this HPC runs directory. Freeze learned routes + quota from that
#   checkpoint, then change ONLY:
#       grl_phoneme_weight: 0.2 -> 0.3
#   Keep speaker GRL norm target at gn=0.0002 post-freeze.
#   Probes: ASR LSTM, PR linear, SID linear, SID stats on z_t,z_L,z_P.
#
# Task 1: U weak-before-freeze -> normal-after-freeze
#   Train a 3-route base to 4k with weak U adversaries.
#   Freeze routes + quota, then continue to 20k with U adversaries:
#       grl_u_weight=0.4, grl_phoneme_u_weight=0.2
#   Probes: PR linear + SID linear on z_t,z_L,z_P,z_U. No stat probes.
#
# Submit both:
#   sbatch Disentanglement/blackwell/slurm/qfreeze_final_two_20k.sh
#
# Submit one:
#   sbatch --array=0 Disentanglement/blackwell/slurm/qfreeze_final_two_20k.sh
#   sbatch --array=1 Disentanglement/blackwell/slurm/qfreeze_final_two_20k.sh
#
# Dry-run:
#   SLURM_ARRAY_TASK_ID=0 DRY_RUN=1 bash Disentanglement/blackwell/slurm/qfreeze_final_two_20k.sh
#   SLURM_ARRAY_TASK_ID=1 DRY_RUN=1 bash Disentanglement/blackwell/slurm/qfreeze_final_two_20k.sh
#
# Git artifact policy for this script:
#   - logs are auto-added/committed/pushed at the end of each array task
#   - checkpoints are also auto-added here because this run explicitly requests
#     checkpoint tracking
#   - git save is best-effort and locked, so a git/network failure will not mark
#     the completed experiment as failed

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module purge 2>/dev/null || true
  module load rhel8/default-amp 2>/dev/null || true
fi

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
DEFAULT_REPO_ROOT="/rds/user/bbg25/hpc-work/Thesis/Final-Proj"
if [[ -z "${REPO_ROOT:-}" ]]; then
  if [[ -d "${DEFAULT_REPO_ROOT}" ]]; then
    REPO_ROOT="${DEFAULT_REPO_ROOT}"
  else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
  fi
fi
DIS_DIR="${DIS_DIR:-${REPO_ROOT}/Disentanglement}"
LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-${REPO_ROOT}/Probing/data/LibriSpeech}"
LEXICON_PATH="${LEXICON_PATH:-${REPO_ROOT}/Probing/data/librispeech-lexicon.txt}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/blackwell_followups}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
AUTO_GIT="${AUTO_GIT:-1}"
AUTO_GIT_PUSH="${AUTO_GIT_PUSH:-1}"
TRACK_CHECKPOINTS="${TRACK_CHECKPOINTS:-1}"

FREEZE_STEP="${FREEZE_STEP:-4000}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
DANN_RAMP_STEPS="${DANN_RAMP_STEPS:-6800}"
BASE_GRL_TARGET="${BASE_GRL_TARGET:-0.00015}"
POST_GRL_TARGET="${POST_GRL_TARGET:-0.0002}"
ROUTE_TOPK_CALIB_BATCHES="${ROUTE_TOPK_CALIB_BATCHES:-20}"

PR_SID_STEPS="${PR_SID_STEPS:-5000}"
ASR_STEPS="${ASR_STEPS:-10000}"
PROBE_VAL_EVERY="${PROBE_VAL_EVERY:-250}"
PROBE_PATIENCE="${PROBE_PATIENCE:-0}"

# U schedule for task 1.
U_GRL_PRE="${U_GRL_PRE:-0.10}"
U_PHONEME_GRL_PRE="${U_PHONEME_GRL_PRE:-0.05}"
U_GRL_POST="${U_GRL_POST:-0.40}"
U_PHONEME_GRL_POST="${U_PHONEME_GRL_POST:-0.20}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

LOG_ROOT="${DIS_DIR}/blackwell/logs/qfreeze_final_two_20k"
mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"

cd "${REPO_ROOT}"

run_or_print() {
  echo
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

git_paths=()

stage_if_exists() {
  local p="$1"
  [[ -n "${p}" && -e "${p}" ]] || return 0
  case "${p}" in
    "${REPO_ROOT}"/*)
      git_paths+=("${p#${REPO_ROOT}/}")
      ;;
    *)
      echo "[git] skip non-repo artifact: ${p}"
      ;;
  esac
}

stage_slurm_logs() {
  local job_name="${SLURM_JOB_NAME:-libri_qf_final2}"
  local array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"
  local array_task_id="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
  if [[ -n "${array_job_id}" ]]; then
    stage_if_exists "${DIS_DIR}/blackwell/logs/${job_name}_${array_job_id}_${array_task_id}.out"
    stage_if_exists "${DIS_DIR}/blackwell/logs/${job_name}_${array_job_id}_${array_task_id}.err"
  fi
}

stage_checkpoint_subset() {
  local ckpt_dir="$1"
  local final_step="$2"
  [[ -d "${ckpt_dir}" ]] || return 0
  stage_if_exists "${ckpt_dir}/metrics.jsonl"
  stage_if_exists "${ckpt_dir}/final.pt"
  stage_if_exists "${ckpt_dir}/stage2_best.pt"
  stage_if_exists "${ckpt_dir}/latest-resume.pt"
  stage_if_exists "${ckpt_dir}/stage2_step${final_step}.pt"
}

copy_checkpoint_subset() {
  local src_dir="$1"
  local dst_dir="$2"
  local final_step="$3"
  [[ -d "${src_dir}" ]] || return 0
  mkdir -p "${dst_dir}"
  local f
  for f in \
    "metrics.jsonl" \
    "final.pt" \
    "stage2_best.pt" \
    "latest-resume.pt" \
    "stage2_step${final_step}.pt"
  do
    if [[ -f "${src_dir}/${f}" ]]; then
      cp -p "${src_dir}/${f}" "${dst_dir}/${f}"
    fi
  done
}

archive_tracked_checkpoints() {
  [[ "${TRACK_CHECKPOINTS}" == "1" ]] || return 0
  copy_checkpoint_subset \
    "${run_dir:-}/checkpoints" \
    "${REPO_ROOT}/checkpoints/blackwell/${run_name:-unknown}" \
    "${TOTAL_STEPS}"
  if [[ -n "${base_run:-}" ]]; then
    copy_checkpoint_subset \
      "${base_dir:-}/checkpoints" \
      "${REPO_ROOT}/checkpoints/blackwell/${base_run}" \
      "${FREEZE_STEP}"
  fi
}

auto_git_save() {
  local status_label="${1:-complete}"
  [[ "${DRY_RUN}" == "1" ]] && return 0
  [[ "${AUTO_GIT}" == "1" ]] || return 0
  git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "[git] not a git worktree, skipping artifact save"
    return 0
  }

  local lock_dir="${REPO_ROOT}/.git/qfreeze_final_two_20k.lock"
  local waited=0
  while ! mkdir "${lock_dir}" 2>/dev/null; do
    waited=$((waited + 5))
    if [[ "${waited}" -gt 900 ]]; then
      echo "[git] timed out waiting for git lock, skipping artifact save"
      return 0
    fi
    sleep 5
  done

  (
    trap 'rmdir "'"${lock_dir}"'" 2>/dev/null || true' EXIT
    git_paths=()
    archive_tracked_checkpoints

    stage_if_exists "${DIS_DIR}/blackwell/slurm/qfreeze_final_two_20k.sh"
    stage_slurm_logs
    stage_if_exists "${json_root:-}"
    stage_if_exists "${run_dir:-}/trainer_logs"
    stage_if_exists "${base_dir:-}/trainer_logs"

    if [[ "${TRACK_CHECKPOINTS}" == "1" ]]; then
      stage_checkpoint_subset "${REPO_ROOT}/checkpoints/blackwell/${run_name:-unknown}" "${TOTAL_STEPS}"
      if [[ -n "${base_dir:-}" ]]; then
        stage_checkpoint_subset "${REPO_ROOT}/checkpoints/blackwell/${base_run:-unknown}" "${FREEZE_STEP}"
      fi
    else
      stage_if_exists "${run_dir:-}/checkpoints/metrics.jsonl"
      stage_if_exists "${base_dir:-}/checkpoints/metrics.jsonl"
    fi

    if [[ "${#git_paths[@]}" -eq 0 ]]; then
      echo "[git] no artifacts found to stage"
      return 0
    fi

    echo "[git] staging ${#git_paths[@]} artifact paths"
    git -C "${REPO_ROOT}" add -f -- "${git_paths[@]}" || {
      echo "[git] add failed; skipping commit/push"
      return 0
    }

    if git -C "${REPO_ROOT}" diff --cached --quiet -- "${git_paths[@]}"; then
      echo "[git] no staged changes for ${run_name:-unknown}"
      return 0
    fi

    local msg="Add qfreeze final2 ${status_label} artifacts: task ${TASK_ID} ${run_name:-unknown}"
    git -C "${REPO_ROOT}" commit -m "${msg}" || {
      echo "[git] commit failed; artifacts remain staged"
      return 0
    }

    if [[ "${AUTO_GIT_PUSH}" == "1" ]]; then
      git -C "${REPO_ROOT}" push || {
        echo "[git] push failed; commit was created locally"
        return 0
      }
    fi
  )
}

on_exit_save_logs() {
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    echo "[job] exiting with rc=${rc}; attempting to save logs/artifacts"
    auto_git_save "failed"
  fi
  exit "${rc}"
}

trap on_exit_save_logs EXIT

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

learned_args() {
  local n_routes="$1"
  local gp="$2"
  local gu="$3"
  local gpu="$4"
  printf '%s\0' \
    --n_routes "${n_routes}" \
    --hard_gumbel_routing \
    --gumbel_tau_start 1.0 \
    --gumbel_tau_end 0.1 \
    --routing_init_std 0.5 \
    --routing_spec_weight 0.01 \
    --grl_linear_mean \
    --grl_grad_norm \
    --alpha 0.8 \
    --beta 0.6 \
    --grl_weight 1.0 \
    --grl_phoneme_weight "${gp}" \
    --grl_u_weight "${gu}" \
    --grl_phoneme_u_weight "${gpu}" \
    --grl_delay_steps 0 \
    --dann_full_discriminator \
    --dann_ramp_steps "${DANN_RAMP_STEPS}" \
    --lr_disc 1e-3 \
    --n_disc_steps 3 \
    --rho 0.0 \
    --lr_routing 1e-3
}

read_null_array() {
  local _out_name="$1"
  local item
  eval "${_out_name}=()"
  while IFS= read -r -d '' item; do
    eval "${_out_name}+=(\"\${item}\")"
  done
}

probe_route_args=()
probe_sources=""
run_name=""
json_root=""

run_pr_probe() {
  local ckpt="$1"
  local sources="$2"
  local label="$3"
  local run_id="diag_${run_name}_${label}_pr_linear_seed${SEED}"
  run_or_print "${PYTHON}" -u Disentanglement/diag_probe/run.py \
    --stage2_ckpt "${ckpt}" \
    --stage1_ckpt "${ckpt}" \
    --run_name "${run_id}" \
    --output_json "${json_root}/${run_id}.json" \
    --local_data \
    --librispeech_root "${LIBRISPEECH_ROOT}" \
    --lexicon_path "${LEXICON_PATH}" \
    --spear_layernorm \
    --topk 256 \
    "${probe_route_args[@]}" \
    --probe_steps "${PR_SID_STEPS}" \
    --probe_val_every "${PROBE_VAL_EVERY}" \
    --probe_patience "${PROBE_PATIENCE}" \
    --probe_warmup_steps 0 \
    --pr_probe_warmup_steps 500 \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --no-pr_checkpoint_sanity \
    --sources "${sources}" \
    --tasks pr \
    --pr_probe_arch linear \
    --pr_probe_lr 5e-4 \
    --pr_max_examples 0
}

run_sid_probe() {
  local ckpt="$1"
  local sources="$2"
  local label="$3"
  local arch="$4"
  local run_id="diag_${run_name}_${label}_sid_${arch}_seed${SEED}"
  run_or_print "${PYTHON}" -u Disentanglement/diag_probe/run.py \
    --stage2_ckpt "${ckpt}" \
    --stage1_ckpt "${ckpt}" \
    --run_name "${run_id}" \
    --output_json "${json_root}/${run_id}.json" \
    --local_data \
    --librispeech_root "${LIBRISPEECH_ROOT}" \
    --lexicon_path "${LEXICON_PATH}" \
    --spear_layernorm \
    --topk 256 \
    "${probe_route_args[@]}" \
    --probe_steps "${PR_SID_STEPS}" \
    --probe_val_every "${PROBE_VAL_EVERY}" \
    --probe_patience "${PROBE_PATIENCE}" \
    --probe_warmup_steps 0 \
    --pr_probe_warmup_steps 500 \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --no-pr_checkpoint_sanity \
    --sources "${sources}" \
    --tasks sid \
    --sid_probe_arch "${arch}" \
    --sid_probe_lr 1e-3
}

run_asr_probe() {
  local ckpt="$1"
  local sources="$2"
  local label="$3"
  local run_id="diag_${run_name}_${label}_asr_lstm_seed${SEED}"
  run_or_print "${PYTHON}" -u Disentanglement/diag_probe/run.py \
    --stage2_ckpt "${ckpt}" \
    --stage1_ckpt "${ckpt}" \
    --run_name "${run_id}" \
    --output_json "${json_root}/${run_id}.json" \
    --local_data \
    --librispeech_root "${LIBRISPEECH_ROOT}" \
    --lexicon_path "${LEXICON_PATH}" \
    --spear_layernorm \
    --topk 256 \
    "${probe_route_args[@]}" \
    --probe_steps "${ASR_STEPS}" \
    --probe_val_every "${PROBE_VAL_EVERY}" \
    --probe_patience "${PROBE_PATIENCE}" \
    --probe_warmup_steps 0 \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --sources "${sources}" \
    --tasks asr \
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
}

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -x "${PYTHON}" ]] || { echo "Missing python: ${PYTHON}" >&2; exit 4; }
  [[ -d "${LIBRISPEECH_ROOT}" ]] || { echo "Missing LibriSpeech root: ${LIBRISPEECH_ROOT}" >&2; exit 5; }
  [[ -f "${LEXICON_PATH}" ]] || { echo "Missing lexicon: ${LEXICON_PATH}" >&2; exit 6; }
fi

case "${TASK_ID}" in
  0)
    case_name="postgp030"
    n_routes=2
    gp_pre="0.2"
    gp_post="0.3"
    gu_pre="0.0"
    gpu_pre="0.0"
    gu_post="0.0"
    gpu_post="0.0"
    base_run="libri_advlearn_hardqfreeze${FREEZE_STEP}_base_dann${DANN_RAMP_STEPS}_gn00015_gp02_aux64_20k_s${SEED}"
    if [[ -n "${GP02_BASE_CKPT:-}" ]]; then
      base_ckpt="${GP02_BASE_CKPT}"
      base_run=""
    else
      base_ckpt="${RUNS_ROOT}/${base_run}/checkpoints/latest-resume.pt"
    fi
    run_name="libri_advlearn_hardqfreeze${FREEZE_STEP}_postgp030_fromgp02base_dann${DANN_RAMP_STEPS}_gn0002_aux64_20k_s${SEED}"
    probe_sources="z_t,z_L,z_P"
    probe_route_args=(--n_routes 2 --hard_gumbel_routing --gumbel_tau_end 0.1)
    ;;
  1)
    case_name="uweak_postu040_020"
    n_routes=3
    gp_pre="0.2"
    gp_post="0.2"
    gu_pre="${U_GRL_PRE}"
    gpu_pre="${U_PHONEME_GRL_PRE}"
    gu_post="${U_GRL_POST}"
    gpu_post="${U_PHONEME_GRL_POST}"
    base_run="libri_advlearn_hardqfreeze${FREEZE_STEP}_uweak_base_dann${DANN_RAMP_STEPS}_gn00015_gp02_gu010_gpu005_aux64_20k_s${SEED}"
    base_ckpt="${RUNS_ROOT}/${base_run}/checkpoints/latest-resume.pt"
    run_name="libri_advlearn_hardqfreeze${FREEZE_STEP}_uweak_postgu040_gpu020_gn0002_dann${DANN_RAMP_STEPS}_gp02_aux64_20k_s${SEED}"
    probe_sources="z_t,z_L,z_P,z_U"
    probe_route_args=(--n_routes 3 --hard_gumbel_routing --gumbel_tau_end 0.1)
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0 or 1" >&2
    exit 2
    ;;
esac

json_root="${LOG_ROOT}/json/${run_name}"
mkdir -p "${json_root}"

run_dir="${RUNS_ROOT}/${run_name}"
mkdir -p "${run_dir}/checkpoints" "${run_dir}/tensorboard" "${run_dir}/trainer_logs"

echo "=== Libri qfreeze final two ==="
echo "started       : $(date)"
echo "task_id       : ${TASK_ID}"
echo "case          : ${case_name}"
echo "run_name      : ${run_name}"
echo "freeze_step   : ${FREEZE_STEP}"
echo "total_steps   : ${TOTAL_STEPS}"
echo "dann_ramp     : ${DANN_RAMP_STEPS}"
echo "speaker gn    : base=${BASE_GRL_TARGET}, post=${POST_GRL_TARGET}"
echo "gp pre/post   : ${gp_pre}/${gp_post}"
echo "U pre/post    : speaker=${gu_pre}/${gu_post}, phone=${gpu_pre}/${gpu_post}"
echo "base_ckpt     : ${base_ckpt}"
echo "probe sources : ${probe_sources}"
echo "runs_root     : ${RUNS_ROOT}"
echo "json_root     : ${json_root}"
echo "dry_run       : ${DRY_RUN}"
echo "gpu           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

if [[ -n "${base_run:-}" ]]; then
  base_dir="${RUNS_ROOT}/${base_run}"
  mkdir -p "${base_dir}/checkpoints" "${base_dir}/tensorboard" "${base_dir}/trainer_logs"

  if [[ "${DRY_RUN}" != "1" && -f "${base_ckpt}" ]]; then
    base_step="$(checkpoint_step "${base_ckpt}")"
  else
    base_step=""
  fi

  if [[ "${base_step}" == "${FREEZE_STEP}" ]]; then
    echo "[base] reusing existing freeze checkpoint: ${base_ckpt}"
  else
    echo "[base] training routing base to step ${FREEZE_STEP}"
    pre_args=()
    read_null_array pre_args < <(learned_args "${n_routes}" "${gp_pre}" "${gu_pre}" "${gpu_pre}")
    run_or_print "${PYTHON}" -u Disentanglement/run.py \
      "${common_train_args[@]}" \
      "${pre_args[@]}" \
      --grl_grad_norm_target "${BASE_GRL_TARGET}" \
      --segment_steps "${FREEZE_STEP}" \
      --resume_every 500 \
      --checkpoint_dir "${base_dir}/checkpoints" \
      --runs_dir "${base_dir}/tensorboard" \
      --log_dir "${base_dir}/trainer_logs"

    if [[ "${DRY_RUN}" != "1" ]]; then
      [[ -f "${base_ckpt}" ]] || { echo "Missing trained base checkpoint: ${base_ckpt}" >&2; exit 7; }
      base_step="$(checkpoint_step "${base_ckpt}")"
      [[ "${base_step}" == "${FREEZE_STEP}" ]] || {
        echo "Bad trained base checkpoint step=${base_step}; expected ${FREEZE_STEP}: ${base_ckpt}" >&2
        exit 8
      }
    fi
  fi
else
  if [[ "${DRY_RUN}" != "1" ]]; then
    [[ -f "${base_ckpt}" ]] || {
      echo "Missing external base checkpoint from GP02_BASE_CKPT: ${base_ckpt}" >&2
      exit 7
    }
    base_step="$(checkpoint_step "${base_ckpt}")"
    [[ "${base_step}" == "${FREEZE_STEP}" ]] || {
      echo "Bad reused gp02 base checkpoint step=${base_step}; expected ${FREEZE_STEP}: ${base_ckpt}" >&2
      exit 8
    }
  fi
fi

echo "[branch] freeze route membership + route-local TopK quota; continue to ${TOTAL_STEPS}"
post_args=()
read_null_array post_args < <(learned_args "${n_routes}" "${gp_post}" "${gu_post}" "${gpu_post}")
run_or_print "${PYTHON}" -u Disentanglement/run.py \
  "${common_train_args[@]}" \
  "${post_args[@]}" \
  --grl_grad_norm_target "${POST_GRL_TARGET}" \
  --resume "${base_ckpt}" \
  --freeze_learned_routing_on_resume \
  --freeze_route_topk_on_resume \
  --route_topk_calib_batches "${ROUTE_TOPK_CALIB_BATCHES}" \
  --resume_every 500 \
  --checkpoint_dir "${run_dir}/checkpoints" \
  --runs_dir "${run_dir}/tensorboard" \
  --log_dir "${run_dir}/trainer_logs"

final_ckpt="$(final_ckpt_for "${run_name}")"
if [[ "${DRY_RUN}" != "1" && ! -f "${final_ckpt}" ]]; then
  echo "Missing final checkpoint: ${final_ckpt}" >&2
  exit 9
fi

echo "[probe] PR linear ${probe_sources}"
run_pr_probe "${final_ckpt}" "${probe_sources}" "${case_name}"

echo "[probe] SID linear ${probe_sources}"
run_sid_probe "${final_ckpt}" "${probe_sources}" "${case_name}" "linear"

if [[ "${TASK_ID}" == "0" ]]; then
  echo "[probe] SID stats ${probe_sources}"
  run_sid_probe "${final_ckpt}" "${probe_sources}" "${case_name}" "stats"

  echo "[probe] ASR LSTM ${probe_sources}"
  run_asr_probe "${final_ckpt}" "${probe_sources}" "${case_name}"
fi

echo "finished      : $(date)"
trap - EXIT
auto_git_save "complete"
