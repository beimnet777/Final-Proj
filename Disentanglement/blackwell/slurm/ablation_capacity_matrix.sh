#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=05:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=libri_ablate_cap
#SBATCH --array=0-12%4
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%A_%a.err

# Libri content/speaker follow-up matrix.
#
# Array tasks:
#   0  SAE recon-only topk=64    + z_t PR-linear/SID-linear/ASR-LSTM probes
#   1  SAE recon-only topk=128   + z_t PR-linear/SID-linear/ASR-LSTM probes
#   2  SAE recon-only topk=512   + z_t PR-linear/SID-linear/ASR-LSTM probes
#
#   Fixed 240L/16P adversary/GRL ablations; probes are PR-linear + SID-linear only
#   on z_t,z_L,z_P.
#   3  no adversaries
#   4  no speaker adversary
#   5  no phone adversary
#   6  speaker adversary without GRL norm
#   7  stronger speaker GRL norm target: 2e-4
#
#   Learned hard-routing quota-freeze ablations; each task trains its own 4k base,
#   freezes learned route membership + learned route-local TopK quota, then trains
#   to 20k. Probes are PR-linear + SID-linear only on z_t,z_L,z_P.
#   8   no adversaries
#   9   no speaker adversary
#   10  no phone adversary
#   11  speaker adversary without GRL norm
#   12  stronger post-freeze speaker GRL norm target: 3e-4
#
# Submit all, up to four running:
#   sbatch Disentanglement/blackwell/slurm/ablation_capacity_matrix.sh
#
# Submit only a subset, e.g. just SAE capacity:
#   sbatch --array=0-2%3 Disentanglement/blackwell/slurm/ablation_capacity_matrix.sh
#
# Dry-run one local task:
#   DRY_RUN=1 TASK_ID=7 bash Disentanglement/blackwell/slurm/ablation_capacity_matrix.sh

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
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/blackwell_ablation}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

LOG_ROOT="${DIS_DIR}/blackwell/logs/ablation_capacity_matrix"
JSON_ROOT="${LOG_ROOT}/json"
mkdir -p "${LOG_ROOT}" "${JSON_ROOT}" "${RUNS_ROOT}"

cd "${REPO_ROOT}"

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing python: ${PYTHON}" >&2
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
fi

run_or_print() {
  echo
  echo "+ $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_dir() {
  printf '%s/%s\n' "${RUNS_ROOT}" "$1"
}

final_ckpt_for() {
  local run_name="$1"
  local total_steps="$2"
  local ckpt_dir
  ckpt_dir="$(run_dir "${run_name}")/checkpoints"
  if [[ -f "${ckpt_dir}/stage2_step${total_steps}.pt" ]]; then
    printf '%s\n' "${ckpt_dir}/stage2_step${total_steps}.pt"
  else
    printf '%s\n' "${ckpt_dir}/final.pt"
  fi
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
  --aux_k 64
  --aux_k_coef 0.03125
  --dead_steps_threshold 256
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

route_probe_args=()
set_route_probe_args() {
  local route_kind="$1"
  local topk="${2:-256}"
  route_probe_args=(--topk "${topk}")
  case "${route_kind}" in
    plain)
      ;;
    fixed_240_16)
      route_probe_args+=(
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
      route_probe_args+=(
        --n_routes 2
        --hard_gumbel_routing
        --gumbel_tau_end 0.1
      )
      ;;
    *)
      echo "Internal error: unknown route_kind=${route_kind}" >&2
      exit 3
      ;;
  esac
}

probe_pr_linear() {
  local train_run_name="$1"
  local ckpt="$2"
  local route_kind="$3"
  local topk="$4"
  local sources="$5"
  local label="$6"
  set_route_probe_args "${route_kind}" "${topk}"
  local json_dir="${JSON_ROOT}/${train_run_name}"
  mkdir -p "${json_dir}"
  local run_name="diag_${train_run_name}_${label}_pr_linear_seed${SEED}"
  run_or_print "${PYTHON}" -u Disentanglement/diag_probe/run.py \
    --stage2_ckpt "${ckpt}" \
    --stage1_ckpt "${ckpt}" \
    --run_name "${run_name}" \
    --output_json "${json_dir}/${run_name}.json" \
    --local_data \
    --librispeech_root "${LIBRISPEECH_ROOT}" \
    --lexicon_path "${LEXICON_PATH}" \
    --spear_layernorm \
    "${route_probe_args[@]}" \
    --probe_steps 5000 \
    --probe_val_every 250 \
    --probe_patience 0 \
    --probe_warmup_steps 0 \
    --pr_probe_warmup_steps 500 \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --sources "${sources}" \
    --tasks pr \
    --pr_probe_arch linear \
    --pr_probe_lr 5e-4 \
    --pr_max_examples 0 \
    --no-pr_checkpoint_sanity
}

probe_sid_linear() {
  local train_run_name="$1"
  local ckpt="$2"
  local route_kind="$3"
  local topk="$4"
  local sources="$5"
  local label="$6"
  set_route_probe_args "${route_kind}" "${topk}"
  local json_dir="${JSON_ROOT}/${train_run_name}"
  mkdir -p "${json_dir}"
  local run_name="diag_${train_run_name}_${label}_sid_linear_seed${SEED}"
  run_or_print "${PYTHON}" -u Disentanglement/diag_probe/run.py \
    --stage2_ckpt "${ckpt}" \
    --stage1_ckpt "${ckpt}" \
    --run_name "${run_name}" \
    --output_json "${json_dir}/${run_name}.json" \
    --local_data \
    --librispeech_root "${LIBRISPEECH_ROOT}" \
    --lexicon_path "${LEXICON_PATH}" \
    --spear_layernorm \
    "${route_probe_args[@]}" \
    --probe_steps 5000 \
    --probe_val_every 250 \
    --probe_patience 0 \
    --probe_warmup_steps 0 \
    --pr_probe_warmup_steps 500 \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --sources "${sources}" \
    --tasks sid \
    --sid_probe_arch linear \
    --sid_probe_lr 1e-3 \
    --no-pr_checkpoint_sanity
}

probe_asr_lstm() {
  local train_run_name="$1"
  local ckpt="$2"
  local route_kind="$3"
  local topk="$4"
  local sources="$5"
  local label="$6"
  set_route_probe_args "${route_kind}" "${topk}"
  local json_dir="${JSON_ROOT}/${train_run_name}"
  mkdir -p "${json_dir}"
  local run_name="diag_${train_run_name}_${label}_asr_lstm_10k_seed${SEED}"
  run_or_print "${PYTHON}" -u Disentanglement/diag_probe/run.py \
    --stage2_ckpt "${ckpt}" \
    --stage1_ckpt "${ckpt}" \
    --run_name "${run_name}" \
    --output_json "${json_dir}/${run_name}.json" \
    --local_data \
    --librispeech_root "${LIBRISPEECH_ROOT}" \
    --lexicon_path "${LEXICON_PATH}" \
    --spear_layernorm \
    "${route_probe_args[@]}" \
    --probe_steps 10000 \
    --probe_val_every 250 \
    --probe_patience 0 \
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
    --asr_max_examples 0 \
    --no-pr_checkpoint_sanity
}

train_recon_topk() {
  local topk="$1"
  local run_name="libri_sae_recon_topk${topk}_aux64_12k_s${SEED}"
  local rd
  rd="$(run_dir "${run_name}")"
  mkdir -p "${rd}/checkpoints" "${rd}/tensorboard" "${rd}/trainer_logs"
  run_or_print "${PYTHON}" -u Disentanglement/run.py \
    "${common_train_args[@]}" \
    --topk "${topk}" \
    --stage2_steps 12000 \
    --stage2_schedule_steps 12000 \
    --no_routing \
    --alpha 0.0 \
    --beta 0.0 \
    --grl_weight 0.0 \
    --grl_phoneme_weight 0.0 \
    --rho 0.0 \
    --n_disc_steps 1 \
    --checkpoint_dir "${rd}/checkpoints" \
    --runs_dir "${rd}/tensorboard" \
    --log_dir "${rd}/trainer_logs"

  local ckpt
  ckpt="$(final_ckpt_for "${run_name}" 12000)"
  if [[ "${DRY_RUN}" != "1" && ! -f "${ckpt}" ]]; then
    echo "Missing trained checkpoint: ${ckpt}" >&2
    exit 7
  fi
  probe_pr_linear "${run_name}" "${ckpt}" plain "${topk}" z_t zt
  probe_sid_linear "${run_name}" "${ckpt}" plain "${topk}" z_t zt
  probe_asr_lstm "${run_name}" "${ckpt}" plain "${topk}" z_t zt
}

train_fixed_ablation() {
  local run_name="$1"
  local grl_weight="$2"
  local phone_weight="$3"
  local use_norm="$4"
  local gn_target="$5"
  local rd
  rd="$(run_dir "${run_name}")"
  mkdir -p "${rd}/checkpoints" "${rd}/tensorboard" "${rd}/trainer_logs"

  local cmd=(
    "${PYTHON}" -u Disentanglement/run.py
    "${common_train_args[@]}"
    --topk 256
    --stage2_steps 12000
    --stage2_schedule_steps 12000
    --fixed_blocks
    --per_block_topk
    --K_L 4096
    --K_P 1024
    --K_U 0
    --topk_L 240
    --topk_P 16
    --topk_U 0
    --grl_linear_mean
    --alpha 0.8
    --beta 0.6
    --grl_weight "${grl_weight}"
    --grl_phoneme_weight "${phone_weight}"
    --grl_delay_steps 0
    --dann_full_discriminator
    --lr_disc 1e-3
    --n_disc_steps 3
    --rho 0.0
    --checkpoint_dir "${rd}/checkpoints"
    --runs_dir "${rd}/tensorboard"
    --log_dir "${rd}/trainer_logs"
  )
  if [[ "${use_norm}" == "1" ]]; then
    cmd+=(--grl_grad_norm --grl_grad_norm_target "${gn_target}")
  fi
  run_or_print "${cmd[@]}"

  local ckpt
  ckpt="$(final_ckpt_for "${run_name}" 12000)"
  if [[ "${DRY_RUN}" != "1" && ! -f "${ckpt}" ]]; then
    echo "Missing trained checkpoint: ${ckpt}" >&2
    exit 7
  fi
  probe_pr_linear "${run_name}" "${ckpt}" fixed_240_16 256 "z_t,z_L,z_P" zt_zL_zP
  probe_sid_linear "${run_name}" "${ckpt}" fixed_240_16 256 "z_t,z_L,z_P" zt_zL_zP
}

train_learned_qfreeze_ablation() {
  local branch_run="$1"
  local grl_weight="$2"
  local phone_weight="$3"
  local use_norm="$4"
  local base_gn="$5"
  local post_gn="$6"
  local total_steps=20000
  local freeze_step=4000
  local dann_ramp=6800

  local base_run="${branch_run}_base_to_${freeze_step}"
  local base_rd branch_rd route_ckpt
  base_rd="$(run_dir "${base_run}")"
  branch_rd="$(run_dir "${branch_run}")"
  route_ckpt="${base_rd}/checkpoints/latest-resume.pt"
  mkdir -p "${base_rd}/checkpoints" "${base_rd}/tensorboard" "${base_rd}/trainer_logs"
  mkdir -p "${branch_rd}/checkpoints" "${branch_rd}/tensorboard" "${branch_rd}/trainer_logs"

  local base_cmd=(
    "${PYTHON}" -u Disentanglement/run.py
    "${common_train_args[@]}"
    --topk 256
    --stage2_steps "${total_steps}"
    --stage2_schedule_steps "${total_steps}"
    --segment_steps "${freeze_step}"
    --resume_every 500
    --n_routes 2
    --hard_gumbel_routing
    --gumbel_tau_start 1.0
    --gumbel_tau_end 0.1
    --routing_init_std 0.5
    --routing_spec_weight 0.01
    --grl_linear_mean
    --alpha 0.8
    --beta 0.6
    --grl_weight "${grl_weight}"
    --grl_phoneme_weight "${phone_weight}"
    --grl_delay_steps 0
    --dann_full_discriminator
    --dann_ramp_steps "${dann_ramp}"
    --lr_disc 1e-3
    --n_disc_steps 3
    --rho 0.0
    --lr_routing 1e-3
    --checkpoint_dir "${base_rd}/checkpoints"
    --runs_dir "${base_rd}/tensorboard"
    --log_dir "${base_rd}/trainer_logs"
  )
  if [[ "${use_norm}" == "1" ]]; then
    base_cmd+=(--grl_grad_norm --grl_grad_norm_target "${base_gn}")
  fi
  run_or_print "${base_cmd[@]}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ ! -f "${route_ckpt}" ]]; then
      echo "Missing learned-route freeze checkpoint: ${route_ckpt}" >&2
      exit 8
    fi
    local step
    step="$(checkpoint_step "${route_ckpt}")"
    if [[ "${step}" != "${freeze_step}" ]]; then
      echo "Bad freeze checkpoint step=${step}; expected ${freeze_step}: ${route_ckpt}" >&2
      exit 9
    fi
  fi

  local branch_cmd=(
    "${PYTHON}" -u Disentanglement/run.py
    "${common_train_args[@]}"
    --topk 256
    --stage2_steps "${total_steps}"
    --stage2_schedule_steps "${total_steps}"
    --resume "${route_ckpt}"
    --freeze_learned_routing_on_resume
    --freeze_route_topk_on_resume
    --route_topk_calib_batches 20
    --resume_every 500
    --n_routes 2
    --hard_gumbel_routing
    --gumbel_tau_start 1.0
    --gumbel_tau_end 0.1
    --routing_init_std 0.5
    --routing_spec_weight 0.01
    --grl_linear_mean
    --alpha 0.8
    --beta 0.6
    --grl_weight "${grl_weight}"
    --grl_phoneme_weight "${phone_weight}"
    --grl_delay_steps 0
    --dann_full_discriminator
    --dann_ramp_steps "${dann_ramp}"
    --lr_disc 1e-3
    --n_disc_steps 3
    --rho 0.0
    --lr_routing 1e-3
    --checkpoint_dir "${branch_rd}/checkpoints"
    --runs_dir "${branch_rd}/tensorboard"
    --log_dir "${branch_rd}/trainer_logs"
  )
  if [[ "${use_norm}" == "1" ]]; then
    branch_cmd+=(--grl_grad_norm --grl_grad_norm_target "${post_gn}")
  fi
  run_or_print "${branch_cmd[@]}"

  local ckpt
  ckpt="$(final_ckpt_for "${branch_run}" "${total_steps}")"
  if [[ "${DRY_RUN}" != "1" && ! -f "${ckpt}" ]]; then
    echo "Missing trained checkpoint: ${ckpt}" >&2
    exit 7
  fi
  probe_pr_linear "${branch_run}" "${ckpt}" learned_hard_binary 256 "z_t,z_L,z_P" zt_zL_zP
  probe_sid_linear "${branch_run}" "${ckpt}" learned_hard_binary 256 "z_t,z_L,z_P" zt_zL_zP
}

echo "=== Libri ablation/capacity matrix ==="
echo "started        : $(date)"
echo "task_id        : ${TASK_ID}"
echo "repo_root      : ${REPO_ROOT}"
echo "runs_root      : ${RUNS_ROOT}"
echo "log_root       : ${LOG_ROOT}"
echo "dry_run        : ${DRY_RUN}"
echo "python         : ${PYTHON}"
echo "gpu            : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"

case "${TASK_ID}" in
  0)  train_recon_topk 64 ;;
  1)  train_recon_topk 128 ;;
  2)  train_recon_topk 512 ;;

  3)  train_fixed_ablation "libri_advfb_ab_noadv_240L16P_aux64_12k_s${SEED}" 0.0 0.0 0 0.0 ;;
  4)  train_fixed_ablation "libri_advfb_ab_nospkadv_gp02_240L16P_aux64_12k_s${SEED}" 0.0 0.2 0 0.0 ;;
  5)  train_fixed_ablation "libri_advfb_ab_nophoneadv_gn00015_240L16P_aux64_12k_s${SEED}" 1.0 0.0 1 0.00015 ;;
  6)  train_fixed_ablation "libri_advfb_ab_nognorm_gp02_240L16P_aux64_12k_s${SEED}" 1.0 0.2 0 0.0 ;;
  7)  train_fixed_ablation "libri_advfb_ab_gn0002_gp02_240L16P_aux64_12k_s${SEED}" 1.0 0.2 1 0.0002 ;;

  8)  train_learned_qfreeze_ablation "libri_advlearn_hardqfreeze4000_ab_noadv_aux64_20k_s${SEED}" 0.0 0.0 0 0.0 0.0 ;;
  9)  train_learned_qfreeze_ablation "libri_advlearn_hardqfreeze4000_ab_nospkadv_gp02_aux64_20k_s${SEED}" 0.0 0.2 0 0.0 0.0 ;;
  10) train_learned_qfreeze_ablation "libri_advlearn_hardqfreeze4000_ab_nophoneadv_gn0002_dann6800_aux64_20k_s${SEED}" 1.0 0.0 1 0.00015 0.0002 ;;
  11) train_learned_qfreeze_ablation "libri_advlearn_hardqfreeze4000_ab_nognorm_gp02_aux64_20k_s${SEED}" 1.0 0.2 0 0.0 0.0 ;;
  12) train_learned_qfreeze_ablation "libri_advlearn_hardqfreeze4000_ab_gn0003_dann6800_gp02_aux64_20k_s${SEED}" 1.0 0.2 1 0.00015 0.0003 ;;

  *)
    echo "Unknown TASK_ID=${TASK_ID}; expected 0..12." >&2
    exit 2
    ;;
esac

echo "finished       : $(date)"
