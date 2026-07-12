#!/usr/bin/env bash
# Final learned-routing quota-freeze search batch.
#
# This is the last GPU-window style suite: deliberately small, sequential, and
# focused on the 4k learned-route quota freeze because the 5k/6k quota runs leaked
# badly.  All branches:
#
#   - learn hard routing until step 4000
#   - freeze learned route membership
#   - calibrate/enforce the learned route-local TopK quota at freeze time
#   - run linear-only PR and linear-only SID probes on z_t/z_L/z_P
#
# Branches:
#   1) 16k total, no reset, post-freeze z_L GRL norm = 2e-4
#   2) 16k total, reset only z_L speaker adversary, delay GRL reverse push to 4500
#   3) 16k total, reset only z_L speaker adversary, no delay
#   4) 20k total, no reset, post-freeze z_L GRL norm = 2e-4
#
# Intentionally NOT run here:
#   - PR direct probes
#   - SID stats probes
#   - full adversary reset (it killed recon/dead units)
#   - post-freeze 2.5e-4 before seeing 2e-4
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SEED="${SEED:-42}"
PROBE_SEED="${PROBE_SEED:-42}"
FREEZE_STEP="${FREEZE_STEP:-4000}"
DANN_RAMP_STEPS="${DANN_RAMP_STEPS:-6800}"
BASE_GRL_TARGET="${BASE_GRL_TARGET:-0.00015}"
POST_GRL_TARGET="${POST_GRL_TARGET:-0.0002}"
ROUTE_TOPK_CALIB_BATCHES="${ROUTE_TOPK_CALIB_BATCHES:-20}"

# For the delayed speaker-reset branch: resume/freeze at 4000, reset grl_head,
# train the discriminator head immediately, but keep the reversed z_L GRL push
# off until this absolute optimizer step.
SPEAKER_RESET_DELAY_STEPS="${SPEAKER_RESET_DELAY_STEPS:-4500}"

PROBE_STEPS="${PROBE_STEPS:-5000}"
PROBE_VAL_EVERY="${PROBE_VAL_EVERY:-250}"
PROBE_PATIENCE="${PROBE_PATIENCE:-10}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PR_PROBE_LR="${PR_PROBE_LR:-5e-4}"
PR_PROBE_WARMUP_STEPS="${PR_PROBE_WARMUP_STEPS:-500}"

SKIP_TRAINING="${SKIP_TRAINING:-0}"
RUN_PROBES="${RUN_PROBES:-1}"
# Keep off by default so this suite is strictly linear diagnostic probes.
RUN_PR_SANITY="${RUN_PR_SANITY:-0}"
REUSE_BASE="${REUSE_BASE:-1}"

# Silence repeated external FutureWarnings from the cached SPEAR HF module.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

for binary_flag in SKIP_TRAINING RUN_PROBES RUN_PR_SANITY REUSE_BASE; do
    value="${!binary_flag}"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "ERROR: $binary_flag must be 0 or 1 (got: $value)" >&2
        exit 2
    fi
done
if (( FREEZE_STEP <= 0 )); then
    echo "ERROR: FREEZE_STEP must be positive." >&2
    exit 2
fi
if (( ROUTE_TOPK_CALIB_BATCHES <= 0 )); then
    echo "ERROR: ROUTE_TOPK_CALIB_BATCHES must be positive." >&2
    exit 2
fi
if (( SPEAKER_RESET_DELAY_STEPS < FREEZE_STEP )); then
    echo "ERROR: SPEAKER_RESET_DELAY_STEPS must be >= FREEZE_STEP." >&2
    exit 2
fi

final_ckpt_for_run() {
    local run_name="$1"
    local total_steps="$2"
    local ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$run_name/checkpoints"
    local ckpt="$ckpt_dir/stage2_step${total_steps}.pt"
    if [[ ! -f "$ckpt" && -f "$ckpt_dir/final.pt" ]]; then
        ckpt="$ckpt_dir/final.pt"
    fi
    printf '%s\n' "$ckpt"
}

checkpoint_step() {
    local ckpt="$1"
    "${BLACKWELL_VENV}/bin/python" - "$ckpt" <<'PY'
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ckpt.get("step", -1)))
PY
}

common_data_train_args() {
    local total_steps="$1"
    printf '%s\n' \
        --stage 2 \
        --stage2_from_scratch \
        --local_data \
        --librispeech_root "$BLACKWELL_DATA_ROOT/LibriSpeech" \
        --lexicon_path "$BLACKWELL_DATA_ROOT/librispeech-lexicon.txt" \
        --train_split_dir train-clean-100 \
        --speaker_stratified_holdout \
        --spear_layernorm \
        --K 5120 \
        --topk 256 \
        --aux_k 64 \
        --aux_k_coef 0.03125 \
        --dead_steps_threshold 256 \
        --stage2_steps "$total_steps" \
        --stage2_schedule_steps "$total_steps" \
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
        --seed "$SEED"
}

learned_train_common_args() {
    local train_run_name="$1"
    local total_steps="$2"
    local grl_target="$3"
    local grl_delay_steps="$4"
    local train_ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/checkpoints"
    printf '%s\n' \
        $(common_data_train_args "$total_steps") \
        --n_routes 2 \
        --hard_gumbel_routing \
        --gumbel_tau_start 1.0 \
        --gumbel_tau_end 0.1 \
        --routing_init_std 0.5 \
        --routing_spec_weight 0.01 \
        --grl_linear_mean \
        --grl_grad_norm \
        --grl_grad_norm_target "$grl_target" \
        --alpha 0.8 \
        --beta 0.6 \
        --grl_weight 1.0 \
        --grl_phoneme_weight 0.2 \
        --grl_delay_steps "$grl_delay_steps" \
        --dann_full_discriminator \
        --dann_ramp_steps "$DANN_RAMP_STEPS" \
        --lr_disc 1e-3 \
        --n_disc_steps 3 \
        --rho 0.0 \
        --lr_routing 1e-3 \
        --checkpoint_dir "$train_ckpt_dir" \
        --runs_dir "$BLACKWELL_OUTPUT_ROOT/$train_run_name/tensorboard" \
        --log_dir "$BLACKWELL_OUTPUT_ROOT/$train_run_name/trainer_logs"
}

routing_probe_flags() {
    printf '%s\n' --n_routes 2 --hard_gumbel_routing --gumbel_tau_end 0.1
}

probe_common_args() {
    local final_ckpt="$1"
    shift
    printf '%s\n' \
        --stage2_ckpt "$final_ckpt" \
        --stage1_ckpt "$final_ckpt" \
        --local_data \
        --librispeech_root "$BLACKWELL_DATA_ROOT/LibriSpeech" \
        --lexicon_path "$BLACKWELL_DATA_ROOT/librispeech-lexicon.txt" \
        --spear_layernorm \
        --topk 256 \
        --probe_steps "$PROBE_STEPS" \
        --probe_val_every "$PROBE_VAL_EVERY" \
        --probe_patience "$PROBE_PATIENCE" \
        --probe_warmup_steps 0 \
        --pr_probe_warmup_steps "$PR_PROBE_WARMUP_STEPS" \
        --seed "$PROBE_SEED" \
        "$@"
}

run_pr_sanity() {
    local train_run_name="$1"
    local final_ckpt="$2"
    shift 2
    local route_probe_args=("$@")

    if [[ "$RUN_PR_SANITY" != "1" ]]; then
        return 0
    fi

    local sanity_run="diag_${train_run_name}_checkpoint_pr_sanity"
    RUN_DESCRIPTION="Checkpoint PR-head sanity check for $train_run_name"
    local sanity_command=(
        python -u Disentanglement/diag_probe/run.py
        $(probe_common_args "$final_ckpt" "${route_probe_args[@]}")
        --run_name "$sanity_run"
        --sources "z_L"
        --tasks "pr"
        --pr_sanity_only
        --pr_max_examples 0
    )
    blackwell_run "$sanity_run" "${sanity_command[@]}"
}

run_pr_linear_probes() {
    local train_run_name="$1"
    local final_ckpt="$2"
    local probe_json_dir="$3"
    shift 3
    local route_probe_args=("$@")

    local pr_source
    for pr_source in z_t z_L z_P; do
        local pr_probe_run="diag_${train_run_name}_${pr_source}_pr_linear_seed${PROBE_SEED}"
        RUN_DESCRIPTION="Final-checkpoint PR-linear probe for $train_run_name: source ${pr_source}"
        local pr_probe_command=(
            python -u Disentanglement/diag_probe/run.py
            $(probe_common_args "$final_ckpt" "${route_probe_args[@]}")
            --run_name "$pr_probe_run"
            --output_json "$probe_json_dir/${pr_probe_run}.json"
            --sources "$pr_source"
            --tasks "pr"
            --pr_probe_arch linear
            --pr_probe_lr "$PR_PROBE_LR"
            --pr_max_examples 0
            --no-pr_checkpoint_sanity
        )
        blackwell_run "$pr_probe_run" "${pr_probe_command[@]}"
    done
}

run_sid_linear_probe() {
    local train_run_name="$1"
    local final_ckpt="$2"
    local probe_json_dir="$3"
    shift 3
    local route_probe_args=("$@")

    local sid_probe_run="diag_${train_run_name}_sid_linear_seed${PROBE_SEED}"
    RUN_DESCRIPTION="Final-checkpoint SID-linear probe for $train_run_name: sources z_t,z_L,z_P"
    local sid_probe_command=(
        python -u Disentanglement/diag_probe/run.py
        $(probe_common_args "$final_ckpt" "${route_probe_args[@]}")
        --run_name "$sid_probe_run"
        --output_json "$probe_json_dir/${sid_probe_run}.json"
        --sources "z_t,z_L,z_P"
        --tasks "sid"
        --sid_probe_arch linear
        --sid_probe_lr "$SID_PROBE_LR"
    )
    blackwell_run "$sid_probe_run" "${sid_probe_command[@]}"
}

run_search_probes() {
    local train_run_name="$1"
    local final_ckpt="$2"
    local probe_json_dir="$3"
    shift 3
    local route_probe_args=("$@")
    mkdir -p "$probe_json_dir"
    export BLACKWELL_LOG_GROUP="$train_run_name"

    echo "[probe plan] $train_run_name: PR=linear-only; SID=linear-only; no direct/stat probes."
    run_pr_sanity "$train_run_name" "$final_ckpt" "${route_probe_args[@]}"
    run_pr_linear_probes "$train_run_name" "$final_ckpt" "$probe_json_dir" "${route_probe_args[@]}"
    run_sid_linear_probe "$train_run_name" "$final_ckpt" "$probe_json_dir" "${route_probe_args[@]}"
}

run_base_to_freeze() {
    local base_run_name="$1"
    local total_steps="$2"
    local route_ckpt="$BLACKWELL_OUTPUT_ROOT/$base_run_name/checkpoints/latest-resume.pt"
    local route_step=""
    export BLACKWELL_LOG_GROUP="$base_run_name"

    if [[ "$REUSE_BASE" == "1" && -f "$route_ckpt" ]]; then
        route_step="$(checkpoint_step "$route_ckpt")"
        if [[ "$route_step" == "$FREEZE_STEP" ]]; then
            echo "[base] reusing freeze checkpoint: $route_ckpt (step=${route_step})"
            return 0
        fi
        echo "[base] found $route_ckpt but step=${route_step}, expected ${FREEZE_STEP}; retraining base."
    fi

    if [[ "$SKIP_TRAINING" == "1" ]]; then
        echo "ERROR: SKIP_TRAINING=1 but usable step-${FREEZE_STEP} base checkpoint is unavailable: $route_ckpt" >&2
        return 3
    fi

    RUN_DESCRIPTION="Base phase: learned hard routing until ${FREEZE_STEP}; total schedule=${total_steps}; DANN ramp=${DANN_RAMP_STEPS}; gn=${BASE_GRL_TARGET}"
    local base_command=(
        python -u Disentanglement/run.py
        $(learned_train_common_args "$base_run_name" "$total_steps" "$BASE_GRL_TARGET" 0)
        --segment_steps "$FREEZE_STEP"
        --resume_every 500
    )
    blackwell_run "${base_run_name}_learn_to_${FREEZE_STEP}" "${base_command[@]}"
    [[ -f "$route_ckpt" ]] || {
        echo "ERROR: base freeze checkpoint missing after training: $route_ckpt" >&2
        return 3
    }
    route_step="$(checkpoint_step "$route_ckpt")"
    [[ "$route_step" == "$FREEZE_STEP" ]] || {
        echo "ERROR: base checkpoint has step=${route_step}, expected ${FREEZE_STEP}: $route_ckpt" >&2
        return 3
    }
}

run_branch_from_freeze() {
    local branch_run_name="$1"
    local total_steps="$2"
    local route_ckpt="$3"
    local post_grl_target="$4"
    local reset_heads="$5"
    local grl_delay_steps="$6"
    local probe_json_dir="$BLACKWELL_OUTPUT_ROOT/$branch_run_name/probe_json"
    local final_ckpt
    export BLACKWELL_LOG_GROUP="$branch_run_name"

    if (( FREEZE_STEP >= total_steps )); then
        echo "ERROR: FREEZE_STEP=${FREEZE_STEP} must be < total_steps=${total_steps}" >&2
        return 2
    fi

    if [[ "$SKIP_TRAINING" == "0" ]]; then
        [[ -f "$route_ckpt" ]] || {
            echo "ERROR: branch resume checkpoint missing: $route_ckpt" >&2
            return 3
        }

        RUN_DESCRIPTION="Branch: freeze learned routing at ${FREEZE_STEP}, enforce learned route-local TopK quota, total=${total_steps}, post_gn=${post_grl_target}, reset_heads=${reset_heads:-none}, grl_delay=${grl_delay_steps}"
        local continue_command=(
            python -u Disentanglement/run.py
            $(learned_train_common_args "$branch_run_name" "$total_steps" "$post_grl_target" "$grl_delay_steps")
            --resume "$route_ckpt"
            --freeze_learned_routing_on_resume
            --freeze_route_topk_on_resume
            --route_topk_calib_batches "$ROUTE_TOPK_CALIB_BATCHES"
            --resume_every 500
        )
        if [[ -n "$reset_heads" ]]; then
            continue_command+=(--reset_adversary_heads_on_resume "$reset_heads")
        fi
        blackwell_run "${branch_run_name}_freeze_quota_continue" "${continue_command[@]}"
    else
        echo "Skipping training for $branch_run_name"
    fi

    final_ckpt="$(final_ckpt_for_run "$branch_run_name" "$total_steps")"
    [[ -f "$final_ckpt" ]] || {
        echo "ERROR: final checkpoint missing for $branch_run_name: $final_ckpt" >&2
        return 4
    }
    if [[ "$RUN_PROBES" == "1" ]]; then
        run_search_probes "$branch_run_name" "$final_ckpt" "$probe_json_dir" $(routing_probe_flags)
    fi
}

echo "Final learned-routing 4k quota-freeze suite"
echo "  freeze step:          ${FREEZE_STEP}"
echo "  DANN ramp:            ${DANN_RAMP_STEPS}"
echo "  base gn:              ${BASE_GRL_TARGET}"
echo "  post gn:              ${POST_GRL_TARGET}"
echo "  speaker reset delay:  ${SPEAKER_RESET_DELAY_STEPS}"
echo "  probes:               PR linear only; SID linear only; PR sanity=${RUN_PR_SANITY}"
echo

base16_run="libri_advlearn_hardqfreeze${FREEZE_STEP}_base_dann${DANN_RAMP_STEPS}_gn00015_gp02_aux64_16k_s${SEED}"
base20_run="libri_advlearn_hardqfreeze${FREEZE_STEP}_base_dann${DANN_RAMP_STEPS}_gn00015_gp02_aux64_20k_s${SEED}"
base16_ckpt="$BLACKWELL_OUTPUT_ROOT/$base16_run/checkpoints/latest-resume.pt"
base20_ckpt="$BLACKWELL_OUTPUT_ROOT/$base20_run/checkpoints/latest-resume.pt"

# Run the most promising/cheapest branches first.
run_base_to_freeze "$base16_run" 16000
run_branch_from_freeze \
    "libri_advlearn_hardqfreeze${FREEZE_STEP}_gn0002_dann${DANN_RAMP_STEPS}_gp02_aux64_16k_s${SEED}" \
    16000 \
    "$base16_ckpt" \
    "$POST_GRL_TARGET" \
    "" \
    0
run_branch_from_freeze \
    "libri_advlearn_hardqfreeze${FREEZE_STEP}_spkresetdelay${SPEAKER_RESET_DELAY_STEPS}_dann${DANN_RAMP_STEPS}_gn00015_gp02_aux64_16k_s${SEED}" \
    16000 \
    "$base16_ckpt" \
    "$BASE_GRL_TARGET" \
    "grl_head" \
    "$SPEAKER_RESET_DELAY_STEPS"
run_branch_from_freeze \
    "libri_advlearn_hardqfreeze${FREEZE_STEP}_spkreset_dann${DANN_RAMP_STEPS}_gn00015_gp02_aux64_16k_s${SEED}" \
    16000 \
    "$base16_ckpt" \
    "$BASE_GRL_TARGET" \
    "grl_head" \
    0

# Longer no-reset continuation last, because it is most expensive.
run_base_to_freeze "$base20_run" 20000
run_branch_from_freeze \
    "libri_advlearn_hardqfreeze${FREEZE_STEP}_gn0002_dann${DANN_RAMP_STEPS}_gp02_aux64_20k_s${SEED}" \
    20000 \
    "$base20_ckpt" \
    "$POST_GRL_TARGET" \
    "" \
    0

echo
echo "All final learned-routing 4k quota-freeze jobs completed."
