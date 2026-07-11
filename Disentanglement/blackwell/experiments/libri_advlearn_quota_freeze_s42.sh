#!/usr/bin/env bash
# Learned routing -> freeze with learned route-local TopK quotas.
#
# Sequential jobs:
#   1) hard learned routing until 4k, DANN ramp chosen so lambda≈0.9 by 2k,
#      freeze membership + learned active quotas, continue to 12k
#   2) hard learned routing until 5k, lambda≈0.9 by 3k, freeze + quotas
#   3) hard learned routing until 6k, lambda≈0.9 by 4k, freeze + quotas
#
# Search probes are intentionally minimal:
#   - PR: linear only on z_t, z_L, z_P
#   - SID: linear only on z_t, z_L, z_P
#   - no PR-direct, no SID-stats, no ASR
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

STAGE2_STEPS="${STAGE2_STEPS:-12000}"
SEED="${SEED:-42}"
PROBE_SEED="${PROBE_SEED:-42}"
PROBE_STEPS="${PROBE_STEPS:-5000}"
PROBE_VAL_EVERY="${PROBE_VAL_EVERY:-250}"
PROBE_PATIENCE="${PROBE_PATIENCE:-10}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PR_PROBE_LR="${PR_PROBE_LR:-5e-4}"
PR_PROBE_WARMUP_STEPS="${PR_PROBE_WARMUP_STEPS:-500}"
ROUTE_TOPK_CALIB_BATCHES="${ROUTE_TOPK_CALIB_BATCHES:-20}"

GRL_TARGET="${GRL_TARGET:-0.00015}"
FREEZE_STEPS="${FREEZE_STEPS:-4000 5000 6000}"

SKIP_TRAINING="${SKIP_TRAINING:-0}"
RUN_PROBES="${RUN_PROBES:-1}"
RUN_PR_SANITY="${RUN_PR_SANITY:-1}"

# Silence repeated external FutureWarnings from the cached SPEAR HF module.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

for binary_flag in SKIP_TRAINING RUN_PROBES RUN_PR_SANITY; do
    value="${!binary_flag}"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "ERROR: $binary_flag must be 0 or 1 (got: $value)" >&2
        exit 2
    fi
done

if (( STAGE2_STEPS <= 0 )); then
    echo "ERROR: STAGE2_STEPS must be positive." >&2
    exit 2
fi
if (( ROUTE_TOPK_CALIB_BATCHES <= 0 )); then
    echo "ERROR: ROUTE_TOPK_CALIB_BATCHES must be positive." >&2
    exit 2
fi

dann_ramp_for_freeze() {
    local freeze_step="$1"
    case "$freeze_step" in
        4000) printf '%s\n' "${DANN_RAMP_4000:-6800}" ;;
        5000) printf '%s\n' "${DANN_RAMP_5000:-10200}" ;;
        6000) printf '%s\n' "${DANN_RAMP_6000:-13600}" ;;
        *)
            # General fallback: lambda≈0.9 two thousand steps before freeze.
            # DANN lambda is tanh(5 * step / ramp_steps); artanh(.9)/5≈0.2944.
            python - "$freeze_step" <<'PY'
import math, sys
freeze = int(sys.argv[1])
target_step = max(1, freeze - 2000)
print(int(round(target_step / (math.atanh(0.9) / 5.0))))
PY
            ;;
    esac
}

final_ckpt_for_run() {
    local run_name="$1"
    local ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$run_name/checkpoints"
    local ckpt="$ckpt_dir/stage2_step${STAGE2_STEPS}.pt"
    if [[ ! -f "$ckpt" && -f "$ckpt_dir/final.pt" ]]; then
        ckpt="$ckpt_dir/final.pt"
    fi
    printf '%s\n' "$ckpt"
}

common_data_train_args() {
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
        --stage2_steps "$STAGE2_STEPS" \
        --stage2_schedule_steps "$STAGE2_STEPS" \
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
    local train_ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/checkpoints"
    printf '%s\n' \
        $(common_data_train_args) \
        --n_routes 2 \
        --hard_gumbel_routing \
        --gumbel_tau_start 1.0 \
        --gumbel_tau_end 0.1 \
        --routing_init_std 0.5 \
        --routing_spec_weight 0.01 \
        --grl_linear_mean \
        --grl_grad_norm \
        --grl_grad_norm_target "$GRL_TARGET" \
        --alpha 0.8 \
        --beta 0.6 \
        --grl_weight 1.0 \
        --grl_phoneme_weight 0.2 \
        --grl_delay_steps 0 \
        --dann_full_discriminator \
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

run_quota_freeze() {
    local freeze_step="$1"
    if (( freeze_step <= 0 || freeze_step >= STAGE2_STEPS )); then
        echo "ERROR: freeze step must be >0 and < STAGE2_STEPS; got ${freeze_step}" >&2
        return 2
    fi

    local ramp_steps
    ramp_steps="$(dann_ramp_for_freeze "$freeze_step")"
    local train_run_name="libri_advlearn_hardqfreeze${freeze_step}_dann${ramp_steps}_gn00015_gp02_aux64_12k_s${SEED}"
    local train_ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/checkpoints"
    local probe_json_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/probe_json"
    local route_ckpt="$train_ckpt_dir/latest-resume.pt"
    local final_ckpt
    final_ckpt="$(final_ckpt_for_run "$train_run_name")"
    export BLACKWELL_LOG_GROUP="$train_run_name"

    local common_args=(
        $(learned_train_common_args "$train_run_name")
        --dann_ramp_steps "$ramp_steps"
    )

    if [[ "$SKIP_TRAINING" == "0" ]]; then
        RUN_DESCRIPTION="Phase 1/2: learned hard routing until ${freeze_step}; DANN ramp=${ramp_steps}; gn=1.5e-4"
        local learn_command=(
            python -u Disentanglement/run.py
            "${common_args[@]}"
            --segment_steps "$freeze_step"
            --resume_every 500
        )
        blackwell_run "${train_run_name}_learn_to_${freeze_step}" "${learn_command[@]}"

        [[ -f "$route_ckpt" ]] || {
            echo "ERROR: freeze checkpoint missing: $route_ckpt" >&2
            return 3
        }

        RUN_DESCRIPTION="Phase 2/2: freeze learned routing at ${freeze_step}, calibrate learned route-local TopK quotas, continue to ${STAGE2_STEPS}; DANN ramp=${ramp_steps}"
        local continue_command=(
            python -u Disentanglement/run.py
            "${common_args[@]}"
            --resume "$route_ckpt"
            --freeze_learned_routing_on_resume
            --freeze_route_topk_on_resume
            --route_topk_calib_batches "$ROUTE_TOPK_CALIB_BATCHES"
            --resume_every 500
        )
        blackwell_run "${train_run_name}_freeze_quota_continue" "${continue_command[@]}"
    else
        echo "Skipping training for $train_run_name; reusing checkpoint: $final_ckpt"
    fi

    final_ckpt="$(final_ckpt_for_run "$train_run_name")"
    [[ -f "$final_ckpt" ]] || {
        echo "ERROR: final checkpoint missing for $train_run_name: $final_ckpt" >&2
        return 4
    }
    if [[ "$RUN_PROBES" == "1" ]]; then
        run_search_probes "$train_run_name" "$final_ckpt" "$probe_json_dir" $(routing_probe_flags)
    fi
}

echo "Quota-freeze learned-routing suite"
echo "  freeze steps: ${FREEZE_STEPS}"
echo "  DANN ramps: 4k=$(dann_ramp_for_freeze 4000), 5k=$(dann_ramp_for_freeze 5000), 6k=$(dann_ramp_for_freeze 6000)"
echo "  probes: PR linear only; SID linear only"
echo

for freeze_step in $FREEZE_STEPS; do
    run_quota_freeze "$freeze_step"
done

echo
echo "All quota-freeze jobs completed."
