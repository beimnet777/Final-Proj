#!/usr/bin/env bash
# Focused learned-routing search suite + reconstruction-only SAE baseline.
#
# Sequential execution, fail-fast:
#   1) reconstruction-only SAE baseline
#   2) learned hard routing forever, gn=1.5e-4
#   3) learned soft routing forever, gn=1.5e-4
#   4) learned hard routing -> freeze@4k, DANN ramp to 1 by 3k, gn=1.5e-4
#   5) learned hard routing -> freeze@5k, DANN ramp to 1 by 3k, gn=1.5e-4
#   6) learned hard routing -> freeze@6k, DANN ramp to 1 by 3k, gn=1.5e-4
#
# Search probes:
#   - no SID-stats
#   - trainable probe patience defaults to 10
#   - learned/freeze runs: PR-linear and SID-linear on z_t,z_L,z_P
#   - reconstruction baseline: PR-linear, SID-linear, ASR-LSTM on z_t
#
# Optional controls:
#   GPU_ID=1 ./Disentanglement/blackwell/experiments/libri_advlearn_search6_s42.sh
#   SKIP_TRAINING=1       reuse existing checkpoints and only run probes
#   RUN_PROBES=0          train only
#   RUN_ASR_FOR_SEARCH=1  also run ASR-LSTM on z_t,z_L,z_P for learned/freeze runs
#   PR_ARCHES="linear"    PR-linear only; direct is intentionally not the default
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
ASR_PROBE_STEPS="${ASR_PROBE_STEPS:-5000}"
ASR_PROBE_LR="${ASR_PROBE_LR:-1e-4}"
ASR_PROBE_WARMUP_STEPS="${ASR_PROBE_WARMUP_STEPS:-500}"

GRL_TARGET="${GRL_TARGET:-0.00015}"
FREEZE_DANN_RAMP_STEPS="${FREEZE_DANN_RAMP_STEPS:-3000}"
FREEZE_STEPS="${FREEZE_STEPS:-4000 5000 6000}"
PR_ARCHES="${PR_ARCHES:-linear}"

SKIP_TRAINING="${SKIP_TRAINING:-0}"
RUN_PROBES="${RUN_PROBES:-1}"
RUN_BASE="${RUN_BASE:-1}"
RUN_HARD_FOREVER="${RUN_HARD_FOREVER:-1}"
RUN_SOFT_FOREVER="${RUN_SOFT_FOREVER:-1}"
RUN_FREEZE="${RUN_FREEZE:-1}"
RUN_BASE_ASR="${RUN_BASE_ASR:-1}"
RUN_ASR_FOR_SEARCH="${RUN_ASR_FOR_SEARCH:-0}"
RUN_PR_SANITY="${RUN_PR_SANITY:-1}"

# SPEAR's cached HuggingFace module currently emits repeated PyTorch
# FutureWarnings for torch.cuda.amp.autocast. They are noisy but not actionable.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

for binary_flag in SKIP_TRAINING RUN_PROBES RUN_BASE RUN_HARD_FOREVER RUN_SOFT_FOREVER RUN_FREEZE RUN_BASE_ASR RUN_ASR_FOR_SEARCH RUN_PR_SANITY; do
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
if (( FREEZE_DANN_RAMP_STEPS < 0 )); then
    echo "ERROR: FREEZE_DANN_RAMP_STEPS must be >= 0." >&2
    exit 2
fi

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

run_pr_probes() {
    local train_run_name="$1"
    local final_ckpt="$2"
    local probe_json_dir="$3"
    local sources="$4"
    shift 4
    local route_probe_args=("$@")

    local pr_arch pr_source
    for pr_arch in $PR_ARCHES; do
        for pr_source in $sources; do
            local pr_probe_run="diag_${train_run_name}_${pr_source}_pr_${pr_arch}_seed${PROBE_SEED}"
            RUN_DESCRIPTION="Final-checkpoint PR-${pr_arch} probe for $train_run_name: source ${pr_source}"
            local pr_probe_command=(
                python -u Disentanglement/diag_probe/run.py
                $(probe_common_args "$final_ckpt" "${route_probe_args[@]}")
                --run_name "$pr_probe_run"
                --output_json "$probe_json_dir/${pr_probe_run}.json"
                --sources "$pr_source"
                --tasks "pr"
                --pr_probe_arch "$pr_arch"
                --pr_probe_lr "$PR_PROBE_LR"
                --pr_max_examples 0
                --no-pr_checkpoint_sanity
            )
            blackwell_run "$pr_probe_run" "${pr_probe_command[@]}"
        done
    done
}

run_sid_linear_probe() {
    local train_run_name="$1"
    local final_ckpt="$2"
    local probe_json_dir="$3"
    local sources_csv="$4"
    shift 4
    local route_probe_args=("$@")

    local sid_probe_run="diag_${train_run_name}_sid_linear_seed${PROBE_SEED}"
    RUN_DESCRIPTION="Final-checkpoint SID-linear probe for $train_run_name: sources ${sources_csv}"
    local sid_probe_command=(
        python -u Disentanglement/diag_probe/run.py
        $(probe_common_args "$final_ckpt" "${route_probe_args[@]}")
        --run_name "$sid_probe_run"
        --output_json "$probe_json_dir/${sid_probe_run}.json"
        --sources "$sources_csv"
        --tasks "sid"
        --sid_probe_arch linear
        --sid_probe_lr "$SID_PROBE_LR"
    )
    blackwell_run "$sid_probe_run" "${sid_probe_command[@]}"
}

run_asr_lstm_probe() {
    local train_run_name="$1"
    local final_ckpt="$2"
    local probe_json_dir="$3"
    local sources_csv="$4"
    shift 4
    local route_probe_args=("$@")

    local asr_probe_run="diag_${train_run_name}_asr_lstm_seed${PROBE_SEED}"
    RUN_DESCRIPTION="Final-checkpoint ASR-LSTM probe for $train_run_name: sources ${sources_csv}"
    local asr_probe_command=(
        python -u Disentanglement/diag_probe/run.py
        $(probe_common_args "$final_ckpt" "${route_probe_args[@]}")
        --run_name "$asr_probe_run"
        --output_json "$probe_json_dir/${asr_probe_run}.json"
        --sources "$sources_csv"
        --tasks "asr"
        --asr_probe_arch lstm
        --probe_steps "$ASR_PROBE_STEPS"
        --asr_probe_lr "$ASR_PROBE_LR"
        --asr_probe_warmup_steps "$ASR_PROBE_WARMUP_STEPS"
        --no-pr_checkpoint_sanity
    )
    blackwell_run "$asr_probe_run" "${asr_probe_command[@]}"
}

run_base_probes() {
    local train_run_name="$1"
    local final_ckpt="$2"
    local probe_json_dir="$3"
    mkdir -p "$probe_json_dir"
    export BLACKWELL_LOG_GROUP="$train_run_name"

    local base_pr_arches="${BASE_PR_ARCHES:-linear}"
    local saved_pr_arches="$PR_ARCHES"
    PR_ARCHES="$base_pr_arches"
    run_pr_probes "$train_run_name" "$final_ckpt" "$probe_json_dir" "z_t"
    PR_ARCHES="$saved_pr_arches"
    run_sid_linear_probe "$train_run_name" "$final_ckpt" "$probe_json_dir" "z_t"
    if [[ "$RUN_BASE_ASR" == "1" ]]; then
        run_asr_lstm_probe "$train_run_name" "$final_ckpt" "$probe_json_dir" "z_t"
    fi
}

run_search_probes() {
    local train_run_name="$1"
    local final_ckpt="$2"
    local probe_json_dir="$3"
    shift 3
    local route_probe_args=("$@")
    mkdir -p "$probe_json_dir"
    export BLACKWELL_LOG_GROUP="$train_run_name"

    run_pr_sanity "$train_run_name" "$final_ckpt" "${route_probe_args[@]}"
    run_pr_probes "$train_run_name" "$final_ckpt" "$probe_json_dir" "z_t z_L z_P" "${route_probe_args[@]}"
    run_sid_linear_probe "$train_run_name" "$final_ckpt" "$probe_json_dir" "z_t,z_L,z_P" "${route_probe_args[@]}"
    if [[ "$RUN_ASR_FOR_SEARCH" == "1" ]]; then
        run_asr_lstm_probe "$train_run_name" "$final_ckpt" "$probe_json_dir" "z_t,z_L,z_P" "${route_probe_args[@]}"
    fi
}

run_recon_only_base() {
    local train_run_name="libri_sae_recon_topk256_aux64_12k_s${SEED}"
    local train_ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/checkpoints"
    local probe_json_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/probe_json"
    local final_ckpt
    final_ckpt="$(final_ckpt_for_run "$train_run_name")"
    export BLACKWELL_LOG_GROUP="$train_run_name"

    local train_command=(
        python -u Disentanglement/run.py
        $(common_data_train_args)
        --no_routing
        --alpha 0.0
        --beta 0.0
        --grl_weight 0.0
        --grl_phoneme_weight 0.0
        --rho 0.0
        --n_disc_steps 1
        --checkpoint_dir "$train_ckpt_dir"
        --runs_dir "$BLACKWELL_OUTPUT_ROOT/$train_run_name/tensorboard"
        --log_dir "$BLACKWELL_OUTPUT_ROOT/$train_run_name/trainer_logs"
    )

    RUN_DESCRIPTION="Reconstruction-only SAE baseline: K=5120 topk=256 AuxK=64, no routing objectives/adversaries"
    if [[ "$SKIP_TRAINING" == "1" ]]; then
        echo "Skipping training for $train_run_name; reusing checkpoint: $final_ckpt"
    else
        blackwell_run "$train_run_name" "${train_command[@]}"
    fi

    final_ckpt="$(final_ckpt_for_run "$train_run_name")"
    [[ -f "$final_ckpt" ]] || {
        echo "ERROR: final checkpoint missing for $train_run_name: $final_ckpt" >&2
        return 3
    }
    if [[ "$RUN_PROBES" == "1" ]]; then
        run_base_probes "$train_run_name" "$final_ckpt" "$probe_json_dir"
    fi
}

learned_train_common_args() {
    local train_run_name="$1"
    local train_ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/checkpoints"
    printf '%s\n' \
        $(common_data_train_args) \
        --n_routes 2 \
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

routing_train_flags() {
    local mode="$1"
    case "$mode" in
        hard)
            printf '%s\n' --hard_gumbel_routing --gumbel_tau_start 1.0 --gumbel_tau_end 0.1
            ;;
        soft)
            printf '%s\n' --no-hard_gumbel_routing --gumbel_tau_start 1.0 --gumbel_tau_end 0.5
            ;;
        *)
            echo "ERROR: unknown routing mode: $mode" >&2
            return 2
            ;;
    esac
}

routing_probe_flags() {
    local mode="$1"
    case "$mode" in
        hard)
            printf '%s\n' --n_routes 2 --hard_gumbel_routing --gumbel_tau_end 0.1
            ;;
        soft)
            printf '%s\n' --n_routes 2 --no-hard_gumbel_routing --gumbel_tau_end 0.5
            ;;
        *)
            echo "ERROR: unknown routing probe mode: $mode" >&2
            return 2
            ;;
    esac
}

run_learned_forever() {
    local mode="$1"
    local mode_label="$mode"
    local train_run_name="libri_advlearn_${mode_label}_gn00015_gp02_aux64_12k_s${SEED}"
    local train_ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/checkpoints"
    local probe_json_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/probe_json"
    local final_ckpt
    final_ckpt="$(final_ckpt_for_run "$train_run_name")"
    export BLACKWELL_LOG_GROUP="$train_run_name"

    local train_command=(
        python -u Disentanglement/run.py
        $(learned_train_common_args "$train_run_name")
        $(routing_train_flags "$mode")
    )

    RUN_DESCRIPTION="Learned ${mode} routing forever: gn=1.5e-4, z_P phone GRL 0.2, AuxK 64, total ${STAGE2_STEPS}"
    if [[ "$SKIP_TRAINING" == "1" ]]; then
        echo "Skipping training for $train_run_name; reusing checkpoint: $final_ckpt"
    else
        blackwell_run "$train_run_name" "${train_command[@]}"
    fi

    final_ckpt="$(final_ckpt_for_run "$train_run_name")"
    [[ -f "$final_ckpt" ]] || {
        echo "ERROR: final checkpoint missing for $train_run_name: $final_ckpt" >&2
        return 3
    }
    if [[ "$RUN_PROBES" == "1" ]]; then
        run_search_probes "$train_run_name" "$final_ckpt" "$probe_json_dir" $(routing_probe_flags "$mode")
    fi
}

run_learned_freeze() {
    local freeze_step="$1"
    if (( freeze_step <= 0 || freeze_step >= STAGE2_STEPS )); then
        echo "ERROR: freeze step must be >0 and < STAGE2_STEPS; got ${freeze_step}" >&2
        return 2
    fi

    local train_run_name="libri_advlearn_hardfreeze${freeze_step}_dann3k_gn00015_gp02_aux64_12k_s${SEED}"
    local train_ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/checkpoints"
    local probe_json_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/probe_json"
    local route_ckpt="$train_ckpt_dir/latest-resume.pt"
    local final_ckpt
    final_ckpt="$(final_ckpt_for_run "$train_run_name")"
    export BLACKWELL_LOG_GROUP="$train_run_name"

    local common_args=(
        $(learned_train_common_args "$train_run_name")
        $(routing_train_flags hard)
        --dann_ramp_steps "$FREEZE_DANN_RAMP_STEPS"
        --stage2_schedule_steps "$STAGE2_STEPS"
    )

    if [[ "$SKIP_TRAINING" == "0" ]]; then
        RUN_DESCRIPTION="Phase 1/2: learned hard routing until ${freeze_step}; DANN ramp=${FREEZE_DANN_RAMP_STEPS}; gn=1.5e-4"
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

        RUN_DESCRIPTION="Phase 2/2: resume from ${freeze_step}, freeze learned routing, continue to ${STAGE2_STEPS}; DANN ramp=${FREEZE_DANN_RAMP_STEPS}"
        local continue_command=(
            python -u Disentanglement/run.py
            "${common_args[@]}"
            --resume "$route_ckpt"
            --freeze_learned_routing_on_resume
            --resume_every 500
        )
        blackwell_run "${train_run_name}_freeze_continue" "${continue_command[@]}"
    else
        echo "Skipping training for $train_run_name; reusing checkpoint: $final_ckpt"
    fi

    final_ckpt="$(final_ckpt_for_run "$train_run_name")"
    [[ -f "$final_ckpt" ]] || {
        echo "ERROR: final checkpoint missing for $train_run_name: $final_ckpt" >&2
        return 4
    }
    if [[ "$RUN_PROBES" == "1" ]]; then
        run_search_probes "$train_run_name" "$final_ckpt" "$probe_json_dir" $(routing_probe_flags hard)
    fi
}

if [[ "$RUN_BASE" == "1" ]]; then
    run_recon_only_base
fi

if [[ "$RUN_HARD_FOREVER" == "1" ]]; then
    run_learned_forever hard
fi

if [[ "$RUN_SOFT_FOREVER" == "1" ]]; then
    run_learned_forever soft
fi

if [[ "$RUN_FREEZE" == "1" ]]; then
    for freeze_step in $FREEZE_STEPS; do
        run_learned_freeze "$freeze_step"
    done
fi

echo
echo "All requested learned-routing search jobs completed."
