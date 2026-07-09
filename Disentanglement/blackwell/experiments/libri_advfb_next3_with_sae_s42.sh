#!/usr/bin/env bash
# SAE unit analysis + next three fixed/learned-routing experiments.
#
# Execution order, fail-fast:
#   0) SAEUnitAnalysis on an existing checkpoint, normally the current best
#      gn00015 240L/16P checkpoint.  This is analysis only; no training.
#
#   1) constant_gn0001_240L16P
#      Test whether weaker z_L speaker cleaning recovers PR without letting
#      speaker ID leak back into z_L.
#
#   2) early_decay_gn0002to00005_240L16P
#      Clean early with 2e-4, then decay from 3k->9k to 5e-5 and hold.  This
#      tests whether sustained late adversarial pressure is what damages PR.
#
#   3) learned_hard_freeze4k_gn00015
#      Learn hard routing for FREEZE_STEP updates, then resume and freeze the
#      learned routing deterministically while continuing to STAGE2_STEPS.
#      This tests whether learned routing failed because routes kept moving.
#
# After every training experiment, probe the FINAL checkpoint only:
#   - checkpoint PR-head sanity
#   - PR linear and PR direct, source-by-source
#   - SID linear and SID stats
#
# Probe logs are grouped under the training run's log folder:
#   Disentanglement/blackwell/logs/<training_run>/
#
# Required:
#   GPU_ID=0..7
#
# Required for SAE analysis unless RUN_SAE_ANALYSIS=0:
#   SAE_DATA_BUNDLE=/path/to/SAEUnitAnalysis bundle directory
#
# Optional:
#   RUN_SAE_ANALYSIS=0      skip the initial SAE analysis
#   SAE_CHECKPOINT=...      checkpoint to analyze; default is current gn00015 best
#   SAE_RUN_NAME=...        log/output group for SAE analysis
#   SAE_PROFILE=full|quick  default full
#   FREEZE_STEP=4000        learned-routing freeze step
#   PROBE_STEPS=5000        default diagnostic probe steps
#   SKIP_TRAINING=1         reuse existing checkpoints and only run probes
#   RUN_PROBES=0            train only
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

STAGE2_STEPS="${STAGE2_STEPS:-12000}"
FREEZE_STEP="${FREEZE_STEP:-4000}"
PROBE_STEPS="${PROBE_STEPS:-5000}"
PROBE_VAL_EVERY="${PROBE_VAL_EVERY:-250}"
PROBE_PATIENCE="${PROBE_PATIENCE:-5}"
PR_PROBE_WARMUP_STEPS="${PR_PROBE_WARMUP_STEPS:-500}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PR_PROBE_LR="${PR_PROBE_LR:-5e-4}"
PROBE_SEED="${PROBE_SEED:-42}"
SEED="${SEED:-42}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
RUN_PROBES="${RUN_PROBES:-1}"
RUN_SAE_ANALYSIS="${RUN_SAE_ANALYSIS:-1}"

# SPEAR's cached HuggingFace module currently emits repeated PyTorch
# FutureWarnings for torch.cuda.amp.autocast.  They are not actionable for this
# experiment and can bury real failures in tmux/log output.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

SAE_ANALYSIS="${SAE_ANALYSIS:-health,atlas,selectivity,clustering,similarity,geometry}"
SAE_PROFILE="${SAE_PROFILE:-full}"
SAE_SEED="${SAE_SEED:-42}"
SAE_DEVICE="${SAE_DEVICE:-cuda}"
SAE_INSTALL_DEPS="${SAE_INSTALL_DEPS:-0}"
SAE_SOURCE_RUN_NAME="${SAE_SOURCE_RUN_NAME:-libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42}"

if (( FREEZE_STEP <= 0 || FREEZE_STEP >= STAGE2_STEPS )); then
    echo "ERROR: FREEZE_STEP must be >0 and < STAGE2_STEPS; got FREEZE_STEP=${FREEZE_STEP}, STAGE2_STEPS=${STAGE2_STEPS}" >&2
    exit 2
fi
if [[ "$SKIP_TRAINING" != "0" && "$SKIP_TRAINING" != "1" ]]; then
    echo "ERROR: SKIP_TRAINING must be 0 or 1 (got: $SKIP_TRAINING)" >&2
    exit 2
fi
if [[ "$RUN_PROBES" != "0" && "$RUN_PROBES" != "1" ]]; then
    echo "ERROR: RUN_PROBES must be 0 or 1 (got: $RUN_PROBES)" >&2
    exit 2
fi
if [[ "$RUN_SAE_ANALYSIS" != "0" && "$RUN_SAE_ANALYSIS" != "1" ]]; then
    echo "ERROR: RUN_SAE_ANALYSIS must be 0 or 1 (got: $RUN_SAE_ANALYSIS)" >&2
    exit 2
fi

abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$REPO_ROOT/$1" ;;
    esac
}

sanitize_name() {
    printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//'
}

default_final_ckpt_for_run() {
    local run_name="$1"
    local ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$run_name/checkpoints"
    local ckpt="$ckpt_dir/stage2_step${STAGE2_STEPS}.pt"
    if [[ ! -f "$ckpt" && -f "$ckpt_dir/final.pt" ]]; then
        ckpt="$ckpt_dir/final.pt"
    fi
    printf '%s\n' "$ckpt"
}

run_sae_analysis() {
    if [[ "$RUN_SAE_ANALYSIS" == "0" ]]; then
        echo "Skipping SAE unit analysis because RUN_SAE_ANALYSIS=0"
        return 0
    fi
    if [[ "$SAE_PROFILE" != "full" && "$SAE_PROFILE" != "quick" ]]; then
        echo "ERROR: SAE_PROFILE must be full or quick (got: $SAE_PROFILE)" >&2
        return 2
    fi
    if [[ "$SAE_INSTALL_DEPS" != "0" && "$SAE_INSTALL_DEPS" != "1" ]]; then
        echo "ERROR: SAE_INSTALL_DEPS must be 0 or 1 (got: $SAE_INSTALL_DEPS)" >&2
        return 2
    fi
    if [[ -z "${SAE_DATA_BUNDLE:-}" ]]; then
        echo "ERROR: set SAE_DATA_BUNDLE, or set RUN_SAE_ANALYSIS=0 to skip the initial analysis." >&2
        return 2
    fi

    local sae_checkpoint="${SAE_CHECKPOINT:-$(default_final_ckpt_for_run "$SAE_SOURCE_RUN_NAME")}"
    local sae_data_bundle="$SAE_DATA_BUNDLE"
    sae_checkpoint="$(abs_path "$sae_checkpoint")"
    sae_data_bundle="$(abs_path "$sae_data_bundle")"

    if [[ ! -f "$sae_checkpoint" ]]; then
        echo "ERROR: SAE checkpoint not found: $sae_checkpoint" >&2
        return 1
    fi
    if [[ ! -d "$sae_data_bundle" ]]; then
        echo "ERROR: SAE data bundle not found: $sae_data_bundle" >&2
        return 1
    fi
    if [[ ! -f "$sae_data_bundle/dataset.yaml" ]]; then
        echo "ERROR: SAE data bundle is missing dataset.yaml: $sae_data_bundle" >&2
        return 1
    fi

    local sae_run_name="${SAE_RUN_NAME:-sae_units_${SAE_SOURCE_RUN_NAME}_${SAE_PROFILE}}"
    sae_run_name="$(sanitize_name "$sae_run_name")"
    if [[ -z "$sae_run_name" ]]; then
        echo "ERROR: SAE_RUN_NAME became empty after sanitization." >&2
        return 2
    fi

    local sae_output_dir="${SAE_OUTPUT_DIR:-$BLACKWELL_OUTPUT_ROOT/$sae_run_name/results}"
    sae_output_dir="$(abs_path "$sae_output_dir")"

    export BLACKWELL_LOG_GROUP="$sae_run_name"

    if [[ "$SAE_INSTALL_DEPS" == "1" ]]; then
        RUN_DESCRIPTION="Install SAEUnitAnalysis dependencies before unit analysis"
        blackwell_run "${sae_run_name}_install_deps" \
            python -m pip install -r SAEUnitAnalysis/requirements.txt
    fi

    RUN_DESCRIPTION="Initial SAE unit analysis: checkpoint=$sae_checkpoint data=$sae_data_bundle analysis=$SAE_ANALYSIS profile=$SAE_PROFILE"
    local command=(
        python -u -m SAEUnitAnalysis
        --checkpoint "$sae_checkpoint"
        --data "$sae_data_bundle"
        --analysis "$SAE_ANALYSIS"
        --output-dir "$sae_output_dir"
        --device "$SAE_DEVICE"
        --seed "$SAE_SEED"
        --profile "$SAE_PROFILE"
    )
    blackwell_run "$sae_run_name" "${command[@]}"

    echo
    echo "SAE unit analysis complete."
    echo "Report: ${sae_output_dir}/report/index.html"
    echo "Tables: ${sae_output_dir}/tables"
    echo "Plots:  ${sae_output_dir}/plots"
}

run_fixed_train_and_probes() {
    if [[ $# -lt 11 ]]; then
        echo "Internal error: run_fixed_train_and_probes got too few arguments" >&2
        return 2
    fi

    local train_run_name="$1"; shift
    local description="$1"; shift
    local k_l="$1"; shift
    local k_p="$1"; shift
    local k_u="$1"; shift
    local topk_l="$1"; shift
    local topk_p="$1"; shift
    local topk_u="$1"; shift
    local grl_target="$1"; shift
    local probe_sources="$1"; shift
    local sid_sources_csv="$1"; shift
    local extra_train_args=("$@")

    local train_ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/checkpoints"
    local probe_json_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/probe_json"
    local final_ckpt="$train_ckpt_dir/stage2_step${STAGE2_STEPS}.pt"
    if [[ ! -f "$final_ckpt" && -f "$train_ckpt_dir/final.pt" ]]; then
        final_ckpt="$train_ckpt_dir/final.pt"
    fi

    mkdir -p "$probe_json_dir"
    export BLACKWELL_LOG_GROUP="$train_run_name"

    local train_command=(
        python -u Disentanglement/run.py
        --stage 2
        --stage2_from_scratch
        --local_data
        --librispeech_root "$BLACKWELL_DATA_ROOT/LibriSpeech"
        --lexicon_path "$BLACKWELL_DATA_ROOT/librispeech-lexicon.txt"
        --train_split_dir train-clean-100
        --speaker_stratified_holdout
        --spear_layernorm
        --K 5120
        --topk 256
        --fixed_blocks
        --per_block_topk
        --K_L "$k_l"
        --K_P "$k_p"
        --K_U "$k_u"
        --topk_L "$topk_l"
        --topk_P "$topk_p"
        --topk_U "$topk_u"
        --grl_linear_mean
        --grl_grad_norm
        --grl_grad_norm_target "$grl_target"
        --alpha 0.8
        --beta 0.6
        --grl_weight 1.0
        --grl_phoneme_weight 0.2
        --grl_delay_steps 0
        --dann_full_discriminator
        --lr_disc 1e-3
        --n_disc_steps 3
        --aux_k 64
        --aux_k_coef 0.03125
        --dead_steps_threshold 256
        --rho 0.0
        --stage2_steps "$STAGE2_STEPS"
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
        --checkpoint_dir "$train_ckpt_dir"
        --runs_dir "$BLACKWELL_OUTPUT_ROOT/$train_run_name/tensorboard"
        --log_dir "$BLACKWELL_OUTPUT_ROOT/$train_run_name/trainer_logs"
        --num_workers 2
        --seed "$SEED"
        "${extra_train_args[@]}"
    )

    RUN_DESCRIPTION="$description"
    if [[ "$SKIP_TRAINING" == "1" ]]; then
        echo "Skipping training for $train_run_name; reusing checkpoint: $final_ckpt"
    else
        blackwell_run "$train_run_name" "${train_command[@]}"
    fi

    if [[ ! -f "$final_ckpt" && -f "$train_ckpt_dir/final.pt" ]]; then
        final_ckpt="$train_ckpt_dir/final.pt"
    fi
    if [[ ! -f "$final_ckpt" ]]; then
        echo "ERROR: final checkpoint missing for $train_run_name: $final_ckpt" >&2
        return 3
    fi

    if [[ "$RUN_PROBES" == "1" ]]; then
        run_fixed_probes "$train_run_name" "$final_ckpt" "$probe_json_dir" \
            "$k_l" "$k_p" "$k_u" "$topk_l" "$topk_p" "$topk_u" \
            "$probe_sources" "$sid_sources_csv"
    fi
}

run_fixed_probes() {
    local train_run_name="$1"; shift
    local final_ckpt="$1"; shift
    local probe_json_dir="$1"; shift
    local k_l="$1"; shift
    local k_p="$1"; shift
    local k_u="$1"; shift
    local topk_l="$1"; shift
    local topk_p="$1"; shift
    local topk_u="$1"; shift
    local probe_sources="$1"; shift
    local sid_sources_csv="$1"; shift

    export BLACKWELL_LOG_GROUP="$train_run_name"

    local common_probe_args=(
        --stage2_ckpt "$final_ckpt"
        --stage1_ckpt "$final_ckpt"
        --local_data
        --librispeech_root "$BLACKWELL_DATA_ROOT/LibriSpeech"
        --lexicon_path "$BLACKWELL_DATA_ROOT/librispeech-lexicon.txt"
        --spear_layernorm
        --topk 256
        --fixed_blocks
        --per_block_topk
        --K_L "$k_l"
        --K_P "$k_p"
        --K_U "$k_u"
        --topk_L "$topk_l"
        --topk_P "$topk_p"
        --topk_U "$topk_u"
        --probe_steps "$PROBE_STEPS"
        --probe_val_every "$PROBE_VAL_EVERY"
        --probe_patience "$PROBE_PATIENCE"
        --probe_warmup_steps 0
        --pr_probe_warmup_steps "$PR_PROBE_WARMUP_STEPS"
        --seed "$PROBE_SEED"
    )

    local sanity_run="diag_${train_run_name}_checkpoint_pr_sanity"
    RUN_DESCRIPTION="Checkpoint PR-head sanity check for $train_run_name"
    local sanity_command=(
        python -u Disentanglement/diag_probe/run.py
        "${common_probe_args[@]}"
        --run_name "$sanity_run"
        --sources "z_L"
        --tasks "pr"
        --pr_sanity_only
        --pr_max_examples 0
    )
    blackwell_run "$sanity_run" "${sanity_command[@]}"

    local pr_arch pr_source
    for pr_arch in linear direct; do
        for pr_source in $probe_sources; do
            local pr_probe_run="diag_${train_run_name}_${pr_source}_pr_${pr_arch}_seed${PROBE_SEED}"
            RUN_DESCRIPTION="Final-checkpoint PR-${pr_arch} probe for $train_run_name: source ${pr_source}"
            local pr_probe_command=(
                python -u Disentanglement/diag_probe/run.py
                "${common_probe_args[@]}"
                --run_name "$pr_probe_run"
                --output_json "$probe_json_dir/${pr_probe_run}.json"
                --sources "$pr_source"
                --tasks "pr"
                --pr_probe_arch "$pr_arch"
                --pr_max_examples 0
                --pr_probe_lr "$PR_PROBE_LR"
                --no-pr_checkpoint_sanity
            )
            blackwell_run "$pr_probe_run" "${pr_probe_command[@]}"
        done
    done

    local sid_arch
    for sid_arch in linear stats; do
        local sid_probe_run="diag_${train_run_name}_sid_${sid_arch}_seed${PROBE_SEED}"
        RUN_DESCRIPTION="Final-checkpoint SID-${sid_arch} probe for $train_run_name: sources ${sid_sources_csv}"
        local sid_probe_command=(
            python -u Disentanglement/diag_probe/run.py
            "${common_probe_args[@]}"
            --run_name "$sid_probe_run"
            --output_json "$probe_json_dir/${sid_probe_run}.json"
            --sources "$sid_sources_csv"
            --tasks "sid"
            --sid_probe_arch "$sid_arch"
            --sid_probe_lr "$SID_PROBE_LR"
        )
        blackwell_run "$sid_probe_run" "${sid_probe_command[@]}"
    done
}

run_learned_freeze_train_and_probes() {
    local train_run_name="libri_advlearn_hardfreeze${FREEZE_STEP}_linmean_gn00015_gp02_aux64_12k_s42"
    local train_ckpt_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/checkpoints"
    local probe_json_dir="$BLACKWELL_OUTPUT_ROOT/$train_run_name/probe_json"
    local route_ckpt="$train_ckpt_dir/latest-resume.pt"
    local final_ckpt="$train_ckpt_dir/stage2_step${STAGE2_STEPS}.pt"
    mkdir -p "$probe_json_dir"
    export BLACKWELL_LOG_GROUP="$train_run_name"

    local common_train_args=(
        --stage 2
        --stage2_from_scratch
        --local_data
        --librispeech_root "$BLACKWELL_DATA_ROOT/LibriSpeech"
        --lexicon_path "$BLACKWELL_DATA_ROOT/librispeech-lexicon.txt"
        --train_split_dir train-clean-100
        --speaker_stratified_holdout
        --spear_layernorm
        --K 5120
        --topk 256
        --n_routes 2
        --hard_gumbel_routing
        --gumbel_tau_start 1.0
        --gumbel_tau_end 0.1
        --routing_init_std 0.5
        --routing_spec_weight 0.01
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
        --aux_k 64
        --aux_k_coef 0.03125
        --dead_steps_threshold 256
        --rho 0.0
        --stage2_steps "$STAGE2_STEPS"
        --stage2_schedule_steps "$STAGE2_STEPS"
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
        --checkpoint_dir "$train_ckpt_dir"
        --runs_dir "$BLACKWELL_OUTPUT_ROOT/$train_run_name/tensorboard"
        --log_dir "$BLACKWELL_OUTPUT_ROOT/$train_run_name/trainer_logs"
        --num_workers 2
        --seed "$SEED"
    )

    if [[ "$SKIP_TRAINING" == "0" ]]; then
        RUN_DESCRIPTION="Phase 1/2: learned hard routing until step ${FREEZE_STEP}, gn=1.5e-4, grl_p=0.2"
        local learn_route_command=(
            python -u Disentanglement/run.py
            "${common_train_args[@]}"
            --segment_steps "$FREEZE_STEP"
            --resume_every 500
        )
        blackwell_run "${train_run_name}_learn_to_${FREEZE_STEP}" "${learn_route_command[@]}"

        [[ -f "$route_ckpt" ]] || {
            echo "ERROR: freeze checkpoint missing: $route_ckpt" >&2
            return 3
        }

        RUN_DESCRIPTION="Phase 2/2: resume from ${FREEZE_STEP}, freeze learned routing, continue to ${STAGE2_STEPS}"
        local frozen_continue_command=(
            python -u Disentanglement/run.py
            "${common_train_args[@]}"
            --resume "$route_ckpt"
            --freeze_learned_routing_on_resume
            --resume_every 500
        )
        blackwell_run "${train_run_name}_freeze_continue" "${frozen_continue_command[@]}"
    else
        echo "Skipping learned-freeze training; reusing checkpoint: $final_ckpt"
    fi

    if [[ ! -f "$final_ckpt" && -f "$train_ckpt_dir/final.pt" ]]; then
        final_ckpt="$train_ckpt_dir/final.pt"
    fi
    if [[ ! -f "$final_ckpt" ]]; then
        echo "ERROR: final checkpoint missing for $train_run_name: $final_ckpt" >&2
        return 4
    fi

    if [[ "$RUN_PROBES" == "1" ]]; then
        run_learned_freeze_probes "$train_run_name" "$final_ckpt" "$probe_json_dir"
    fi
}

run_learned_freeze_probes() {
    local train_run_name="$1"; shift
    local final_ckpt="$1"; shift
    local probe_json_dir="$1"; shift

    export BLACKWELL_LOG_GROUP="$train_run_name"

    local common_probe_args=(
        --stage2_ckpt "$final_ckpt"
        --stage1_ckpt "$final_ckpt"
        --local_data
        --librispeech_root "$BLACKWELL_DATA_ROOT/LibriSpeech"
        --lexicon_path "$BLACKWELL_DATA_ROOT/librispeech-lexicon.txt"
        --spear_layernorm
        --topk 256
        --n_routes 2
        --hard_gumbel_routing
        --gumbel_tau_end 0.1
        --probe_steps "$PROBE_STEPS"
        --probe_val_every "$PROBE_VAL_EVERY"
        --probe_patience "$PROBE_PATIENCE"
        --probe_warmup_steps 0
        --pr_probe_warmup_steps "$PR_PROBE_WARMUP_STEPS"
        --seed "$PROBE_SEED"
    )

    local sanity_run="diag_${train_run_name}_checkpoint_pr_sanity"
    RUN_DESCRIPTION="Checkpoint PR-head sanity check for $train_run_name"
    local sanity_command=(
        python -u Disentanglement/diag_probe/run.py
        "${common_probe_args[@]}"
        --run_name "$sanity_run"
        --sources "z_L"
        --tasks "pr"
        --pr_sanity_only
        --pr_max_examples 0
    )
    blackwell_run "$sanity_run" "${sanity_command[@]}"

    local pr_arch pr_source
    for pr_arch in linear direct; do
        for pr_source in z_t z_L z_P; do
            local pr_probe_run="diag_${train_run_name}_${pr_source}_pr_${pr_arch}_seed${PROBE_SEED}"
            RUN_DESCRIPTION="Final-checkpoint PR-${pr_arch} probe for $train_run_name: source ${pr_source}"
            local pr_probe_command=(
                python -u Disentanglement/diag_probe/run.py
                "${common_probe_args[@]}"
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

    local sid_arch
    for sid_arch in linear stats; do
        local sid_probe_run="diag_${train_run_name}_sid_${sid_arch}_seed${PROBE_SEED}"
        RUN_DESCRIPTION="Final-checkpoint SID-${sid_arch} probe for $train_run_name: sources z_t,z_L,z_P"
        local sid_probe_command=(
            python -u Disentanglement/diag_probe/run.py
            "${common_probe_args[@]}"
            --run_name "$sid_probe_run"
            --output_json "$probe_json_dir/${sid_probe_run}.json"
            --sources "z_t,z_L,z_P"
            --tasks "sid"
            --sid_probe_arch "$sid_arch"
            --sid_probe_lr "$SID_PROBE_LR"
        )
        blackwell_run "$sid_probe_run" "${sid_probe_command[@]}"
    done
}

run_sae_analysis

run_fixed_train_and_probes \
    "libri_advfb_next_gn0001_240L16P_aux64_12k_s42" \
    "Next ablation 1/3: fixed 240L/16P, constant z_L speaker GRL target 1e-4, z_P phone GRL 0.2, AuxK 64" \
    4096 1024 0 \
    240 16 0 \
    0.0001 \
    "z_t z_L z_P" \
    "z_t,z_L,z_P"

run_fixed_train_and_probes \
    "libri_advfb_next_earlydecay_gn0002to00005_240L16P_aux64_12k_s42" \
    "Next ablation 2/3: fixed 240L/16P, z_L speaker GRL target 2e-4 until 3k, decay to 5e-5 by 9k, hold to 12k; z_P phone GRL 0.2, AuxK 64" \
    4096 1024 0 \
    240 16 0 \
    0.0002 \
    "z_t z_L z_P" \
    "z_t,z_L,z_P" \
    --grl_grad_norm_decay_start 3000 \
    --grl_grad_norm_decay_end 9000 \
    --grl_grad_norm_final_target 0.00005

run_learned_freeze_train_and_probes

echo
echo "All requested SAE + three experiment jobs completed."
