#!/usr/bin/env bash
# Four-run fixed-block PR-recovery suite.
#
# Goal:
#   Keep the strong z_L speaker cleaning from the 240L/16P fixed-block run, but
#   test whether PR degradation is caused by sustained anti-speaker pressure,
#   lack of residual capacity, or simply too large a constant z_L GRL push.
#
# Execution order, fail-fast:
#   1) decay_240L16P
#      topk_L/P/U=240/16/0, z_L speaker GRL target 2.5e-4 until 6k,
#      linear decay to 5e-5 from 6k→11k, then hold 5e-5 until 12k.
#
#   2) U24_with_adversaries
#      K_L/P/U=3680/1024/416, topk_L/P/U=216/16/24.
#      z_U has anti-speaker weight 0.5 and anti-phone weight 0.15.
#
#   3) U24_without_adversaries
#      Same U capacity, but no z_U adversaries.  This tells us whether U helps
#      as a harmless residual buffer or becomes an information escape hatch.
#
#   4) constant_gn00015_240L16P
#      topk_L/P/U=240/16/0, constant z_L speaker GRL target 1.5e-4.
#
# After each training run, probe the FINAL checkpoint only:
#   - checkpoint PR-head sanity, once
#   - PR direct and PR linear, source-by-source
#   - SID linear and SID stats
#
# Probe logs are grouped in the same blackwell/logs/<training_run>/ folder as
# the training log.  Probe JSON goes to <BLACKWELL_OUTPUT_ROOT>/<run>/probe_json.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

STAGE2_STEPS="${STAGE2_STEPS:-12000}"
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

run_train_and_probes() {
    if [[ $# -lt 11 ]]; then
        echo "Internal error: run_train_and_probes got too few arguments" >&2
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
    elif [[ "$SKIP_TRAINING" == "0" ]]; then
        blackwell_run "$train_run_name" "${train_command[@]}"
    else
        echo "ERROR: SKIP_TRAINING must be 0 or 1 (got: $SKIP_TRAINING)" >&2
        return 2
    fi

    if [[ ! -f "$final_ckpt" && -f "$train_ckpt_dir/final.pt" ]]; then
        final_ckpt="$train_ckpt_dir/final.pt"
    fi
    if [[ ! -f "$final_ckpt" ]]; then
        echo "ERROR: final checkpoint missing for $train_run_name: $final_ckpt" >&2
        return 3
    fi

    if [[ "$RUN_PROBES" == "0" ]]; then
        return 0
    elif [[ "$RUN_PROBES" != "1" ]]; then
        echo "ERROR: RUN_PROBES must be 0 or 1 (got: $RUN_PROBES)" >&2
        return 2
    fi

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
    RUN_DESCRIPTION="Checkpoint PR-head reconstruction sanity check for $train_run_name"
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

    local pr_source
    local pr_arch
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

run_train_and_probes \
    "libri_advfb_prrecover_decay_gn00025to00005_240L16P_aux64_12k_s42" \
    "PR-recovery ablation 1/4: fixed 240L/16P, z_L linear-mean speaker GRL target 2.5e-4 until 6k, decay to 5e-5 by 11k, hold to 12k; z_P phone GRL 0.2, AuxK 64" \
    4096 1024 0 \
    240 16 0 \
    0.00025 \
    "z_t z_L z_P" \
    "z_t,z_L,z_P" \
    --grl_grad_norm_decay_start 6000 \
    --grl_grad_norm_decay_end 11000 \
    --grl_grad_norm_final_target 0.00005

run_train_and_probes \
    "libri_advfb_prrecover_U24adv_gn00025_gp02_gu05_gpu015_aux64_12k_s42" \
    "PR-recovery ablation 2/4: fixed L/P/U with 24 active U units, z_U anti-speaker 0.5 and anti-phone 0.15, z_L linear-mean speaker GRL target 2.5e-4, z_P phone GRL 0.2, AuxK 64" \
    3680 1024 416 \
    216 16 24 \
    0.00025 \
    "z_t z_L z_P z_U" \
    "z_t,z_L,z_P,z_U" \
    --grl_u_weight 0.5 \
    --grl_phoneme_u_weight 0.15

run_train_and_probes \
    "libri_advfb_prrecover_U24open_gn00025_gp02_aux64_12k_s42" \
    "PR-recovery ablation 3/4: fixed L/P/U with 24 active U units, no z_U adversaries, z_L linear-mean speaker GRL target 2.5e-4, z_P phone GRL 0.2, AuxK 64" \
    3680 1024 416 \
    216 16 24 \
    0.00025 \
    "z_t z_L z_P z_U" \
    "z_t,z_L,z_P,z_U"

run_train_and_probes \
    "libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42" \
    "PR-recovery ablation 4/4: fixed 240L/16P, lower constant z_L linear-mean speaker GRL target 1.5e-4, z_P phone GRL 0.2, AuxK 64" \
    4096 1024 0 \
    240 16 0 \
    0.00015 \
    "z_t z_L z_P" \
    "z_t,z_L,z_P"
