#!/usr/bin/env bash
# Fixed-block non-CLUB adversarial run with a lighter paralinguistic active
# budget than the 224L/32P run.
#
# Purpose:
#   Test whether the previous fixed-block run cleaned speaker too aggressively
#   by reserving too many active units for z_P.  This keeps structural protection
#   against P collapse, but gives z_L more per-frame capacity:
#
#     topk_L/topk_P/topk_U = 240/16/0
#
# Training:
#   K=5120 topk=256 fixed_blocks per_block_topk
#   K_L/K_P/K_U = 4096/1024/0
#   z_L speaker adversary = linear-mean only
#   speaker GRL grad-norm target = 2.5e-4
#   grl_weight=1.0  grl_phoneme_weight=0.2
#   DANN full discriminator, n_disc_steps=3
#   AuxK=64, coef=1/32, dead threshold=256
#   lr_sid_head=1e-3, lr_min=1e-5, stage2_steps=12000
#
# Probes on final checkpoint:
#   PR-linear   : existing diagnostic projector probe, K->256->74
#   PR-direct   : training-head-matched fresh probe, K->74
#   SID-linear  : projector->mean-pool->linear
#   SID-stats   : projector->ReLU->mean+std->linear
#
# PR probes are split source-by-source to avoid large PR feature-cache kills.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

TRAIN_RUN_NAME="libri_advfb4096_1024_linmean_gn00025_gp02_aux64_240L16P_12k_s42"
RUN_DESCRIPTION="Fixed-block adversarial LibriSpeech run: topk_L/P/U=240/16/0, z_L linear-mean GRL grad-norm target 2.5e-4, z_P phoneme GRL 0.2, AuxK 64"

STAGE2_STEPS=12000
PROBE_STEPS="${PROBE_STEPS:-10000}"
PROBE_VAL_EVERY="${PROBE_VAL_EVERY:-250}"
PROBE_PATIENCE="${PROBE_PATIENCE:-0}"
PROBE_WARMUP_STEPS="${PROBE_WARMUP_STEPS:-0}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PR_PROBE_LR="${PR_PROBE_LR:-1e-4}"
PROBE_SEED="${PROBE_SEED:-42}"
SKIP_TRAINING="${SKIP_TRAINING:-0}"
PR_SANITY_ONLY="${PR_SANITY_ONLY:-0}"
PR_ARCHES="${PR_ARCHES:-linear direct}"
PR_SOURCES="${PR_SOURCES:-z_t z_L z_P}"
RUN_SID_PROBES="${RUN_SID_PROBES:-1}"

TRAIN_CKPT_DIR="$BLACKWELL_OUTPUT_ROOT/$TRAIN_RUN_NAME/checkpoints"
PROBE_JSON_DIR="$BLACKWELL_OUTPUT_ROOT/$TRAIN_RUN_NAME/probe_json"
FINAL_CKPT="$TRAIN_CKPT_DIR/stage2_step${STAGE2_STEPS}.pt"
if [[ ! -f "$FINAL_CKPT" && -f "$TRAIN_CKPT_DIR/final.pt" ]]; then
    FINAL_CKPT="$TRAIN_CKPT_DIR/final.pt"
fi

TRAIN_COMMAND=(
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
    --K_L 4096
    --K_P 1024
    --K_U 0
    --topk_L 240
    --topk_P 16
    --topk_U 0
    --grl_linear_mean
    --grl_grad_norm
    --grl_grad_norm_target 0.00025
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
    --checkpoint_dir "$TRAIN_CKPT_DIR"
    --runs_dir "$BLACKWELL_OUTPUT_ROOT/$TRAIN_RUN_NAME/tensorboard"
    --log_dir "$BLACKWELL_OUTPUT_ROOT/$TRAIN_RUN_NAME/trainer_logs"
    --num_workers 2
    --seed 42
)

if [[ "$SKIP_TRAINING" == "1" ]]; then
    echo "Skipping training; reusing checkpoint: $FINAL_CKPT"
elif [[ "$SKIP_TRAINING" == "0" ]]; then
    blackwell_run "$TRAIN_RUN_NAME" "${TRAIN_COMMAND[@]}"
else
    echo "ERROR: SKIP_TRAINING must be 0 or 1 (got: $SKIP_TRAINING)" >&2
    exit 2
fi

if [[ ! -f "$FINAL_CKPT" && -f "$TRAIN_CKPT_DIR/final.pt" ]]; then
    FINAL_CKPT="$TRAIN_CKPT_DIR/final.pt"
fi
[[ -f "$FINAL_CKPT" ]] || {
    echo "ERROR: final checkpoint missing: $FINAL_CKPT" >&2
    exit 3
}

# Keep all probe logs beside the training log. Individual filenames retain the
# probe run name, so architectures and sources remain distinguishable.
export BLACKWELL_LOG_GROUP="$TRAIN_RUN_NAME"

COMMON_PROBE_ARGS=(
    --stage2_ckpt "$FINAL_CKPT"
    --stage1_ckpt "$FINAL_CKPT"
    --local_data
    --librispeech_root "$BLACKWELL_DATA_ROOT/LibriSpeech"
    --lexicon_path "$BLACKWELL_DATA_ROOT/librispeech-lexicon.txt"
    --spear_layernorm
    --topk 256
    --fixed_blocks
    --per_block_topk
    --K_L 4096
    --K_P 1024
    --K_U 0
    --topk_L 240
    --topk_P 16
    --topk_U 0
    --probe_steps "$PROBE_STEPS"
    --probe_val_every "$PROBE_VAL_EVERY"
    --probe_patience "$PROBE_PATIENCE"
    --probe_warmup_steps "$PROBE_WARMUP_STEPS"
    --seed "$PROBE_SEED"
)

if [[ "$PR_SANITY_ONLY" == "1" ]]; then
    SANITY_RUN="diag_${TRAIN_RUN_NAME}_checkpoint_pr_sanity"
    RUN_DESCRIPTION="Checkpoint PR-head reconstruction sanity check for $TRAIN_RUN_NAME"
    SANITY_COMMAND=(
        python -u Disentanglement/diag_probe/run.py
        "${COMMON_PROBE_ARGS[@]}"
        --run_name "$SANITY_RUN"
        --sources "z_L"
        --tasks "pr"
        --pr_sanity_only
        --pr_max_examples 0
    )
    blackwell_run "$SANITY_RUN" "${SANITY_COMMAND[@]}"
    exit $?
elif [[ "$PR_SANITY_ONLY" != "0" ]]; then
    echo "ERROR: PR_SANITY_ONLY must be 0 or 1 (got: $PR_SANITY_ONLY)" >&2
    exit 2
fi

read -r -a PR_ARCH_LIST <<< "$PR_ARCHES"
read -r -a PR_SOURCE_LIST <<< "$PR_SOURCES"
for PR_ARCH in "${PR_ARCH_LIST[@]}"; do
    [[ "$PR_ARCH" == "linear" || "$PR_ARCH" == "direct" ]] || {
        echo "ERROR: unsupported PR_ARCHES entry: $PR_ARCH" >&2
        exit 2
    }
    for PR_SOURCE in "${PR_SOURCE_LIST[@]}"; do
        [[ "$PR_SOURCE" == "z_t" || "$PR_SOURCE" == "z_L" || "$PR_SOURCE" == "z_P" ]] || {
            echo "ERROR: unsupported PR_SOURCES entry: $PR_SOURCE" >&2
            exit 2
        }
        PR_PROBE_RUN="diag_${TRAIN_RUN_NAME}_${PR_SOURCE}_pr_${PR_ARCH}_seed${PROBE_SEED}"
        RUN_DESCRIPTION="Final-checkpoint PR-${PR_ARCH} probe for $TRAIN_RUN_NAME: source ${PR_SOURCE}"
        PR_PROBE_COMMAND=(
            python -u Disentanglement/diag_probe/run.py
            "${COMMON_PROBE_ARGS[@]}"
            --run_name "$PR_PROBE_RUN"
            --output_json "$PROBE_JSON_DIR/${PR_PROBE_RUN}.json"
            --sources "$PR_SOURCE"
            --tasks "pr"
            --pr_probe_arch "$PR_ARCH"
            --pr_max_examples 0
            --pr_probe_lr "$PR_PROBE_LR"
        )
        blackwell_run "$PR_PROBE_RUN" "${PR_PROBE_COMMAND[@]}"
    done
done

if [[ "$RUN_SID_PROBES" == "0" ]]; then
    exit 0
elif [[ "$RUN_SID_PROBES" != "1" ]]; then
    echo "ERROR: RUN_SID_PROBES must be 0 or 1 (got: $RUN_SID_PROBES)" >&2
    exit 2
fi

for SID_ARCH in linear stats; do
    SID_PROBE_RUN="diag_${TRAIN_RUN_NAME}_ztzLzP_sid_${SID_ARCH}_seed${PROBE_SEED}"
    RUN_DESCRIPTION="Final-checkpoint SID-${SID_ARCH} probe for $TRAIN_RUN_NAME: sources z_t,z_L,z_P"
    SID_PROBE_COMMAND=(
        python -u Disentanglement/diag_probe/run.py
        "${COMMON_PROBE_ARGS[@]}"
        --run_name "$SID_PROBE_RUN"
        --output_json "$PROBE_JSON_DIR/${SID_PROBE_RUN}.json"
        --sources "z_t,z_L,z_P"
        --tasks "sid"
        --sid_probe_arch "$SID_ARCH"
        --sid_probe_lr "$SID_PROBE_LR"
    )
    blackwell_run "$SID_PROBE_RUN" "${SID_PROBE_COMMAND[@]}"
done
