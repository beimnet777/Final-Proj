#!/usr/bin/env bash
# Learned-routing adversarial run, then freeze the learned routing partition and
# continue training from the exact resume state.
#
# Purpose:
#   Test whether learned-routing failures are caused by the route assignment
#   moving while the adversaries/probes chase it.  This is NOT fixed-block
#   routing: the partition is learned for FREEZE_STEP updates, then frozen.
#
# Training:
#   K=5120 topk=256 learned hard two-route routing
#   z_L speaker adversary = linear-mean only
#   speaker GRL grad-norm target = 2e-4
#   grl_weight=1.0  grl_phoneme_weight=0.2
#   DANN full discriminator, n_disc_steps=3
#   AuxK=64, coef=1/32, dead threshold=256
#   lr_sid_head=1e-3, lr_min=1e-5
#   phase 1: learn routing until FREEZE_STEP
#   phase 2: exact-resume, freeze learned routing, continue to STAGE2_STEPS
#
# Probes:
#   PR-linear   : existing diagnostic projector probe, K->256->74
#   PR-direct   : training-head-matched fresh probe, K->74
#   SID-linear  : projector->mean-pool->linear
#   SID-stats   : projector->ReLU->mean+std->linear
#
# Runtime controls:
#   FREEZE_STEP=5000 PROBE_STEPS=5000 GPU_ID=0 ./.../this_script.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

STAGE2_STEPS="${STAGE2_STEPS:-12000}"
FREEZE_STEP="${FREEZE_STEP:-5000}"
if (( FREEZE_STEP <= 0 || FREEZE_STEP >= STAGE2_STEPS )); then
    echo "ERROR: FREEZE_STEP must be >0 and < STAGE2_STEPS; got FREEZE_STEP=${FREEZE_STEP}, STAGE2_STEPS=${STAGE2_STEPS}" >&2
    exit 2
fi

TRAIN_RUN_NAME="libri_advlearn_linmean_gn0002_gp02_aux64_freeze${FREEZE_STEP}_to${STAGE2_STEPS}_s42"
RUN_DESCRIPTION="Learned-routing then frozen-routing continuation: freeze at step ${FREEZE_STEP}, continue to ${STAGE2_STEPS}; z_L linear-mean GRL grad-norm target 2e-4, z_P phoneme GRL 0.2, AuxK 64"

PROBE_STEPS="${PROBE_STEPS:-10000}"
PROBE_VAL_EVERY="${PROBE_VAL_EVERY:-250}"
PROBE_PATIENCE="${PROBE_PATIENCE:-0}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PR_PROBE_LR="${PR_PROBE_LR:-1e-4}"
PROBE_SEED="${PROBE_SEED:-42}"

TRAIN_CKPT_DIR="$BLACKWELL_OUTPUT_ROOT/$TRAIN_RUN_NAME/checkpoints"
PROBE_JSON_DIR="$BLACKWELL_OUTPUT_ROOT/$TRAIN_RUN_NAME/probe_json"
ROUTE_CKPT="$TRAIN_CKPT_DIR/latest-resume.pt"
FINAL_CKPT="$TRAIN_CKPT_DIR/stage2_step${STAGE2_STEPS}.pt"

COMMON_TRAIN_ARGS=(
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
    --grl_grad_norm_target 0.0002
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
    --checkpoint_dir "$TRAIN_CKPT_DIR"
    --runs_dir "$BLACKWELL_OUTPUT_ROOT/$TRAIN_RUN_NAME/tensorboard"
    --log_dir "$BLACKWELL_OUTPUT_ROOT/$TRAIN_RUN_NAME/trainer_logs"
    --num_workers 2
    --seed 42
)

LEARN_ROUTE_RUN="${TRAIN_RUN_NAME}_learn_to_${FREEZE_STEP}"
RUN_DESCRIPTION="Phase 1: learned hard routing until step ${FREEZE_STEP}; then save latest-resume.pt for frozen continuation"
LEARN_ROUTE_COMMAND=(
    python -u Disentanglement/run.py
    "${COMMON_TRAIN_ARGS[@]}"
    --segment_steps "$FREEZE_STEP"
    --resume_every 500
)
blackwell_run "$LEARN_ROUTE_RUN" "${LEARN_ROUTE_COMMAND[@]}"

[[ -f "$ROUTE_CKPT" ]] || {
    echo "ERROR: freeze checkpoint missing: $ROUTE_CKPT" >&2
    exit 3
}

FROZEN_CONTINUE_RUN="${TRAIN_RUN_NAME}_freeze_continue"
RUN_DESCRIPTION="Phase 2: exact-resume from step ${FREEZE_STEP}, freeze learned routing deterministically, continue to ${STAGE2_STEPS}"
FROZEN_CONTINUE_COMMAND=(
    python -u Disentanglement/run.py
    "${COMMON_TRAIN_ARGS[@]}"
    --resume "$ROUTE_CKPT"
    --freeze_learned_routing_on_resume
    --resume_every 500
)
blackwell_run "$FROZEN_CONTINUE_RUN" "${FROZEN_CONTINUE_COMMAND[@]}"

if [[ ! -f "$FINAL_CKPT" && -f "$TRAIN_CKPT_DIR/final.pt" ]]; then
    FINAL_CKPT="$TRAIN_CKPT_DIR/final.pt"
fi
[[ -f "$FINAL_CKPT" ]] || {
    echo "ERROR: final checkpoint missing: $FINAL_CKPT" >&2
    exit 4
}

COMMON_PROBE_ARGS=(
    --stage2_ckpt "$FINAL_CKPT"
    --stage1_ckpt "$FINAL_CKPT"
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
    --seed "$PROBE_SEED"
)

for PR_ARCH in linear direct; do
    for PR_SOURCE in z_t z_L z_P; do
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
            --pr_probe_lr "$PR_PROBE_LR"
            --pr_max_examples 0
        )
        blackwell_run "$PR_PROBE_RUN" "${PR_PROBE_COMMAND[@]}"
    done
done

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
