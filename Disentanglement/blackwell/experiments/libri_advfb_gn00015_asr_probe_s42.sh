#!/usr/bin/env bash
# ASR diagnostic probe for the current best fixed-block model.
#
# Default target:
#   libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42
#
# This trains fresh character-CTC ASR probes on frozen z_t/z_L features.
# z_t is the full-SAE control; z_L is the linguistic route we care about.
# Override ASR_SOURCES=z_t,z_L,z_P if you also want the negative-control z_P run.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42}"
STAGE2_STEPS="${STAGE2_STEPS:-12000}"
PROBE_SEED="${PROBE_SEED:-42}"
ASR_SOURCES="${ASR_SOURCES:-z_t,z_L}"
ASR_PROBE_ARCH="${ASR_PROBE_ARCH:-linear}"
ASR_PROBE_LR="${ASR_PROBE_LR:-5e-4}"
ASR_PROBE_WARMUP="${ASR_PROBE_WARMUP:-500}"
ASR_MAX_EXAMPLES="${ASR_MAX_EXAMPLES:-0}"
PROBE_STEPS="${PROBE_STEPS:-10000}"
PROBE_VAL_EVERY="${PROBE_VAL_EVERY:-250}"
PROBE_PATIENCE="${PROBE_PATIENCE:-5}"

TRAIN_CKPT_DIR="$BLACKWELL_OUTPUT_ROOT/$TRAIN_RUN_NAME/checkpoints"
FINAL_CKPT="$TRAIN_CKPT_DIR/stage2_step${STAGE2_STEPS}.pt"
if [[ ! -f "$FINAL_CKPT" && -f "$TRAIN_CKPT_DIR/final.pt" ]]; then
    FINAL_CKPT="$TRAIN_CKPT_DIR/final.pt"
fi
[[ -f "$FINAL_CKPT" ]] || {
    echo "ERROR: checkpoint missing: $FINAL_CKPT" >&2
    exit 3
}

ASR_PROBE_RUN="diag_${TRAIN_RUN_NAME}_asr_${ASR_PROBE_ARCH}_seed${PROBE_SEED}"
RUN_DESCRIPTION="ASR diagnostic probe for $TRAIN_RUN_NAME: sources=${ASR_SOURCES}, arch=${ASR_PROBE_ARCH}, char-CTC WER/CER"
BLACKWELL_LOG_GROUP="$TRAIN_RUN_NAME"
export BLACKWELL_LOG_GROUP

ASR_PROBE_COMMAND=(
    python -u Disentanglement/diag_probe/run.py
    --stage2_ckpt "$FINAL_CKPT"
    --stage1_ckpt "$FINAL_CKPT"
    --run_name "$ASR_PROBE_RUN"
    --output_json "$BLACKWELL_OUTPUT_ROOT/$TRAIN_RUN_NAME/diagnostic-probes/asr/${ASR_PROBE_RUN}.json"
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
    --sources "$ASR_SOURCES"
    --tasks "asr"
    --asr_probe_arch "$ASR_PROBE_ARCH"
    --asr_probe_lr "$ASR_PROBE_LR"
    --asr_probe_warmup_steps "$ASR_PROBE_WARMUP"
    --asr_max_examples "$ASR_MAX_EXAMPLES"
    --probe_steps "$PROBE_STEPS"
    --probe_val_every "$PROBE_VAL_EVERY"
    --probe_patience "$PROBE_PATIENCE"
    --seed "$PROBE_SEED"
    --num_workers 2
)

blackwell_run "$ASR_PROBE_RUN" "${ASR_PROBE_COMMAND[@]}"
