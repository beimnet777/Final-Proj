#!/usr/bin/env bash
# Fixed-block non-CLUB adversarial run + targeted final-checkpoint probes.
#
# Training matches the requested override set:
#   K=5120 topk=256 fixed_blocks per_block_topk
#   K_L/K_P/K_U = 4096/1024/0
#   topk_L/topk_P/topk_U = 224/32/0
#   z_L speaker adversary = linear-mean only
#   speaker GRL grad-norm target = 2e-4
#   grl_weight=1.0  grl_phoneme_weight=0.2
#   DANN full discriminator, n_disc_steps=3
#   AuxK=64, coef=1/32, dead threshold=256
#   lr_sid_head=1e-3, lr_min=1e-5, stage2_steps=12000
#
# After training, probe the FINAL checkpoint on z_t/z_L/z_P:
#   1) SID-linear + PR-linear with the existing SUPERB-style 5120->256->74 probe
#   2) PR-direct with a training-head-matched fresh 5120->74 probe
#   3) SID-stats
#
# For PR, "linear" means the existing diagnostic projector probe
# (5120->256->74).  "direct" is the additional training-head-matched
# fresh probe (5120->74), with lr=5e-4 and 500-step PR warmup.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

TRAIN_RUN_NAME="libri_advfb4096_1024_linmean_gn0002_gp02_aux64_12k_s42"
RUN_DESCRIPTION="Fixed-block non-CLUB adversarial LibriSpeech run: K_L/K_P/K_U=4096/1024/0, topk_L/P/U=224/32/0, z_L linear-mean GRL grad-norm target 2e-4, z_P phoneme GRL 0.2, AuxK 64, final probes on z_t/z_L/z_P"

STAGE2_STEPS=12000
PROBE_STEPS="${PROBE_STEPS:-10000}"
PROBE_VAL_EVERY="${PROBE_VAL_EVERY:-250}"
PROBE_PATIENCE="${PROBE_PATIENCE:-0}"
SID_PROBE_LR="${SID_PROBE_LR:-1e-3}"
PR_PROBE_LR="${PR_PROBE_LR:-5e-4}"
PR_PROBE_WARMUP_STEPS="${PR_PROBE_WARMUP_STEPS:-500}"
PROBE_SEED="${PROBE_SEED:-42}"

TRAIN_CKPT_DIR="$BLACKWELL_OUTPUT_ROOT/$TRAIN_RUN_NAME/checkpoints"
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
    --topk_L 224
    --topk_P 32
    --topk_U 0
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

blackwell_run "$TRAIN_RUN_NAME" "${TRAIN_COMMAND[@]}"

if [[ ! -f "$FINAL_CKPT" && -f "$TRAIN_CKPT_DIR/final.pt" ]]; then
    FINAL_CKPT="$TRAIN_CKPT_DIR/final.pt"
fi
[[ -f "$FINAL_CKPT" ]] || {
    echo "ERROR: final checkpoint missing: $FINAL_CKPT" >&2
    exit 3
}

LINEAR_PROBE_RUN="diag_${TRAIN_RUN_NAME}_ztzLzP_prsid_linear_seed${PROBE_SEED}"
RUN_DESCRIPTION="Final-checkpoint probes for $TRAIN_RUN_NAME: sources z_t,z_L,z_P with SID-linear and PR-linear"
LINEAR_PROBE_COMMAND=(
    python -u Disentanglement/diag_probe/run.py
    --stage2_ckpt "$FINAL_CKPT"
    --stage1_ckpt "$FINAL_CKPT"
    --run_name "$LINEAR_PROBE_RUN"
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
    --topk_L 224
    --topk_P 32
    --topk_U 0
    --sources "z_t,z_L,z_P"
    --tasks "pr,sid"
    --sid_probe_arch linear
    --pr_probe_arch linear
    --probe_steps "$PROBE_STEPS"
    --probe_val_every "$PROBE_VAL_EVERY"
    --probe_patience "$PROBE_PATIENCE"
    --pr_max_examples 0
    --pr_probe_lr "$PR_PROBE_LR"
    --sid_probe_lr "$SID_PROBE_LR"
    --probe_warmup_steps 0
    --pr_probe_warmup_steps "$PR_PROBE_WARMUP_STEPS"
    --seed "$PROBE_SEED"
)
blackwell_run "$LINEAR_PROBE_RUN" "${LINEAR_PROBE_COMMAND[@]}"

for PR_SOURCE in z_t z_L z_P; do
    DIRECT_PR_PROBE_RUN="diag_${TRAIN_RUN_NAME}_${PR_SOURCE}_pr_direct_seed${PROBE_SEED}"
    RUN_DESCRIPTION="Final-checkpoint direct PR probe for $TRAIN_RUN_NAME: source ${PR_SOURCE}, fresh training-head-matched Linear(K->vocab)"
    DIRECT_PR_PROBE_COMMAND=(
        python -u Disentanglement/diag_probe/run.py
        --stage2_ckpt "$FINAL_CKPT"
        --stage1_ckpt "$FINAL_CKPT"
        --run_name "$DIRECT_PR_PROBE_RUN"
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
        --topk_L 224
        --topk_P 32
        --topk_U 0
        --sources "$PR_SOURCE"
        --tasks "pr"
        --pr_probe_arch direct
        --probe_steps "$PROBE_STEPS"
        --probe_val_every "$PROBE_VAL_EVERY"
        --probe_patience "$PROBE_PATIENCE"
        --pr_max_examples 0
        --pr_probe_lr "$PR_PROBE_LR"
        --probe_warmup_steps 0
        --pr_probe_warmup_steps "$PR_PROBE_WARMUP_STEPS"
        --seed "$PROBE_SEED"
    )
    blackwell_run "$DIRECT_PR_PROBE_RUN" "${DIRECT_PR_PROBE_COMMAND[@]}"
done

STATS_PROBE_RUN="diag_${TRAIN_RUN_NAME}_ztzLzP_sid_stats_seed${PROBE_SEED}"
RUN_DESCRIPTION="Final-checkpoint probes for $TRAIN_RUN_NAME: sources z_t,z_L,z_P with SID-stats"
STATS_PROBE_COMMAND=(
    python -u Disentanglement/diag_probe/run.py
    --stage2_ckpt "$FINAL_CKPT"
    --stage1_ckpt "$FINAL_CKPT"
    --run_name "$STATS_PROBE_RUN"
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
    --topk_L 224
    --topk_P 32
    --topk_U 0
    --sources "z_t,z_L,z_P"
    --tasks "sid"
    --sid_probe_arch stats
    --probe_steps "$PROBE_STEPS"
    --probe_val_every "$PROBE_VAL_EVERY"
    --probe_patience "$PROBE_PATIENCE"
    --sid_probe_lr "$SID_PROBE_LR"
    --probe_warmup_steps 0
    --seed "$PROBE_SEED"
)
blackwell_run "$STATS_PROBE_RUN" "${STATS_PROBE_COMMAND[@]}"
