#!/usr/bin/env bash
# Dose-bump + AuxK follow-up to libri_clubhybrid_proj256_warm1k_lite_s42.
#
# Two scientific changes vs libri_clubhybrid_proj256_warm1k_lite_s42:
#
#   1. --club_grad_norm_target 0.001 -> 0.003
#      Delivered per-frame CLUB gradient on z_L: 3e-4 -> 9e-4 (3x).
#      Reference: Job 2's dense speaker adversary targeted 1e-3/frame
#      (CLAUDE.md §3), so 9e-4 sits just below that historical operating
#      point; grl_p on z_P currently delivers ~2.8e-4/frame.
#
#   2. --aux_k 64 --aux_k_coef 0.03125  (Gao et al. 2024 AuxK)
#      aux_k=64 rather than D/2=640 so the mechanism engages EARLY (around
#      step 2000 based on prior run's death trajectory) instead of waiting
#      until step ~5500 when 640 latents are already dead. Revival capacity
#      at 64/step is ~250x the observed peak death rate (0.25/step), which
#      keeps a factor-2 margin against death-rate surprises from the 3x
#      CLUB dose bump. dead_steps_threshold kept at the default 256 steps
#      (~1.3M valid tokens at ~5k tokens/step). Target failure mode from
#      the prior run: dead % climbed 0% -> 20% by step 8700 while recon
#      rose 0.126 -> 0.215.
#
# All other flags are identical to libri_clubhybrid_proj256_warm1k_lite_s42
# so any change in z_L val PR / z_P val SID / recon / dead % is
# attributable to (1)+(2). We are NOT running dose bump in isolation and
# AuxK in isolation — attribution between the two would require additional
# arms if the outcome moves in a way we care about.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

RUN_NAME="libri_clubhybrid_proj256_warm800_gnt003_auxk_s42"
RUN_DESCRIPTION="LibriSpeech CLUB-hybrid with 3x delivered CLUB gradient on z_L (grad_norm_target 0.001 -> 0.003) AND AuxK dead-latent revival on (aux_k=64 for early activation, coef=0.03125, dead_threshold=256). Otherwise identical to libri_clubhybrid_proj256_warm1k_lite_s42; seed=42"

COMMAND=(
    python -u Disentanglement/run.py
    --stage 2
    --stage2_from_scratch
    --n_routes 2
    --no-hard_gumbel_routing
    --gumbel_tau_start 1.0
    --gumbel_tau_end 1.0
    --routing_init_std 0.5
    --local_data
    --librispeech_root "$BLACKWELL_DATA_ROOT/LibriSpeech"
    --lexicon_path "$BLACKWELL_DATA_ROOT/librispeech-lexicon.txt"
    --train_split_dir train-clean-100
    --speaker_stratified_holdout
    --spear_layernorm
    --K 5120
    --topk 256
    # AuxK dead-latent revival (Gao et al. 2024). Reconstruct residual using
    # the top-aux_k DEAD latents; that path gives dead latents gradient so
    # they don't stay dead. aux_k=64 chosen for early activation (triggers
    # around step ~2000 based on prior run's death trajectory); revival
    # capacity ~250x peak death rate; dead_steps_threshold left at default
    # 256 steps (~1.3M tokens at ~5k valid tokens/step).
    --aux_k 64
    --aux_k_coef 0.03125
    --dual_invariance
    --inv_L_weight 0.0
    --inv_P_weight 1.0
    --inv_var_weight 0.1
    --inv_var_gamma 1.0
    --vicreg_full
    --vicreg_cov_weight 0.0
    --club_enabled
    --club_weight 0.3
    --club_inner_steps 15
    --club_hidden 512
    --club_lr 1e-3
    --club_grad_norm
    # 3x delivered CLUB gradient on z_L vs the previous run.
    --club_grad_norm_target 0.003
    --club_projection_dim 256
    --club_warmup_steps 800
    --club_pretrain_inner_steps 20
    --club_no_collision_negatives
    --club_full_diagnostics
    --club_diagnostics_every 100
    --pair_alpha_arctic_w 0.0
    --pair_alpha_pert_w 1.0
    --pair_beta_libri_w 1.0
    --pairs_alpha_per_step 8
    --pairs_beta_per_step 8
    --inv_L_interp_frames 200
    --inv_f0_low 0.7
    --inv_f0_high 1.5
    --inv_formant_low 0.85
    --inv_formant_high 1.3
    --alpha 0.8
    --beta 0.6
    --grl_weight 0.0
    --grl_phoneme_weight 0.2
    --rho 0.001
    --routing_spec_weight 0.01
    --stage2_steps 10000
    --warmup_steps 500
    --batch_size 16
    --eval_batch_size 32
    --lr 1e-4
    --lr_min 1e-6
    --lr_heads 1e-4
    --lr_sid_head 5e-4
    --lr_routing 1e-3
    --weight_decay 1e-4
    --grad_clip 1.0
    --grad_log_every 200
    --log_every 100
    --ckpt_every 1000
    --checkpoint_dir "$BLACKWELL_OUTPUT_ROOT/$RUN_NAME/checkpoints"
    --runs_dir "$BLACKWELL_OUTPUT_ROOT/$RUN_NAME/tensorboard"
    --log_dir "$BLACKWELL_OUTPUT_ROOT/$RUN_NAME/trainer_logs"
    --num_workers 2
    --seed 42
)

blackwell_run "$RUN_NAME" "${COMMAND[@]}"
