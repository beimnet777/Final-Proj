#!/usr/bin/env bash
# LibriSpeech CLUB-hybrid follow-up incorporating the diagnosed fixes from
# `libri_clubhybrid_vicreg_softtau1_clubgn001_grlp02_s42` and the hard-tau
# full-diag run:
#
#   1. Drop VICReg covariance regulariser (cov_weight 0.2 -> 0). The cov term
#      exploded from 0.014 to 292 in the soft-tau run and was fighting the
#      SAE geometry, not the CLUB objective.
#   2. Drop dual-invariance on z_L (inv_L_weight 1.0 -> 0). inv_L rose from
#      0.03 to 9.97 alongside the cov blow-up; it was reshaping z_L geometry
#      in ways that let q_phi keep finding speaker structure.
#   3. Learned projection prepended to q_phi: 10240-d sparse pool -> 256-d
#      code before the classifier hidden stack. Matches VQMIVC / Mun 2022
#      practice of feeding CLUB a small projection rather than a raw high-dim
#      sparse latent.
#   4. CLUB warmup + q_phi pretraining. Effective CLUB weight is held at 0
#      for the first 1000 optimiser steps while q_phi trains with 20 inner
#      CE steps per accumulation boundary. After warmup the loss weight
#      switches on and inner_steps drops to 15. This matches Cheng 2020
#      Algorithm 1: converge q_phi before descending its bound.
#   5. CLUB weight left at 0.3 (softtau1 operating point) but inner_steps
#      raised 3 -> 15. Cutting the weight was recommended earlier by analogy
#      to VQMIVC's mi_weight=0.01, but that analogy compares Gaussian CLUB on
#      continuous latents to our categorical vCLUB-S on 251-way logits;
#      bound magnitudes are on different natural scales. The softtau1 dose
#      of 3e-4 per frame (0.3 * grad_norm_target 0.001) was already the
#      configuration that reached best-observed z_L PR=0.15 at step 2000.
#      Direction quality — not magnitude — is what the other five changes
#      are supposed to fix.
#   6. Zero-collision negative labels. Removes the ~6% floor observed in the
#      hard-tau full-diag run when using in-batch random permutation.
#
# Held constant vs the softtau1 baseline (so the effect of the above is
# attributable): SPEAR encoder + LayerNorm, K=5120, TopK=256, soft two-route
# learned routing at tau=1.0, phoneme GRL weight 0.2, seed 42, 10k steps,
# batch 16.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

RUN_NAME="libri_clubhybrid_proj256_warm1k_lite_s42"
RUN_DESCRIPTION="LibriSpeech CLUB-hybrid with q_phi projection_dim=256, 1k-step CLUB warmup with 20 pretrain inner steps, no-collision negatives, weight cut to 0.02, VICReg cov and inv_L disabled; seed=42"

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
    --dual_invariance
    # Fix 2: drop invariance on z_L; keep z_P side.
    --inv_L_weight 0.0
    --inv_P_weight 1.0
    --inv_var_weight 0.1
    --inv_var_gamma 1.0
    --vicreg_full
    # Fix 1: drop VICReg covariance regulariser.
    --vicreg_cov_weight 0.0
    --club_enabled
    # Weight held at 0.3 (the softtau1 operating point that produced the best
    # z_L PR=0.15 at step 2000). With --club_grad_norm on, this sets delivered
    # per-frame magnitude on z_L to 0.3*0.001=3e-4, matching the old run's
    # dose. Direction quality is what the other five changes (projection,
    # warmup, no-collision, cov=0, inv_L=0) are supposed to fix.
    --club_weight 0.3
    --club_inner_steps 15
    --club_hidden 512
    --club_lr 1e-3
    --club_grad_norm
    --club_grad_norm_target 0.001
    # Fix 3: 10240 -> 256 learned projection before q_phi's classifier stack.
    --club_projection_dim 256
    # Fix 4: hold CLUB loss at 0 for 1000 steps; boost q_phi to 20 inner steps
    # during warmup so it converges to p(speaker|z) before we descend its bound.
    --club_warmup_steps 800
    --club_pretrain_inner_steps 20
    # Fix 6: rejection-sample the shuffled negatives; no collisions.
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
