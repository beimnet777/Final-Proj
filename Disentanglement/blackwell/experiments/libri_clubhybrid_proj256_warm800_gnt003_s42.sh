#!/usr/bin/env bash
# Isolated dose-bump follow-up to libri_clubhybrid_proj256_warm1k_lite_s42.
#
# Sole scientific change: --club_grad_norm_target 0.001 -> 0.003.
#   Delivered per-frame CLUB gradient on z_L: 3e-4 -> 9e-4 (3x).
#   Equivalent to what would be delivered at raw club_weight ~1.0 with target 0.001.
#   Reference point: Job 2's dense speaker adversary targeted 1e-3/frame
#   (documented in CLAUDE.md §3), so 9e-4 sits just below that historical
#   operating point. `grl_p` on z_P currently delivers ~2.8e-4/frame, so at
#   9e-4 CLUB on z_L is delivering ~3x what the working phoneme adversary
#   delivers on z_P.
#
# All other flags are identical to libri_clubhybrid_proj256_warm1k_lite_s42
# so any difference in outcome is attributable to the dose bump alone. In
# particular this run keeps inv_L_weight=0 and vicreg_cov_weight=0 — we are
# NOT combining the dose bump with inv_L, on purpose, to isolate whether
# raw magnitude was the CLUB bottleneck.
#
# What we expect this to tell us (from the diagnostic step-1500 read):
#   - `q_phi` accuracy trajectory: should climb more slowly if magnitude was
#     the bottleneck; unchanged if q_phi's inner race is winning regardless.
#   - `club` bound trajectory: should stop climbing (or drop) if the dose is
#     now enough to actually reduce log(q_pos) - log(q_neg); keep climbing if
#     the direction q_phi provides is intrinsically thin/noisy.
#   - `z_L val PR`: should stay near the current run's 0.123 best if the
#     direction is aligned with speaker (not phoneme). If it degrades sharply,
#     the CLUB direction is partially co-linear with phoneme content and
#     amplifying it damages phoneme retention.
#   - Encoder/geometry: watch zL_rms, dead %, recon. 3x the dose on z_L
#     could over-drive the encoder into a different equilibrium.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

RUN_NAME="libri_clubhybrid_proj256_warm800_gnt003_s42"
RUN_DESCRIPTION="LibriSpeech CLUB-hybrid with 3x delivered CLUB gradient magnitude on z_L (club_grad_norm_target 0.001 -> 0.003, delivered 9e-4/frame). Otherwise identical to libri_clubhybrid_proj256_warm1k_lite_s42 (projection_dim=256, warmup=800, pretrain_inner=20, no-collision negatives, VICReg cov and inv_L disabled); seed=42"

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
    # SOLE SCIENTIFIC CHANGE vs libri_clubhybrid_proj256_warm1k_lite_s42:
    # delivered per-frame CLUB gradient on z_L bumped 3x (0.001 -> 0.003).
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
