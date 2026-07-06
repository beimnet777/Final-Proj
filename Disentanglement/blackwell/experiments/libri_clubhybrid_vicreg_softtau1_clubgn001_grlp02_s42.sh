#!/usr/bin/env bash
# Full LibriSpeech CLUB-hybrid run with normalized speaker-CLUB gradients.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

RUN_NAME="libri_clubhybrid_vicreg_softtau1_clubgn001_grlp02_s42"
RUN_DESCRIPTION="LibriSpeech CLUB-hybrid + VICReg; soft routing tau=1; speaker CLUB grad target=0.001; phoneme GRL=0.2; seed=42"

# Scientific configuration:
#   - soft two-route learned routing
#   - pair-alpha/pair-beta dual invariance with VICReg variance + covariance
#   - speaker CLUB on z_L; speaker GRL off
#   - sign-preserving normalized CLUB gradient, target 0.001
#   - phoneme GRL on z_P; phoneme CLUB off
# This is an HPC-style direct trainer launch: every non-default scientific
# choice is recorded below, without the Colab/experiment_runner layer.
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
    --inv_L_weight 1.0
    --inv_P_weight 1.0
    --inv_var_weight 0.1
    --inv_var_gamma 1.0
    --vicreg_full
    --vicreg_cov_weight 0.2
    --club_enabled
    --club_weight 0.3
    --club_inner_steps 3
    --club_hidden 512
    --club_lr 1e-3
    --club_grad_norm
    --club_grad_norm_target 0.001
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
