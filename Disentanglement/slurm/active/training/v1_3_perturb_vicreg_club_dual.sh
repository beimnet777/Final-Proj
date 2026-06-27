#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=16:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=v1_3_pert_vicreg_club_grlp
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/v1_3/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/v1_3/%x_%j.err

# v1.3 — HYBRID: perturbation-only pair-α + VICReg-full
#        + CLUB MI-min on z_L (speaker)      <- probe-architecture-agnostic side
#        + GRL  CTC    on z_P (phoneme)      <- classic adversary side
#
# Why hybrid (user choice, 2026-06-26):
#   Phoneme CLUB on z_P was dropped for v1.3 to keep debugging focused on the
#   speaker CLUB cold-start fix.  z_P phoneme leakage is suppressed by the
#   battle-tested GRL_p path (Job 1/2 used grl_phoneme_weight=0.2) instead.
#   Trade-off: the probe-architecture-agnostic dissertation claim now applies
#   ONLY to the speaker side (z_L / Fano).  The phoneme side is "we beat the
#   stats-pool CTC probe", same caveat as Job 2.  The hybrid is the minimum
#   change to get the speaker CLUB validated end-to-end before scaling.
#
# q_phi fix (already in code, not flags):
#   v1.2 (runs 31044101 / 31045234) wedged q_phi at uniform output:
#     Default-init Linear(10240, 256) on sparse z_pool gives O(1e-3) pre-acts
#     -> softmax uniform -> CLUB bound = 0 exactly -> zero gradient -> stuck.
#   Fix:
#     LayerNorm(in_dim) -> Linear(in,512) -> GELU -> LayerNorm(512) -> Linear(512,C)
#     hidden default bumped 256 -> 512.
#     bf16 retained throughout (LayerNorm puts pre-acts at O(1), well within bf16).
#
# Evaluation gate (single moderately-strong probe, decide before scaling):
#   stats-pool, patience=0, ARCTIC matched (18 speakers) + LibriSpeech (251).
#   Joint success criterion:
#     z_L PR ≤ 0.10  AND  z_L SID < 0.10  on BOTH Libri and ARCTIC   (CLUB target)
#     z_P PR ≥ 0.50  AND  z_P SID > 0.90  on Libri                   (GRL_p target)
#   If met -> expand to multi-seed × multi-arch + an independent MI estimator
#   on frozen z_L to corroborate the probe-agnostic claim for the speaker side.

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"

mkdir -p "${DIS_DIR}/logs/train/stage2/v1_3"
cd "${DIS_DIR}"

RUN_NAME="v1_3_pert_vicreg_club_grlp"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_FINAL="${CKPT_DIR}/stage2_step10000.pt"
ARCTIC_ROOT="${DIS_DIR}/../Probing/data/CMU_ARCTIC"

echo "=== ${RUN_NAME}: perturbation-only VICReg-full + CLUB (z_L speaker) + GRL_p (z_P phoneme) ==="
echo "started : $(date)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "routing : soft Gumbel-softmax, tau=1.0 (no anneal); n_routes=2 (no z_U)"
echo "pair α  : perturbed-LibriSpeech 100%  (ARCTIC DROPPED — no bilinear-align hack)"
echo "pair β  : within-chapter LibriSpeech"
echo "inv_L   : VICReg L2 per-frame (frame-aligned only)"
echo "cov     : VICReg covariance reg on z_L,z_P bucket dims (weight=0.2)"
echo "CLUB-spk: I(stats_pool(z_L); speaker_id) min, weight=0.3, inner=3, lr=1e-3, hidden=512"
echo "CLUB-phn: OFF (dropped for v1.3 — debugging focus on speaker CLUB cold-start)"
echo "GRL-spk : OFF (CLUB handles speaker side)"
echo "GRL-phn : ON, weight=0.2 (classic CTC phoneme adversary on z_P)"

${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch \
    --n_routes 2 \
    --no-hard_gumbel_routing \
    --gumbel_tau_start 1.0 --gumbel_tau_end 1.0 \
    --routing_init_std 0.5 \
    --local_data --train_split_dir train-clean-100 --spear_layernorm \
    --K 5120 --topk 256 \
    --dual_invariance \
    --inv_L_weight 1.0 --inv_P_weight 1.0 \
    --inv_var_weight 0.1 --inv_var_gamma 1.0 \
    --vicreg_full --vicreg_cov_weight 0.2 \
    --club_enabled --club_weight 0.3 --club_inner_steps 3 \
    --club_hidden 512 --club_lr 1e-3 \
    --pair_alpha_arctic_w 0.0 --pair_alpha_pert_w 1.0 \
    --pair_beta_libri_w 1.0 \
    --pairs_alpha_per_step 8 --pairs_beta_per_step 8 \
    --inv_L_interp_frames 200 \
    --inv_f0_low 0.7  --inv_f0_high 1.5 \
    --inv_formant_low 0.85 --inv_formant_high 1.3 \
    --alpha 0.8 --beta 0.6 \
    --grl_weight 0.0 --grl_phoneme_weight 0.2 \
    --rho 0.001 --routing_spec_weight 0.01 \
    --stage2_steps 10000 --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --lr_sid_head 5e-4 --lr_routing 1e-3 \
    --grad_clip 1.0 \
    --grad_log_every 200 --log_every 100 \
    --checkpoint_dir "${CKPT_DIR}" \
    --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" \
    --num_workers 2 \
    --seed 42

[[ -f "${STAGE2_FINAL}" ]] || { echo "ERROR: ${STAGE2_FINAL} missing (training did not save final step?)" >&2; exit 3; }

# ---- End-of-run probes: moderately strong, single seed, two datasets ----
# Use stage2_step10000.pt (NOT stage2_best.pt — disent_score selector is biased;
# lesson from probe_final4 / v1 retrospection).
PROBE_COMMON=(
    --stage2_ckpt "${STAGE2_FINAL}"
    --stage1_ckpt "${STAGE2_FINAL}"
    --no-hard_gumbel_routing --gumbel_tau_end 1.0
    --spear_layernorm
    --sources "z_L,z_P" --tasks "pr,sid"
    --sid_probe_arch stats
    --probe_steps 10000 --probe_val_every 250 --probe_patience 0
    --pr_probe_lr 5e-4 --sid_probe_lr 1e-3
    --probe_warmup_steps 0 --seed 42
)

echo
echo "----- diag probe A: stats-pool, patience=0, LibriSpeech 251 speakers -----"
date
${PYTHON} -u diag_probe/run.py \
    "${PROBE_COMMON[@]}" \
    --run_name "diag_${RUN_NAME}_libri_stats_p0" \
    --sid_dataset libri

echo
echo "----- diag probe B: stats-pool, patience=0, ARCTIC 18 speakers (matched-distribution) -----"
date
${PYTHON} -u diag_probe/run.py \
    "${PROBE_COMMON[@]}" \
    --run_name "diag_${RUN_NAME}_arctic_stats_p0" \
    --sid_dataset arctic --arctic_root "${ARCTIC_ROOT}" --arctic_sid_seed 42

echo "Finished: $(date)"
