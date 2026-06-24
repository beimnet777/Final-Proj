#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=15:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=dual_inv_v1_soft_grl
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/dual_inv_v1/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/dual_inv_v1/%x_%j.err

# v1 dual-invariance, SOFT learned routing, WITH GRL (speaker on z_L, phoneme on z_P).
#   GRL head: default MLP (projector -> ReLU -> mean-pool -> linear)
#             — NOT stats_pool, NOT dense_context, NOT grad_norm
#               (Job 2's grad_norm was implicated in dose collapse; this run
#                relies on overall grad_clip=1.0 instead.)
#   weights : grl_weight=0.3  (z_L speaker), grl_phoneme_weight=0.2  (z_P phoneme)
#               — lighter than Job 2 (1.0/0.5) because invariance is the primary
#                 signal here and GRL is a safety net, not the main mechanism.
#   disc    : 1 step / iter, lr_disc 1e-3

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

mkdir -p "${DIS_DIR}/logs/train/stage2/dual_inv_v1"
cd "${DIS_DIR}"

RUN_NAME="dual_inv_v1_soft_grl"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"

echo "=== ${RUN_NAME}: dual-invariance v1, SOFT learned routing, GRL on z_L and z_P ==="
echo "started : $(date)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "routing : soft Gumbel-softmax, tau=1.0 (no anneal)"
echo "n_routes: 2 (z_L, z_P — no z_U)"
echo "pair α  : ARCTIC 60% + perturbed-LibriSpeech 40%"
echo "pair β  : within-chapter LibriSpeech"
echo "GRL z_L : weight=0.3  head=MLP mean-pool  (no grad_norm, no dense_context)"
echo "GRL z_P : weight=0.2  head=MLP mean-pool"
echo "grad_clip=1.0  disc_lr=1e-3  n_disc_steps=1"

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
    --pair_alpha_arctic_w 0.6 --pair_alpha_pert_w 0.4 \
    --pair_beta_libri_w 1.0 \
    --pairs_alpha_per_step 8 --pairs_beta_per_step 8 \
    --inv_L_interp_frames 200 \
    --inv_f0_low 0.7  --inv_f0_high 1.5 \
    --inv_formant_low 0.85 --inv_formant_high 1.3 \
    --alpha 0.8 --beta 0.6 \
    --grl_weight 0.3 --grl_phoneme_weight 0.2 \
    --lr_disc 1e-3 --n_disc_steps 1 \
    --rho 0.001 --routing_spec_weight 0.01 \
    --stage2_steps 12000 --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --lr_routing 1e-3 \
    --grad_clip 1.0 \
    --grad_log_every 200 --log_every 100 \
    --checkpoint_dir "${CKPT_DIR}" \
    --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" \
    --num_workers 2 \
    --seed 42

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: ${STAGE2_CKPT} missing" >&2; exit 3; }

# ----- diagnostic probes (TWO arch variants, same checkpoint) -----
PROBE_COMMON=(
    --stage2_ckpt "${STAGE2_CKPT}"
    --stage1_ckpt "${STAGE2_CKPT}"
    --no-hard_gumbel_routing
    --gumbel_tau_end 1.0
    --spear_layernorm
    --sources "z_L,z_P" --tasks "pr,sid"
    --probe_steps 10000 --probe_val_every 250 --probe_patience 8
    --pr_probe_lr 5e-4 --sid_probe_lr 1e-3
    --probe_warmup_steps 0 --seed 42
)

echo
echo "----- diag probe A: SUPERB linear (no non-linearity) -----"
date
${PYTHON} -u diag_probe/run.py \
    "${PROBE_COMMON[@]}" \
    --run_name "diag_${RUN_NAME}_linear" \
    --sid_probe_arch linear --pr_probe_arch linear

echo
echo "----- diag probe B: SUPERB + one ReLU (mlp) -----"
date
${PYTHON} -u diag_probe/run.py \
    "${PROBE_COMMON[@]}" \
    --run_name "diag_${RUN_NAME}_mlp" \
    --sid_probe_arch mlp --pr_probe_arch mlp

echo "Finished: $(date)"
