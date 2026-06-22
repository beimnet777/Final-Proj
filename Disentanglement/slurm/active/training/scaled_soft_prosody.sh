#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=scaled_soft_pros
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/scaled_soft_prosody/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/scaled_soft_prosody/%x_%j.error

# Scaled, LEARNED SOFT routing (Gumbel-softmax, fractional masks) + PROSODY.
# Soft routing ALONE starved z_P in scaled_3partition (same emergent failure as
# global top-k), so this pairs it with the per-frame prosody task: can a *learned*
# partition keep z_P alive when prosody gives the router a frame-level signal to
# latch onto?  Anti-prosody adversaries push F0/energy into z_P.  Watch actP and
# the routing membership L/P/U: does P grow + stay active instead of collapsing?

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
mkdir -p "${DIS_DIR}/logs/train/stage2/scaled_soft_prosody"
cd "${DIS_DIR}"

SEED="${SEED:-42}"
STAGE2_STEPS="${STAGE2_STEPS:-26000}"
PROBE_STEPS="${PROBE_STEPS:-8000}"
RUN_NAME="scaled_soft_prosody"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"
# Learned SOFT routing (no hard Gumbel, softmax masks); router learns membership.
ROUTING=(--n_routes 3 --no-hard_gumbel_routing --gumbel_tau_start 1.0 --gumbel_tau_end 0.5
         --routing_init_std 0.5 --routing_spec_weight 0.0 --lr_routing 1e-3)

echo "=== scaled SOFT routing + prosody ===  $(date)"
echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "routing: ${ROUTING[*]}"

# ----------------------------- Unified training -----------------------------
${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch \
    --local_data --train_split_dir train-clean-360 \
    --spear_layernorm \
    --K 16384 --topk 64 \
    "${ROUTING[@]}" --rho 0.01 \
    --aux_k 512 --aux_k_coef 0.03125 --dead_steps_threshold 256 \
    --geom_median_bias --renorm_decoder \
    --alpha 0.8 --beta 0.6 \
    --grl_weight 0.7 --grl_phoneme_weight 0.7 \
    --grl_u_weight 0.5 --grl_phoneme_u_weight 0.5 \
    --prosody --prosody_weight 0.5 \
    --grl_prosody_weight 0.5 --grl_prosody_u_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator \
    --lr_disc 1e-3 --n_disc_steps 3 \
    --stage2_steps "${STAGE2_STEPS}" --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed "${SEED}"

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: checkpoint missing" >&2; exit 3; }

# ----------------------------- Probe (incl. prosody) -----------------------------
echo; echo "----- probe -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${STAGE2_CKPT}" --stage1_ckpt "${STAGE2_CKPT}" \
    --run_name "diag_probe_${RUN_NAME}" \
    --no-hard_gumbel_routing --gumbel_tau_end 0.5 \
    --topk 64 --spear_layernorm \
    --sources "z_t,z_L,z_P,z_U" --tasks "pr,sid" --prosody --sid_probe_arch stats \
    --probe_steps "${PROBE_STEPS}" --probe_val_every 250 --probe_patience 6 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 --probe_warmup_steps 0 \
    --seed "${SEED}"

echo; echo "Finished: $(date)"
