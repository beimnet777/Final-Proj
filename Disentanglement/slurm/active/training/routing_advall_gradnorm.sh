#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=routing_advall_gn
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/routing_advall/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/routing_advall/%x_%j.err

# FLOATING ROUTING (no fixed blocks) + adversaries on ALL THREE buckets, with the
# speaker adversaries GRAD-NORMALIZED.  Lower capacity: topk 128 (was 256), and z_L
# size is emergent (not pinned at 160).
#   grl (pooled) on z_L   |  grl_p on z_P   |  grl_u + pr_grl_u on z_U (force residual)
#   speaker adversaries (grl, grl_u) use per-frame grad-norm (target 0.001).
#   rho/spec keep the routing buckets from collapsing.  NO probe.
#   WATCH: actL/P/U (bucket balance), grl/grl_p/grlU, recon, pr.

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null; module load rhel8/default-amp 2>/dev/null
PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"
mkdir -p "${DIS_DIR}/logs/train/stage2/routing_advall"
cd "${DIS_DIR}"
RUN_NAME="routing_advall_gn"
STAGE1_CKPT="${DIS_DIR}/checkpoints/ln_sae/stage1_best.pt"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"

[[ -f "${STAGE1_CKPT}" ]] || { echo "ERROR: missing ln_sae stage1 ${STAGE1_CKPT}" >&2; exit 2; }
echo "=== FLOATING ROUTING + adv(z_L,z_P,z_U) grad-norm, topk 128 ===  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
${PYTHON} -u run.py \
    --stage 2 --stage1_ckpt "${STAGE1_CKPT}" --spear_layernorm \
    --topk 128 \
    --n_routes 3 --hard_gumbel_routing --gumbel_tau_start 1.0 --gumbel_tau_end 0.5 \
    --routing_init_std 0.5 --routing_spec_weight 0.01 --lr_routing 1e-3 --rho 0.01 \
    --alpha 0.3 --beta 0.2 \
    --grl_weight 0.5 --grl_phoneme_weight 0.5 \
    --grl_u_weight 0.5 --grl_phoneme_u_weight 0.5 \
    --grl_grad_norm --grl_grad_norm_target 0.001 \
    --grl_delay_steps 0 --dann_full_discriminator --lr_disc 1e-3 --n_disc_steps 3 \
    --stage2_steps 12000 --warmup_steps 500 \
    --lr 5e-5 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed 42
echo "Finished (NO probe): $(date)"
