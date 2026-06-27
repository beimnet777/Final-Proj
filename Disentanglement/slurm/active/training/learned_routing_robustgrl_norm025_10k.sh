#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=6:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=lr_rg025_10k
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/learned_routing_inv_statsgrl/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/learned_routing_inv_statsgrl/%x_%j.err

# Focused rerun of the hard learned-routing robust-GRL experiment.
# The lower normalized GRL target tests whether the two-branch adversary can
# retain linear SID suppression without driving TopK utilisation out of z_P.

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
TRAIN_SEED=42
PROBE_SEED=42
STAGE2_STEPS=10000
PROBE_STEPS=10000
GRL_NORM_TARGET=0.00025
RUN_NAME="lr_robustgrl_gp02_hard_gn025_10k_seed${TRAIN_SEED}"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
FINAL_CKPT="${CKPT_DIR}/stage2_step${STAGE2_STEPS}.pt"

export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"

mkdir -p "${DIS_DIR}/logs/train/stage2/learned_routing_inv_statsgrl"
cd "${DIS_DIR}"

echo "=== Hard learned-routing robust-GRL, reduced norm, 10k ==="
echo "started          : $(date)"
echo "node             : ${SLURMD_NODENAME:-unknown}"
echo "gpu              : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "run_name         : ${RUN_NAME}"
echo "routing          : hard ST-Gumbel, binary L/P, no z_U"
echo "grl_head         : signed linear mean + GELU mean/std"
echo "grl_norm_target  : ${GRL_NORM_TARGET}"
echo "grl_weight       : 1.0"
echo "grl_p_weight     : 0.2"
echo "grad_clip        : 1.0"
echo "sid_head_lr      : 5e-4"
echo "stage2_steps     : ${STAGE2_STEPS}"
echo "train_seed       : ${TRAIN_SEED}"
echo "final_ckpt       : ${FINAL_CKPT}"
echo "probe            : final checkpoint, z_L->SID, linear, seed ${PROBE_SEED}"

${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch \
    --hard_gumbel_routing --n_routes 2 \
    --grl_robust_sid --grl_robust_activation gelu \
    --grl_grad_norm --grl_grad_norm_target "${GRL_NORM_TARGET}" \
    --alpha 0.8 --beta 0.6 \
    --grl_weight 1.0 --grl_phoneme_weight 0.15 \
    --grl_delay_steps 0 \
    --dann_full_discriminator --lr_disc 1e-3 --n_disc_steps 3 \
    --rho 0.0 --grad_clip 1.0 \
    --local_data --train_split_dir train-clean-100 --spear_layernorm \
    --num_workers 8 \
    --stage2_steps "${STAGE2_STEPS}" --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --lr_sid_head 5e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" \
    --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" \
    --seed "${TRAIN_SEED}"

[[ -f "${FINAL_CKPT}" ]] || {
    echo "ERROR: final checkpoint missing: ${FINAL_CKPT}" >&2
    echo "Refusing to fall back to stage2_best.pt." >&2
    exit 5
}

echo
echo "=== Final-checkpoint linear z_L SID probe ==="
echo "probe_started    : $(date)"

${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${FINAL_CKPT}" \
    --stage1_ckpt "${FINAL_CKPT}" \
    --run_name "diag_${RUN_NAME}_final_zL_sid_linear_seed${PROBE_SEED}" \
    --hard_gumbel_routing --n_routes 2 --spear_layernorm \
    --sources "z_L" --tasks "sid" --sid_probe_arch "linear" \
    --probe_steps "${PROBE_STEPS}" --probe_val_every 250 \
    --probe_patience 0 \
    --pr_max_examples 0 --sid_probe_lr 1e-3 \
    --probe_warmup_steps 0 --seed "${PROBE_SEED}"

echo
echo "finished         : $(date)"
