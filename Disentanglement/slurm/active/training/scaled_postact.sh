#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=scaled_postact
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/scaled_postact/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/scaled_postact/%x_%j.err

# Scaled run, FIXED POST-ACTIVATION partition (--fixed_blocks --per_block_topk):
# membership AND per-block active budget are fixed, so z_P is GUARANTEED active
# units and cannot lose the per-frame top-k selection race.  This is the direct
# fix for the z_P starvation seen in the emergent (global top-k) scaled runs,
# where pooled SID gave no per-frame select signal and z_P collapsed to ~3 units.
#
# Block sizes 9830/3277/3277 (3:1:1, exp1 ratio scaled to K=16384); active budget
# 40/16/8 (5:2:1, exp1 ratio scaled to topk=64) → z_P now has 16 active units.
# Mean-pool GRL (attentive pooling is a separate test) so this isolates the budget.

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
mkdir -p "${DIS_DIR}/logs/train/stage2/scaled_postact"
cd "${DIS_DIR}"

SEED="${SEED:-42}"
STAGE2_STEPS="${STAGE2_STEPS:-26000}"   # 4 passes/utt on 360h
PROBE_STEPS="${PROBE_STEPS:-8000}"
RUN_NAME="scaled_postact"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"
BLOCKS=(--fixed_blocks --per_block_topk --K_L 9830 --K_P 3277 --K_U 3277 --topk_L 40 --topk_P 16 --topk_U 8)

echo "=== scaled FIXED post-activation (per-block budget) ===  $(date)"
echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "blocks: ${BLOCKS[*]}"

# ----------------------------- Unified training -----------------------------
${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch \
    --local_data --train_split_dir train-clean-360 \
    --spear_layernorm \
    --K 16384 --topk 64 \
    "${BLOCKS[@]}" --rho 0.0 \
    --aux_k 512 --aux_k_coef 0.03125 --dead_steps_threshold 256 \
    --geom_median_bias --renorm_decoder \
    --alpha 0.8 --beta 0.6 \
    --grl_weight 0.7 --grl_phoneme_weight 0.7 \
    --grl_u_weight 0.5 --grl_phoneme_u_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator \
    --lr_disc 1e-3 --n_disc_steps 3 \
    --stage2_steps "${STAGE2_STEPS}" --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed "${SEED}"

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: checkpoint missing" >&2; exit 3; }

# ----------------------------- Probe (robust, z_U included, high ceiling) -----------------------------
echo; echo "----- probe -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${STAGE2_CKPT}" --stage1_ckpt "${STAGE2_CKPT}" \
    --run_name "diag_probe_${RUN_NAME}" \
    "${BLOCKS[@]}" \
    --topk 64 --spear_layernorm \
    --sources "z_t,z_L,z_P,z_U" --tasks "pr,sid" --sid_probe_arch stats \
    --probe_steps "${PROBE_STEPS}" --probe_val_every 250 --probe_patience 6 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 --probe_warmup_steps 0 \
    --seed "${SEED}"

echo; echo "Finished: $(date)"
