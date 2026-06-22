#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=16:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=inv_only_nr
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/invariance/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/invariance/%x_%j.err

# Matched invariance-only control for job1_inv_dense:
# full invariance from step 1, no encoder-side speaker GRL, same fixed blocks,
# optimizer, data, duration, and held-out stats probe.

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

cd "${DIS_DIR}"

TRAIN_SEED=42
PROBE_SEED=42
RUN_NAME="invariance_only_w4_noramp"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"
BLOCKS=(--fixed_blocks --per_block_topk
        --K_L 3072 --K_P 1024 --K_U 1024
        --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== Matched invariance-only control ==="
echo "run_name   : ${RUN_NAME}"
echo "train_seed : ${TRAIN_SEED}"
echo "probe_seed : ${PROBE_SEED}"
echo "started    : $(date)"
echo "gpu        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"

${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch "${BLOCKS[@]}" \
    --invariance --inv_weight 4.0 --inv_ramp_end 0 \
    --local_data --train_split_dir train-clean-100 --spear_layernorm \
    --num_workers 12 \
    --alpha 0.8 --beta 0.6 --grl_weight 0.0 --grl_phoneme_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator --lr_disc 1e-3 \
    --n_disc_steps 3 --rho 0.0 \
    --stage2_steps 12000 --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" \
    --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" \
    --seed "${TRAIN_SEED}"

[[ -f "${STAGE2_CKPT}" ]] || {
    echo "ERROR: training finished but ${STAGE2_CKPT} is missing" >&2
    exit 3
}

echo
echo "----- held-out diagnostic probe: z_L and z_P -----"
date
${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${STAGE2_CKPT}" \
    --stage1_ckpt "${STAGE2_CKPT}" \
    --run_name "diag_${RUN_NAME}" \
    "${BLOCKS[@]}" --spear_layernorm \
    --sources "z_L,z_P" --tasks "pr,sid" --sid_probe_arch stats \
    --probe_steps 10000 --probe_val_every 250 --probe_patience 8 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 \
    --probe_warmup_steps 0 --seed "${PROBE_SEED}"

echo
echo "Finished: $(date)"
