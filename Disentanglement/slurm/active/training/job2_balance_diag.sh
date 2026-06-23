#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=job2_balance_diag
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/balance_diag/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/balance_diag/%x_%j.err

# BALANCING DIAGNOSTIC (Step 1 of the multi-task-balancing plan).
#
# WHAT THIS RUN IS FOR
#   Harvest dense per-loss gradient norms AND pairwise gradient cosines on the
#   shared SAE trunk under the *current best* Job-2 config, so we can choose
#   the right Step-2 mechanism (PCGrad / per-task gradient clip / two-timescale
#   discriminator) from real numbers rather than guesswork.  The cosines were
#   wired into train.py:_log_grad_norms_stage2 on 2026-06-23 and exposed as
#   `grad_cos/*` TB scalars + the "Gradient Conflict" panel.
#
# WHAT THIS RUN IS NOT
#   Not a Job-2 replication.  No final probe (we are not evaluating leakage —
#   probe-seed sweep + MDL handle that).  No new checkpoint to compare against
#   the canonical Job-2.  Checkpoints go to a throwaway dir.
#
# DESIGN CHOICES
#   * 2500 steps  = warmup (500) + 2000 post-warmup steps.  Enough to cross
#     the GRL ramp and see steady-state conflict, not a full training run.
#   * grad_log_every=50  = 50× denser than the production config; cosines
#     are cheap (one extra dot-product per loss pair per snapshot).  Yields
#     ~50 cosine snapshots — enough to draw a clean trace.
#   * Same alpha/beta/grl_weight/grl_phoneme_weight/disc as Job 2 so the
#     measured conflict reflects exactly the dynamics we are trying to
#     improve.  Changing any of these would conflate "what's happening
#     now" with "what we'd see under different weights".
#   * Wall ~3h on A100 (budget 4h).  Single GPU.  Single seed (42) so
#     traces are comparable to the canonical Job-2 logs.
#
# AFTER THE RUN
#   grep '\[grad_cos' <log>      # human-readable cosine trace
#   tensorboard --logdir runs/job2_balance_diag/    # "Gradient Conflict" panel
#   Read these and decide Step 2 — see Experiment Tree §5.3 / §6.3.

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
mkdir -p "${DIS_DIR}/logs/train/stage2/balance_diag"
cd "${DIS_DIR}"

RUN_NAME="job2_balance_diag"
# throwaway ckpt dir — explicitly NOT the canonical job2_dense_gradnorm path
CKPT_DIR="${DIS_DIR}/checkpoints/diag_${RUN_NAME}"
BLOCKS=(--fixed_blocks --per_block_topk --K_L 3072 --K_P 1024 --K_U 1024 \
        --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== JOB2 BALANCING DIAGNOSTIC (grad cosines) ===  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
echo "Steps=2500  grad_log_every=50  →  expect ~50 grad-cosine snapshots"
echo "Cosine output: grep '\[grad_cos' on the .out file"

${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch "${BLOCKS[@]}" \
    --grl_dense_context --grl_context_kernel 31 \
    --grl_grad_norm --grl_grad_norm_target 0.001 \
    --local_data --train_split_dir train-clean-100 --spear_layernorm \
    --alpha 0.8 --beta 0.6 --grl_weight 1.0 --grl_phoneme_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps 2500 --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 50 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed 42

echo
echo "=== diagnostic finished $(date) ==="
echo "Next: read grad cosines and pick Step 2 mechanism."
echo "  conflict (cos<-0.05) between recon ↔ adv  →  PCGrad"
echo "  norms differ >5x between any two losses    →  per-task gradient clip"
echo "  adv cos with disc updates is unstable       →  two-timescale (slow disc_lr)"
