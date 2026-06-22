#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=grl_frame
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/grl_framelevel/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/grl_framelevel/%x_%j.err

# Settle the frame-vs-pooled question: exp1's config EXACTLY, but the SPEAKER
# adversary is now PER-FRAME (--grl_frame_level), just like the phoneme adversary
# grl_p.  Direct A/B vs exp1 (pooled GRL, z_L->SID 0.822): does a dense per-frame
# speaker gradient remove more speaker from z_L, or hit the same intrinsic ceiling?
# Training reads LOCAL flac (no CDN).  z_L->SID probe gets a HIGH step ceiling
# (it undertrains at 4000) and the robust STATS probe.

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
mkdir -p "${DIS_DIR}/logs/train/stage2/grl_framelevel"
cd "${DIS_DIR}"

SEED="${SEED:-42}"
STAGE2_STEPS="${STAGE2_STEPS:-12000}"   # match exp1 for a clean A/B
PROBE_STEPS="${PROBE_STEPS:-10000}"     # z_L->SID undertrains at 4000 — give it room
RUN_NAME="fxblk_grlframe"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"
BLOCKS=(--fixed_blocks --per_block_topk --K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== frame-level speaker GRL (exp1 + --grl_frame_level) ===  $(date)"
echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# ----------------------------- Train -----------------------------
${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch \
    "${BLOCKS[@]}" \
    --grl_frame_level \
    --local_data --train_split_dir train-clean-100 \
    --spear_layernorm \
    --alpha 0.8 --beta 0.6 --grl_weight 0.5 --grl_phoneme_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator \
    --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps "${STAGE2_STEPS}" --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed "${SEED}"

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: checkpoint missing" >&2; exit 3; }

# ----------------------------- Probe (robust z_L->SID, high ceiling) -----------------------------
echo; echo "----- probe -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${STAGE2_CKPT}" --stage1_ckpt "${STAGE2_CKPT}" \
    --run_name "diag_probe_${RUN_NAME}" \
    "${BLOCKS[@]}" \
    --spear_layernorm \
    --sources "z_t,z_L,z_P" --tasks "pr,sid" --sid_probe_arch stats \
    --probe_steps "${PROBE_STEPS}" --probe_val_every 250 --probe_patience 8 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 --probe_warmup_steps 0 \
    --seed "${SEED}"

echo; echo "Finished $(date)"
