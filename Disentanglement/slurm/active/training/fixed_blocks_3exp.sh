#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=fxblk_3exp
#SBATCH --array=0-2%3
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/fixed_blocks_3exp/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/fixed_blocks_3exp/%x_%A_%a.err

# Three fixed-block supervised-SAE experiments, all from scratch, all with the
# FIXED GRL (awakened speaker discriminator) and the cranked phoneme adversary.
#
# Shared "fixed-GRL" base:
#   - calibrated builds: alpha 0.8 (PR->L), beta 0.6 (SID->P)   [beta keeps P alive]
#   - speaker adversary grl=0.5, phoneme adversary grl_p=0.5 (cranked from 0.1 to
#     push z_P->PR to chance — safe, recon sources phonemes from z_L)
#   - grl_delay=0 (smooth ramp, no jump),  dann_full_discriminator
#   - lr_disc=1e-3 + n_disc_steps=3  -> discriminators track the moving encoder
#   - SAE from scratch at full lr 1e-4
#
#   0  exp1_perblock      fixed membership + per-block TopK 160/64/32 (post-activation)
#   1  exp2_perblock_IN   exp1 + instance-norm z_L (structural speaker removal)
#   2  exp3_global_equal  equal membership + global TopK (emergent allocation)

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
mkdir -p "${DIS_DIR}/logs/train/stage2/fixed_blocks_3exp"
cd "${DIS_DIR}"

i="${SLURM_ARRAY_TASK_ID:-0}"
# ---- per-experiment config ----
# TOPK_ARGS: train SAE TopK mode.  IN_ARGS: instance norm.  BLOCK_ARGS: membership.
# PROBE_TOPK / PROBE_IN: must mirror training so the probe rebuilds the same model.
case "$i" in
  0) NAME="exp1_perblock"
     BLOCK_ARGS=(--K_L 3072 --K_P 1024 --K_U 1024)
     TOPK_ARGS=(--per_block_topk --topk_L 160 --topk_P 64 --topk_U 32)
     IN_ARGS=()
     PROBE_TOPK=(--per_block_topk --topk_L 160 --topk_P 64 --topk_U 32)
     PROBE_IN=() ;;
  1) NAME="exp2_perblock_IN"
     BLOCK_ARGS=(--K_L 3072 --K_P 1024 --K_U 1024)
     TOPK_ARGS=(--per_block_topk --topk_L 160 --topk_P 64 --topk_U 32)
     IN_ARGS=(--instance_norm_zL)
     PROBE_TOPK=(--per_block_topk --topk_L 160 --topk_P 64 --topk_U 32)
     PROBE_IN=(--instance_norm_zL) ;;
  2) NAME="exp3_global_equal"
     BLOCK_ARGS=(--K_L 1706 --K_P 1707 --K_U 1707)
     TOPK_ARGS=(--no-per_block_topk)
     IN_ARGS=()
     PROBE_TOPK=(--no-per_block_topk)
     PROBE_IN=() ;;
  *) echo "bad array index $i" >&2; exit 1 ;;
esac

SEED="${SEED:-42}"
STAGE2_STEPS="${STAGE2_STEPS:-12000}"
PROBE_STEPS="${PROBE_STEPS:-4000}"
RUN_NAME="fxblk_${NAME}"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"

echo "=== fixed-block experiment ${i}: ${NAME} ==="
echo "Job/Array  : ${SLURM_JOB_ID} / ${i}"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"
echo "blocks     : ${BLOCK_ARGS[*]}"
echo "topk mode  : ${TOPK_ARGS[*]}"
echo "instance N : [${IN_ARGS[*]:-none}]"
echo "GRL        : grl=0.5 grl_p=0.5 delay=0 dann  lr_disc=1e-3 n_disc_steps=3"

# ----------------------------- Train (supervised SAE) -----------------------------
${PYTHON} -u run.py \
    --stage              2 \
    --stage2_from_scratch \
    --fixed_blocks \
    "${BLOCK_ARGS[@]}" \
    "${TOPK_ARGS[@]}" \
    "${IN_ARGS[@]}" \
    --spear_layernorm \
    --alpha              0.8 \
    --beta               0.6 \
    --grl_weight         0.5 \
    --grl_phoneme_weight 0.5 \
    --grl_delay_steps    0 \
    --dann_full_discriminator \
    --lr_disc            1e-3 \
    --n_disc_steps       3 \
    --rho                0.0 \
    --stage2_steps       "${STAGE2_STEPS}" \
    --warmup_steps       500 \
    --lr                 1e-4 \
    --lr_min             1e-6 \
    --lr_heads           1e-4 \
    --grad_log_every     500 \
    --checkpoint_dir     "${CKPT_DIR}" \
    --runs_dir           "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir            "${DIS_DIR}/logs" \
    --seed               "${SEED}"

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: training finished but checkpoint missing" >&2; exit 3; }

# ----------------------------- Probe (unified) -----------------------------
# TopK mode + instance norm + block sizes MUST match training.
echo; echo "----- probe -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt        "${STAGE2_CKPT}" \
    --stage1_ckpt        "${STAGE2_CKPT}" \
    --run_name           "diag_probe_${RUN_NAME}" \
    --fixed_blocks \
    "${BLOCK_ARGS[@]}" \
    "${PROBE_TOPK[@]}" \
    "${PROBE_IN[@]}" \
    --spear_layernorm \
    --sources            "z_t,z_L,z_P" \
    --tasks              "pr,sid" \
    --probe_steps        "${PROBE_STEPS}" \
    --probe_val_every    "${PROBE_VAL_EVERY:-250}" \
    --probe_patience     "${PROBE_PATIENCE:-5}" \
    --seed               "${SEED}" \
    --pr_max_examples    0 \
    --pr_probe_lr        5e-4 \
    --sid_probe_lr       1e-3 \
    --probe_warmup_steps 0

echo; echo "Finished   : $(date)"
