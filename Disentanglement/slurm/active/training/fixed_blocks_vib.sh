#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=fxblk_vib
#SBATCH --array=0-2%3
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/fixed_blocks_vib/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/fixed_blocks_vib/%x_%A_%a.err

# IMPROVEMENT: VIB (information bottleneck) on z_L, NO instance-norm.
# Forces z_L to keep only what PR needs (KL compresses mu→0 for everything else),
# shedding the *separable* speaker by compression rather than adversarially.
# Base = exp1 (per-block fixed blocks) + the fixed GRL (awakened discriminators).
# Sweeps the one critical unknown — the KL weight beta — over 1e-4 / 1e-3 / 1e-2,
# ramped in over the first half so z_L forms before it is compressed.
#   0  vib_beta=1e-4   1  vib_beta=1e-3   2  vib_beta=1e-2

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
mkdir -p "${DIS_DIR}/logs/train/stage2/fixed_blocks_vib"
cd "${DIS_DIR}"

i="${SLURM_ARRAY_TASK_ID:-0}"
case "$i" in
  0) VIB=1e-4 ;;
  1) VIB=1e-3 ;;
  2) VIB=1e-2 ;;
  *) echo "bad index $i" >&2; exit 1 ;;
esac
NAME="vib_b${VIB}"

SEED="${SEED:-42}"
STAGE2_STEPS="${STAGE2_STEPS:-12000}"
PROBE_STEPS="${PROBE_STEPS:-4000}"
RUN_NAME="fxblk_${NAME}"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"

echo "=== VIB experiment ${i}: beta=${VIB} (no IN) ===  $(date)"
echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"

${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch --fixed_blocks --per_block_topk \
    --K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32 \
    --vib_zL_weight "${VIB}" --vib_zL_ramp_end 6000 \
    --spear_layernorm \
    --alpha 0.8 --beta 0.6 --grl_weight 0.5 --grl_phoneme_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator \
    --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps "${STAGE2_STEPS}" --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed "${SEED}"

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: checkpoint missing" >&2; exit 3; }

echo; echo "----- probe -----"; date
# Probe z_L speaker with BOTH the robust (stats) and linear probes so we see the
# honest number; include z_U to watch the residual.
${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${STAGE2_CKPT}" --stage1_ckpt "${STAGE2_CKPT}" \
    --run_name "diag_probe_${RUN_NAME}" \
    --fixed_blocks --per_block_topk \
    --K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32 \
    --vib_zL_weight "${VIB}" \
    --spear_layernorm \
    --sources "z_t,z_L,z_P,z_U" --tasks "pr,sid" --sid_probe_arch stats \
    --probe_steps "${PROBE_STEPS}" --probe_val_every 250 --probe_patience 5 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 --probe_warmup_steps 0

echo; echo "Finished ${i}: $(date)"
