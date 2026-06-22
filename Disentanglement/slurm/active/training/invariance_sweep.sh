#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=08:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=invariance
#SBATCH --array=0-1%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/invariance/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/invariance/%x_%A_%a.error

# z_L speaker-INVARIANCE (pitch+formant perturbed-pair consistency) on the healthy
# K=5120 config.  Speaker adversary kept at grl_weight=0 = a PASSIVE MONITOR (reads
# z_L speaker but no removal pressure), so we can watch whether invariance removes
# speaker WITHOUT a probe.  TRAIN ONLY.  Sweep inv_weight to find the balance.
#   Watch:  inv  (should DROP = z_L becoming invariant)
#           grl  (should RISE toward chance ln251=5.52 = speaker no longer readable)
#           recon ~0.10 and pr low  (content preserved; if they break, inv_weight too high)

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null; module load rhel8/default-amp 2>/dev/null
PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"
mkdir -p "${DIS_DIR}/logs/train/stage2/invariance"
cd "${DIS_DIR}"

i="${SLURM_ARRAY_TASK_ID:-0}"
WEIGHTS=(1.0 4.0)
W="${WEIGHTS[$i]}"
RUN_NAME="inv_w${W}"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
BLOCKS=(--fixed_blocks --per_block_topk --K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== invariance inv_weight=${W} (grl=0 monitor) ===  $(date)"
echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"

${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch \
    "${BLOCKS[@]}" \
    --invariance --inv_weight "${W}" --inv_ramp_end 3000 \
    --local_data --train_split_dir train-clean-100 \
    --spear_layernorm --num_workers 12 \
    --alpha 0.8 --beta 0.6 --grl_weight 0.0 --grl_phoneme_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator \
    --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps 12000 --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed 42

echo "Finished ${i} (TRAIN ONLY): $(date)"
