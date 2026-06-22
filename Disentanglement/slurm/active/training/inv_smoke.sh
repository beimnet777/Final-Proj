#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=01:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=inv_smoke
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/invariance/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/invariance/%x_%j.error

# SMOKE: does the SCALE-NORMALIZED invariance loss actually DROP in a short run?
#   fast ramp (full weight by step 150) so we watch ~550 steps at full strength.
#   Watch:  inv   -> should DROP from ~1.0 (z_L becoming perturbation-invariant)
#           recon -> stays ~0.10-0.20 (content preserved; if it climbs, weight too high)
#           pr    -> stays low
#           grl_p -> stays healthy >~2 (z_P factorization not broken)
#   grl_weight=0 = passive monitor (no removal pressure); pure invariance test.

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
BLOCKS=(--fixed_blocks --per_block_topk --K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== inv SMOKE: inv_weight=4.0 fast-ramp(150) 700 steps ===  $(date)"
echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"

${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch \
    "${BLOCKS[@]}" \
    --invariance --inv_weight 4.0 --inv_ramp_end 150 \
    --local_data --train_split_dir train-clean-100 \
    --spear_layernorm --num_workers 12 \
    --alpha 0.8 --beta 0.6 --grl_weight 0.0 --grl_phoneme_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator \
    --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps 700 --warmup_steps 100 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 100 \
    --checkpoint_dir "${DIS_DIR}/checkpoints/inv_smoke" --runs_dir "${DIS_DIR}/runs/inv_smoke" \
    --log_dir "${DIS_DIR}/logs" --seed 42

echo "Finished SMOKE: $(date)"
