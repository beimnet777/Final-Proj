#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=job3_smoke
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection_reconstruct/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection_reconstruct/%x_%j.err

# JOB 3 CALIBRATION SMOKE (700 steps, NO probe): read the actual loss magnitudes
# so vib_zL_weight / alpha / beta are set from data, not guessed.
#   Watch:  recon, pr, sid, grl, grl_p, AND vib=KL(w=...)  <- the unknown we need.
#   GRL uses grad-norm so grl_weight needs no calibration (per-frame push = 1.0*0.01).
#   VIB ramps fast (200) so KL bites within the smoke.

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null; module load rhel8/default-amp 2>/dev/null
PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"
mkdir -p "${DIS_DIR}/logs/train/stage2/projection_reconstruct"
cd "${DIS_DIR}"
STAGE1_CKPT="${DIS_DIR}/checkpoints/ln_sae/stage1_best.pt"
PROJ=(--projection_disentanglement --projection_reconstruct --projection_nonlinear --projection_hidden 512 --projection_dim 256)

[[ -f "${STAGE1_CKPT}" ]] || { echo "ERROR: missing ln_sae stage1 ${STAGE1_CKPT}" >&2; exit 2; }
echo "=== JOB3 SMOKE: read loss scales (nonlin proj + VIB + grad-norm GRL) ===  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
${PYTHON} -u run.py \
    --stage 2 --stage1_ckpt "${STAGE1_CKPT}" --spear_layernorm \
    "${PROJ[@]}" \
    --vib_zL_weight 1e-3 --vib_zL_ramp_end 200 \
    --grl_grad_norm --grl_grad_norm_target 0.01 \
    --alpha 0.3 --beta 0.2 --grl_weight 1.0 --grl_phoneme_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps 500 --warmup_steps 100 \
    --lr 3e-5 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 100 \
    --checkpoint_dir "${DIS_DIR}/checkpoints/job3_smoke" --runs_dir "${DIS_DIR}/runs/job3_smoke" \
    --log_dir "${DIS_DIR}/logs" --seed 42
echo "Finished SMOKE: $(date)"
