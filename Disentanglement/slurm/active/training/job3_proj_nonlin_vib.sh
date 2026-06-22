#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=proj_nonlin_vib
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection_reconstruct/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection_reconstruct/%x_%j.err

# JOB 3: NONLINEAR projection demixer (z_t -> z_L,z_P via MLP) + speaker GRL on z_L
# + phoneme GRL on z_P + reconstruction SOLELY through the views, with a VIB
# (KL) information bottleneck on z_L.  Reuses the pretrained LayerNorm SAE.
# The nonlinear views can synthesize a speaker-free content code; the VIB-KL
# squeezes z_L so recon must source speaker from z_P.  Probe z_L,z_P only.

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
RUN_NAME="job3_proj_nonlin_vib"
STAGE1_CKPT="${DIS_DIR}/checkpoints/ln_sae/stage1_best.pt"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"; STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"
PROJ=(--projection_disentanglement --projection_reconstruct --projection_nonlinear --projection_hidden 512 --projection_dim 256)

[[ -f "${STAGE1_CKPT}" ]] || { echo "ERROR: missing ln_sae stage1 ${STAGE1_CKPT}" >&2; exit 2; }
echo "=== JOB3 nonlinear projection + GRLs + recon-IB(KL) ===  $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
${PYTHON} -u run.py \
    --stage 2 --stage1_ckpt "${STAGE1_CKPT}" --spear_layernorm \
    "${PROJ[@]}" \
    --vib_zL_weight 1e-3 --vib_zL_ramp_end 200 --vib_zL_layernorm \
    --grl_grad_norm --grl_grad_norm_target 0.01 \
    --alpha 0.3 --beta 0.2 --grl_weight 1.0 --grl_phoneme_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps 8000 --warmup_steps 500 \
    --lr 3e-5 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed 42

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: checkpoint missing" >&2; exit 3; }
echo; echo "----- probe (z_L, z_P only) -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${STAGE2_CKPT}" --stage1_ckpt "${STAGE1_CKPT}" \
    --run_name "diag_${RUN_NAME}" --spear_layernorm \
    --projection_disentanglement --projection_reconstruct --projection_nonlinear \
    --projection_dim 256 --projection_hidden 512 --vib_zL_weight 1e-3 --vib_zL_layernorm \
    --sources "z_L,z_P" --tasks "pr,sid" --sid_probe_arch stats \
    --probe_steps 10000 --probe_val_every 250 --probe_patience 8 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 --seed 42
echo; echo "Finished $(date)"
