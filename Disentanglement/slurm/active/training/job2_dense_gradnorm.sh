#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=dense_gradnorm
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/grl_dense/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/grl_dense/%x_%j.err

# JOB 2: DENSE per-frame speaker GRL with PER-FRAME GRADIENT NORMALIZATION.
# Each valid frame receives an equal removal push of L2 norm = grl_weight*target
# (here 1.0*0.001 = 0.001/frame, ~3x the phoneme adversary's 0.00034 that works),
# regardless of the discriminator's confidence — directly fixing the per-frame
# dilution we measured.  grl_p keeps z_P clean.  Probe z_L,z_P only.

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null; module load rhel8/default-amp 2>/dev/null
PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"
mkdir -p "${DIS_DIR}/logs/train/stage2/grl_dense"
cd "${DIS_DIR}"
RUN_NAME="job2_dense_gradnorm"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"; STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"
BLOCKS=(--fixed_blocks --per_block_topk --K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== JOB2 dense GRL + grad-norm (target 0.001) ===  $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader
${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch "${BLOCKS[@]}" \
    --grl_dense_context --grl_context_kernel 31 \
    --grl_grad_norm --grl_grad_norm_target 0.001 \
    --local_data --train_split_dir train-clean-100 --spear_layernorm \
    --alpha 0.8 --beta 0.6 --grl_weight 1.0 --grl_phoneme_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps 12000 --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed 42

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: checkpoint missing" >&2; exit 3; }
echo; echo "----- probe (z_L, z_P only) -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${STAGE2_CKPT}" --stage1_ckpt "${STAGE2_CKPT}" \
    --run_name "diag_${RUN_NAME}" "${BLOCKS[@]}" --spear_layernorm \
    --sources "z_L,z_P" --tasks "pr,sid" --sid_probe_arch stats \
    --probe_steps 10000 --probe_val_every 250 --probe_patience 8 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 --seed 42
echo; echo "Finished $(date)"
