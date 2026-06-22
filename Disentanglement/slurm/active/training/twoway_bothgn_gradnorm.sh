#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=twoway_bothgn_gradnorm
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/twoway_bothgn/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/twoway_bothgn/%x_%j.err

# EXP 2 — identical to twoway_bothgn_manual.sh EXCEPT the cooperative build weights
# are balanced automatically by GradNorm (Chen et al. 2018) on recon, pr, sid, aux.
# Adversaries keep their own per-frame grad-norm + manual lambda (NOT GradNorm-managed).
# Watch the live weights in the step line: gn=[recon:.. pr:.. sid:.. aux:..].

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null; module load rhel8/default-amp 2>/dev/null
PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"
mkdir -p "${DIS_DIR}/logs/train/stage2/twoway_bothgn"
cd "${DIS_DIR}"
RUN_NAME="twoway_bothgn_gradnorm"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
BLOCKS=(--fixed_blocks --per_block_topk --K_L 3072 --K_P 2048 --K_U 0 --topk_L 160 --topk_P 96 --topk_U 0)

echo "=== ${RUN_NAME}: 2-way, both adversaries grad-normed, AuxK on, GradNorm(recon,pr,sid,aux) ===  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch "${BLOCKS[@]}" \
    --grl_dense_context --grl_context_kernel 31 \
    --grl_grad_norm --grl_grad_norm_target 0.001 \
    --grl_p_grad_norm --grl_p_grad_norm_target 0.001 \
    --aux_k 512 --aux_k_coef 0.03125 --dead_steps_threshold 256 --geom_median_bias --renorm_decoder \
    --gradnorm --gradnorm_tasks "recon,pr,sid,aux" --gradnorm_alpha 1.5 --gradnorm_lr 0.025 \
    --local_data --train_split_dir train-clean-100 --spear_layernorm \
    --alpha 0.8 --beta 0.6 --grl_weight 1.0 --grl_phoneme_weight 1.0 \
    --grl_delay_steps 0 --dann_full_discriminator --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps 12000 --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed 42
echo; echo "Finished (no probe) $(date)"
