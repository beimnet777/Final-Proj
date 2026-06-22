#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=dense_gradnorm_advU
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/dense_gradnorm_advU/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/dense_gradnorm_advU/%x_%j.err

# A/B vs job2_dense_gradnorm — identical base, three changes:
#   (1) adversaries ALSO on z_U: grl_u (speaker, auto grad-normed) + pr_grl_u (phoneme)
#       -> no bucket is a free sink anymore.  Tests whether the fixed-block setup
#          (which can't flee like routing_advall) can clean z_U without breaking
#          recon / re-dirtying z_L (the additive all-pure test).
#   (2) z_P phoneme adversary cranked 0.5 -> 1.0 (push z_P content-clean).
#   (3) dead-latent LOGGING only: aux_k on but aux_k_coef=0 -> dead=% is logged with
#       ZERO training effect, no revival (geom-median/renorm stay OFF).  A/B-clean.
# NO probe in this run; per-bucket val accuracy (z_L/z_P/z_U x PR/SID) is logged in
# the [val] lines instead.  Watch: z_L SID (holds ~chance?), z_U PR/SID (both fall?),
# z_P PR (rises?), and recon + dead% (the cost of forbidding the sink).

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null; module load rhel8/default-amp 2>/dev/null
PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"
mkdir -p "${DIS_DIR}/logs/train/stage2/dense_gradnorm_advU"
cd "${DIS_DIR}"
RUN_NAME="dense_gradnorm_advU"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
BLOCKS=(--fixed_blocks --per_block_topk --K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== ${RUN_NAME}: dense GRL grad-norm + z_U adversaries + cranked grl_p ===  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader
${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch "${BLOCKS[@]}" \
    --grl_dense_context --grl_context_kernel 31 \
    --grl_grad_norm --grl_grad_norm_target 0.001 \
    --local_data --train_split_dir train-clean-100 --spear_layernorm \
    --alpha 0.8 --beta 0.6 --grl_weight 1.0 --grl_phoneme_weight 1.0 \
    --grl_u_weight 1.0 --grl_phoneme_u_weight 0.5 \
    --aux_k 512 --aux_k_coef 0 --dead_steps_threshold 256 \
    --grl_delay_steps 0 --dann_full_discriminator --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps 12000 --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed 42
echo; echo "Finished (no probe) $(date)"
