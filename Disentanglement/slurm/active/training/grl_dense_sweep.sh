#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=grl_dense
#SBATCH --array=0-3%4
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/grl_dense/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/grl_dense/%x_%A_%a.error

# Make the SPEAKER adversary update z_L DENSELY (like grl_p updates z_P), instead
# of one diluted pooled gradient.  Healthy K=5120 config (= grl_attn baseline),
# TRAIN ONLY (no probe — watch the trends first).
#   0  dense_context k=31, grl_weight 0.5   (baseline dense)
#   1  dense_context k=31, grl_weight 2.0   (dense + higher weight = the LR/magnitude idea)
#   2  dense_context k=61, grl_weight 0.5   (more temporal context)
#   3  attention-pool,     grl_weight 2.0   (CONTROL: magnitude alone on the POOLED adversary)
#
# Watch: does grl (speaker-adv loss) rise toward chance (ln 251 = 5.52)?  Rising =
# removal is happening; flat-low (~0.78 like grl_attn) = still not removing.
# Also watch recon (should stay ~0.10) and actL/P/U.

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

i="${SLURM_ARRAY_TASK_ID:-0}"
case "$i" in
  0) NAME="dense_k31_w0.5"; ADV=(--grl_dense_context --grl_context_kernel 31); GRLW=0.5 ;;
  1) NAME="dense_k31_w2.0"; ADV=(--grl_dense_context --grl_context_kernel 31); GRLW=2.0 ;;
  2) NAME="dense_k61_w0.5"; ADV=(--grl_dense_context --grl_context_kernel 61); GRLW=0.5 ;;
  3) NAME="attn_w2.0";      ADV=(--grl_attention_pool);                        GRLW=2.0 ;;
  *) echo "bad index $i" >&2; exit 1 ;;
esac

SEED="${SEED:-42}"
RUN_NAME="grl_${NAME}"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
BLOCKS=(--fixed_blocks --per_block_topk --K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== ${NAME}: speaker adv ${ADV[*]}  grl_weight=${GRLW} ===  $(date)"
echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"

${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch \
    "${BLOCKS[@]}" \
    "${ADV[@]}" \
    --local_data --train_split_dir train-clean-100 \
    --spear_layernorm \
    --alpha 0.8 --beta 0.6 --grl_weight "${GRLW}" --grl_phoneme_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator \
    --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps 12000 --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed "${SEED}"

echo "Finished ${i} (TRAIN ONLY — no probe): $(date)"
