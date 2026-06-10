#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=proj_recon
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection_reconstruct/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection_reconstruct/%x_%j.err

# Reconstructive projection (separate experiment family — does not overwrite).
#   * h_t reconstructed SOLELY through z_L/z_P: z_hat = up_L(z_L)+up_P(z_P), decode(z_hat)
#   * d = 256 (= topk);  SUPERB-comparable LayerNorm'd encoder
#   * default 2-way (no z_U).  Set U_DIM>0 (+ U_L2) to add a penalized residual z_U.
# Requires the LayerNorm stage-1 SAE (stage1_ln_sae.sh) — same h_t in both stages.

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

mkdir -p "${DIS_DIR}/logs/train/stage2/projection_reconstruct"
cd "${DIS_DIR}"

# z_U is OFF by default (2-way).  Override: sbatch --export=ALL,U_DIM=64,U_L2=0.01 ...
U_DIM="${U_DIM:-0}"
U_L2="${U_L2:-0.0}"
PROJECTION_DIM="${PROJECTION_DIM:-256}"
STAGE2_STEPS="${STAGE2_STEPS:-8000}"
SEED="${SEED:-42}"
ALPHA="${ALPHA:-0.02}"
BETA="${BETA:-0.03}"
GRL_WEIGHT="${GRL_WEIGHT:-0.01}"
GRL_P_WEIGHT="${GRL_P_WEIGHT:-0.01}"

if [[ "${U_DIM}" -gt 0 ]]; then U_TAG="_zU${U_DIM}"; else U_TAG=""; fi
RUN_NAME="${RUN_NAME:-proj_recon_ln_d${PROJECTION_DIM}${U_TAG}}"
STAGE1_CKPT="${STAGE1_CKPT:-${DIS_DIR}/checkpoints/ln_sae/stage1_best.pt}"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"

echo "=== Stage 2 — reconstructive projection ==="
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node              : $(hostname)"
echo "GPU               : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started           : $(date)"
echo "run_name          : ${RUN_NAME}"
echo "stage1_ckpt       : ${STAGE1_CKPT}"
echo "projection_dim    : ${PROJECTION_DIM}"
echo "z_U dim / l2      : ${U_DIM} / ${U_L2}"
echo "alpha/beta/grl    : ${ALPHA} / ${BETA} / ${GRL_WEIGHT}"
echo "grl_p_weight      : ${GRL_P_WEIGHT}"
echo "stage2_steps      : ${STAGE2_STEPS}"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
    echo "ERROR: missing LayerNorm stage-1 checkpoint: ${STAGE1_CKPT}" >&2
    echo "       run stage1_ln_sae.sh first." >&2
    exit 2
fi

${PYTHON} -u run.py \
    --stage                  2 \
    --stage1_ckpt            "${STAGE1_CKPT}" \
    --spear_layernorm \
    --projection_disentanglement \
    --projection_reconstruct \
    --projection_dim         "${PROJECTION_DIM}" \
    --projection_u_dim       "${U_DIM}" \
    --projection_u_l2        "${U_L2}" \
    --stage2_steps           "${STAGE2_STEPS}" \
    --warmup_steps           500 \
    --alpha                  "${ALPHA}" \
    --beta                   "${BETA}" \
    --grl_weight             "${GRL_WEIGHT}" \
    --grl_delay_steps        0 \
    --grl_phoneme_weight     "${GRL_P_WEIGHT}" \
    --rho                    0 \
    --lr                     3e-5 \
    --lr_min                 1e-6 \
    --lr_heads               1e-4 \
    --grad_log_every         500 \
    --checkpoint_dir         "${CKPT_DIR}" \
    --runs_dir               "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir                "${DIS_DIR}/logs" \
    --seed                   "${SEED}"

echo "Finished          : $(date)"
