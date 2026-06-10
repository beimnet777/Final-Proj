#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=proj_recon_e2e
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection_reconstruct/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection_reconstruct/%x_%j.err

# End-to-end reconstructive-projection experiment in ONE allocation (no inter-stage
# queue wait):
#   1) Stage-1 SAE on SUPERB-comparable (LayerNorm'd) h_t   -> checkpoints/ln_sae
#   2) Stage-2 reconstructive projection (recon SOLELY via z_L/z_P, d=256)
#   3) Diagnostic probe of the stage-2 best checkpoint (native 41-phone labels)
# Separate dirs throughout — does not overwrite any existing run.
# 2-way by default; set U_DIM>0 (+U_L2) for the penalized-residual z_U variant.

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

mkdir -p "${DIS_DIR}/logs/train/stage1"
mkdir -p "${DIS_DIR}/logs/train/stage2/projection_reconstruct"
cd "${DIS_DIR}"

# ----- knobs (env-overridable) -----
PROJECTION_DIM="${PROJECTION_DIM:-256}"
STAGE1_STEPS="${STAGE1_STEPS:-6000}"
STAGE2_STEPS="${STAGE2_STEPS:-8000}"
SEED="${SEED:-42}"
ALPHA="${ALPHA:-0.02}"
BETA="${BETA:-0.03}"
GRL_WEIGHT="${GRL_WEIGHT:-0.01}"
GRL_P_WEIGHT="${GRL_P_WEIGHT:-0.01}"
U_DIM="${U_DIM:-0}"
U_L2="${U_L2:-0.0}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
PR_LABEL_SET="${PR_LABEL_SET:-dis}"

if [[ "${U_DIM}" -gt 0 ]]; then U_TAG="_zU${U_DIM}"; else U_TAG=""; fi
RUN_NAME="${RUN_NAME:-proj_recon_ln_d${PROJECTION_DIM}${U_TAG}}"
LN_SAE_DIR="${DIS_DIR}/checkpoints/ln_sae"
STAGE1_CKPT="${LN_SAE_DIR}/stage1_best.pt"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"
PROBE_RUN_NAME="diag_probe_${RUN_NAME}"

echo "=== Reconstructive projection — end-to-end (stage1 -> stage2 -> probe) ==="
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node              : $(hostname)"
echo "GPU               : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started           : $(date)"
echo "run_name          : ${RUN_NAME}"
echo "projection_dim    : ${PROJECTION_DIM}"
echo "z_U dim / l2      : ${U_DIM} / ${U_L2}"
echo "stage1/stage2 stp : ${STAGE1_STEPS} / ${STAGE2_STEPS}"
echo "alpha/beta/grl/gp : ${ALPHA} / ${BETA} / ${GRL_WEIGHT} / ${GRL_P_WEIGHT}"
echo "probe label set   : ${PR_LABEL_SET}"

# ============================ Stage 1: LayerNorm SAE ============================
echo; echo "----- [1/3] Stage-1 SAE on LayerNorm'd h_t -----"; date
${PYTHON} -u run.py \
    --stage              1 \
    --spear_layernorm \
    --max_train_examples 0 \
    --max_val_examples   500 \
    --total_steps        "${STAGE1_STEPS}" \
    --batch_size         16 \
    --K                  5120 \
    --topk               256 \
    --lr                 1e-4 \
    --lr_min             1e-6 \
    --warmup_steps       500 \
    --checkpoint_dir     "${LN_SAE_DIR}" \
    --runs_dir           "${DIS_DIR}/runs/ln_sae" \
    --log_dir            "${DIS_DIR}/logs" \
    --seed               "${SEED}"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
    echo "ERROR: stage 1 finished but checkpoint missing: ${STAGE1_CKPT}" >&2; exit 2
fi

# ===================== Stage 2: reconstructive projection ======================
echo; echo "----- [2/3] Stage-2 reconstructive projection -----"; date
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

if [[ ! -f "${STAGE2_CKPT}" ]]; then
    echo "ERROR: stage 2 finished but checkpoint missing: ${STAGE2_CKPT}" >&2; exit 3
fi

# ============================ Stage 3: diagnostic probe ========================
echo; echo "----- [3/3] Diagnostic probe of ${STAGE2_CKPT} -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage1_ckpt        "${STAGE1_CKPT}" \
    --stage2_ckpt        "${STAGE2_CKPT}" \
    --run_name           "${PROBE_RUN_NAME}" \
    --spear_layernorm \
    --pr_label_set       "${PR_LABEL_SET}" \
    --sources            "z_t,z_L,z_P" \
    --tasks              "pr,sid" \
    --probe_steps        "${PROBE_STEPS}" \
    --seed               "${SEED}" \
    --pr_max_examples    0 \
    --pr_probe_lr        5e-4 \
    --sid_probe_lr       1e-3 \
    --probe_warmup_steps 0

echo; echo "Finished          : $(date)"
