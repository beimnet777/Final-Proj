#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=routing_4exp
#SBATCH --array=0-3%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/routing_4exp/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/routing_4exp/%x_%A_%a.err

# Four routing experiments (reuse ln_sae, stage-2 + unified probe), anchored on
# Exp 2 (hard). 1 vs 2 = soft vs hard; 2 vs 3 = normal vs DANN adversary;
# 2 vs 4 = IB off vs on. GRL adversary on in all (grl=0.5, grl_p=0.1).
#   0  soft            soft masks (train+test), tau->0.5, normal adv
#   1  hard            hard STE, tau->0.1, normal adv
#   2  hard + DANN     hard, full-discriminator adversary
#   3  hard + IB       hard, ramped L/P capacity penalty, rho=0

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
mkdir -p "${DIS_DIR}/logs/train/stage2/routing_4exp"
cd "${DIS_DIR}"

i="${SLURM_ARRAY_TASK_ID:-0}"
# ---- per-experiment config ----
MASK_ARGS=(); ADV_ARGS=(); IB_ARGS=(); TAU_END=0.1; RHO=0.001
case "$i" in
  0) NAME="exp1_soft"
     MASK_ARGS=(--no-hard_gumbel_routing); TAU_END=0.5 ;;
  1) NAME="exp2_hard"
     MASK_ARGS=(--hard_gumbel_routing) ;;
  2) NAME="exp3_hard_dann"
     MASK_ARGS=(--hard_gumbel_routing); ADV_ARGS=(--dann_full_discriminator) ;;
  3) NAME="exp4_hard_ib"
     MASK_ARGS=(--hard_gumbel_routing)
     IB_ARGS=(--ub_weight 0.01 --ub_ramp_start 2000 --ub_ramp_end 6000); RHO=0.0 ;;
  *) echo "bad array index $i" >&2; exit 1 ;;
esac

SEED="${SEED:-42}"
STAGE2_STEPS="${STAGE2_STEPS:-8000}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
RUN_NAME="routing_${NAME}"
STAGE1_CKPT="${DIS_DIR}/checkpoints/ln_sae/stage1_best.pt"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"

echo "=== Routing experiment ${i}: ${NAME} ==="
echo "Job/Array  : ${SLURM_JOB_ID} / ${i}"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"
echo "mask       : ${MASK_ARGS[*]}   tau_end=${TAU_END}"
echo "adversary  : grl=0.5 grl_p=0.1  dann=[${ADV_ARGS[*]:-none}]"
echo "IB         : [${IB_ARGS[*]:-none}]   rho=${RHO}"

[[ -f "${STAGE1_CKPT}" ]] || { echo "ERROR: missing ${STAGE1_CKPT}" >&2; exit 2; }

# ----------------------------- Stage 2 (routing) -----------------------------
${PYTHON} -u run.py \
    --stage              2 \
    --stage1_ckpt        "${STAGE1_CKPT}" \
    --spear_layernorm \
    --n_routes           3 \
    "${MASK_ARGS[@]}" \
    --gumbel_tau_start   1.0 \
    --gumbel_tau_end     "${TAU_END}" \
    --routing_init_std   0.5 \
    --routing_spec_weight 0.01 \
    --lr_routing         1e-3 \
    --rho                "${RHO}" \
    "${IB_ARGS[@]}" \
    --alpha              0.02 \
    --beta               0.03 \
    --grl_weight         0.5 \
    --grl_phoneme_weight 0.1 \
    --grl_delay_steps    0 \
    "${ADV_ARGS[@]}" \
    --stage2_steps       "${STAGE2_STEPS}" \
    --warmup_steps       500 \
    --lr                 3e-5 \
    --lr_min             1e-6 \
    --lr_heads           1e-4 \
    --grad_log_every     500 \
    --checkpoint_dir     "${CKPT_DIR}" \
    --runs_dir           "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir            "${DIS_DIR}/logs" \
    --seed               "${SEED}"

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: stage 2 finished but checkpoint missing" >&2; exit 3; }

# ----------------------------- Probe (unified) -----------------------------
# Mask mode + eval tau MUST match training (routing carves z_L/z_P in the forward).
echo; echo "----- probe -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage1_ckpt        "${STAGE1_CKPT}" \
    --stage2_ckpt        "${STAGE2_CKPT}" \
    --run_name           "diag_probe_${RUN_NAME}" \
    --spear_layernorm \
    "${MASK_ARGS[@]}" \
    --gumbel_tau_end     "${TAU_END}" \
    --sources            "z_t,z_L,z_P" \
    --tasks              "pr,sid" \
    --probe_steps        "${PROBE_STEPS}" \
    --seed               "${SEED}" \
    --pr_max_examples    0 \
    --pr_probe_lr        5e-4 \
    --sid_probe_lr       1e-3 \
    --probe_warmup_steps 0

echo; echo "Finished   : $(date)"
