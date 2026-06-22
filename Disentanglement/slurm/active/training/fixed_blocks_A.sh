#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=fixed_blocks_A
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/fixed_blocks_A/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/fixed_blocks_A/%x_%j.err

# Option A — fixed-block SUPERVISED SAE (routing is dead).
#   - K partitioned into FIXED index blocks  L=3072 / P=1024 / U=1024.
#   - PER-BLOCK TopK  160 / 64 / 32  → each factor a guaranteed active budget
#     (no global-TopK starvation; the failure that gave actP=1).
#   - Supervision shapes the DICTIONARY: PR(z_L) + SID(z_P) build, GRL(z_L) +
#     GRL_p(z_P) scrub the cross-factor.  No Gumbel router, no lr_routing.
#   - Trained FROM SCRATCH (no stage1_ckpt) so features are born monosemantic
#     instead of un-mixed from a polysemantic basin (the stage-2/B failure).
#   - SAE at FULL lr 1e-4 the whole run — the encoder reshapes, ungated.
# Task weights reuse the grad-calibrated values (alpha 0.8 / beta 0.6 ≈ adversary
# parity); adversary delayed so recon + blocks form first, then the scrub ramps in.

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
mkdir -p "${DIS_DIR}/logs/train/stage2/fixed_blocks_A"
cd "${DIS_DIR}"

SEED="${SEED:-42}"
STAGE2_STEPS="${STAGE2_STEPS:-12000}"   # from scratch → more steps than warm-started runs
PROBE_STEPS="${PROBE_STEPS:-4000}"
KL="${KL:-3072}"; KP="${KP:-1024}"; KU="${KU:-1024}"
TKL="${TKL:-160}"; TKP="${TKP:-64}"; TKU="${TKU:-32}"
MAX_TRAIN="${MAX_TRAIN:-0}"; MAX_EVAL="${MAX_EVAL:-500}"
RUN_NAME="${RUN_NAME:-fixed_blocks_A}"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"

echo "=== Option A: fixed-block supervised SAE (from scratch) ==="
echo "Job        : ${SLURM_JOB_ID}"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"
echo "blocks     : L/P/U = ${KL}/${KP}/${KU}   per-block topk = ${TKL}/${TKP}/${TKU}"
echo "tasks      : alpha=0.8 beta=0.6   adversary grl=0.5 (dann, delay 1500) grl_p=0.1"
echo "SAE lr     : 1e-4 (full, ungated)   steps=${STAGE2_STEPS}"

# ----------------------------- Train (supervised SAE) -----------------------------
${PYTHON} -u run.py \
    --stage              2 \
    --stage2_from_scratch \
    --fixed_blocks \
    --K_L ${KL} --K_P ${KP} --K_U ${KU} \
    --topk_L ${TKL} --topk_P ${TKP} --topk_U ${TKU} \
    --spear_layernorm \
    --alpha              0.8 \
    --beta               0.6 \
    --grl_weight         0.5 \
    --grl_phoneme_weight 0.1 \
    --grl_delay_steps    1500 \
    --dann_full_discriminator \
    --rho                0.0 \
    --stage2_steps       "${STAGE2_STEPS}" \
    --max_train_examples "${MAX_TRAIN}" \
    --max_val_examples   "${MAX_EVAL}" \
    --warmup_steps       500 \
    --lr                 1e-4 \
    --lr_min             1e-6 \
    --lr_heads           1e-4 \
    --grad_log_every     500 \
    --checkpoint_dir     "${CKPT_DIR}" \
    --runs_dir           "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir            "${DIS_DIR}/logs" \
    --seed               "${SEED}"

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: training finished but checkpoint missing" >&2; exit 3; }

# ----------------------------- Probe (unified) -----------------------------
# fixed_blocks + block sizes MUST match training (the SAE per-block TopK + masks).
echo; echo "----- probe -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt        "${STAGE2_CKPT}" \
    --stage1_ckpt        "${STAGE2_CKPT}" \
    --run_name           "diag_probe_${RUN_NAME}" \
    --spear_layernorm \
    --fixed_blocks \
    --K_L ${KL} --K_P ${KP} --K_U ${KU} \
    --topk_L ${TKL} --topk_P ${TKP} --topk_U ${TKU} \
    --sources            "z_t,z_L,z_P" \
    --tasks              "pr,sid" \
    --probe_steps        "${PROBE_STEPS}" \
    --probe_val_every    "${PROBE_VAL_EVERY:-250}" \
    --probe_patience     "${PROBE_PATIENCE:-5}" \
    --max_train_examples "${MAX_TRAIN}" \
    --max_val_examples   "${MAX_EVAL}" \
    --max_test_examples  "${MAX_EVAL}" \
    --seed               "${SEED}" \
    --pr_max_examples    0 \
    --pr_probe_lr        5e-4 \
    --sid_probe_lr       1e-3 \
    --probe_warmup_steps 0

echo; echo "Finished   : $(date)"
