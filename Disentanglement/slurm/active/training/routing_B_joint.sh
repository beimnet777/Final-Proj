#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=routing_B_joint
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/routing_B_joint/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/routing_B_joint/%x_%j.err

# Option B — JOINT end-to-end disentanglement.
#   - SAE UNFROZEN at full lr (1e-4): features can reshape, not just be re-sorted.
#   - Task losses raised ~25x (alpha/beta 0.02/0.03 -> 0.5/0.5) so PR/SID gradients
#     actually compete with reconstruction for control of the features.
#   - Speaker adversary on z_L at full DANN strength (grl=1.0), ramped in after a
#     short delay so L populates/specializes BEFORE speaker info is scrubbed.
#   - rho raised 0.001 -> 0.01 to help the routing bootstrap out of the U sink.
# Per-loss SAE grad norms are logged every 500 steps ([grad_norms @step]) — read
# them to check the task/adversary terms reached comparable magnitude to recon and
# re-tune alpha/beta/grl for the next pass if needed.
#
# Warm-started from the ln_sae stage-1 checkpoint; this is joint fine-tuning of the
# whole stack rather than from-scratch (cheaper, same co-adaptation dynamics).

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
mkdir -p "${DIS_DIR}/logs/train/stage2/routing_B_joint"
cd "${DIS_DIR}"

SEED="${SEED:-42}"
STAGE2_STEPS="${STAGE2_STEPS:-8000}"
PROBE_STEPS="${PROBE_STEPS:-4000}"   # ceiling; dev early-stopping usually ends sooner
TAU_END="${TAU_END:-0.1}"
RUN_NAME="${RUN_NAME:-routing_B_joint}"
MAX_TRAIN="${MAX_TRAIN:-0}"          # 0 = full train-clean-100 (smoke: small)
MAX_EVAL="${MAX_EVAL:-500}"          # val/test cap (smoke: small)
STAGE1_CKPT="${DIS_DIR}/checkpoints/ln_sae/stage1_best.pt"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"

echo "=== Option B: joint end-to-end disentanglement ==="
echo "Job        : ${SLURM_JOB_ID}"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"
echo "SAE        : UNFROZEN, lr=1e-4 (joint)"
echo "tasks      : alpha=0.5 beta=0.5   adversary grl=1.0 (ramped, delay 500) grl_p=0.2"
echo "routing    : hard STE, rho=0.01, spec=0.01, tau 1.0->${TAU_END}"

[[ -f "${STAGE1_CKPT}" ]] || { echo "ERROR: missing ${STAGE1_CKPT}" >&2; exit 2; }

# ----------------------------- Stage 2 (joint) -----------------------------
${PYTHON} -u run.py \
    --stage              2 \
    --stage1_ckpt        "${STAGE1_CKPT}" \
    --spear_layernorm \
    --n_routes           3 \
    --hard_gumbel_routing \
    --gumbel_tau_start   1.0 \
    --gumbel_tau_end     "${TAU_END}" \
    --routing_init_std   0.5 \
    --routing_spec_weight 0.01 \
    --lr_routing         1e-3 \
    --rho                0.01 \
    --alpha              0.5 \
    --beta               0.5 \
    --grl_weight         1.0 \
    --grl_phoneme_weight 0.2 \
    --grl_delay_steps    500 \
    --dann_full_discriminator \
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

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: stage 2 finished but checkpoint missing" >&2; exit 3; }

# ----------------------------- Probe (unified) -----------------------------
# Mask mode + eval tau MUST match training (routing carves z_L/z_P in the forward).
echo; echo "----- probe -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage1_ckpt        "${STAGE1_CKPT}" \
    --stage2_ckpt        "${STAGE2_CKPT}" \
    --run_name           "diag_probe_${RUN_NAME}" \
    --spear_layernorm \
    --hard_gumbel_routing \
    --gumbel_tau_end     "${TAU_END}" \
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
