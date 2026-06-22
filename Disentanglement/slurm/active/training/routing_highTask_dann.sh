#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=routing_hiTask
#SBATCH --array=0-1%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/routing_hiTask_dann/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/routing_hiTask_dann/%x_%A_%a.err

# Two stage-2 runs (soft vs hard), both with HIGH task weights + DANN, calibrated
# from the prior runs' per-loss SAE grad norms (logs/.../routing_4exp).
#
# Prior diagnosis (exp3_hard_dann @8000): on the SAE features the scrubbing
# adversaries dwarfed the constructive tasks — effective |g|/recon was
#   PR-build-L 0.4x, SID-build-P 0.8x  vs  GRL-scrub-L 16x, GRL_p-scrub-P 23x.
# So evicted speaker features had no SID pull into P (actP collapsed to 1) and
# everything pooled in U (L/P/U = 653/162/4305).  Fix: bring the builds up to
# adversary parity (~16x recon):
#   alpha 0.02 -> 0.8   (0.8 x pr_raw 0.55  ≈ 0.44 ≈ grl)
#   beta  0.03 -> 0.6   (0.6 x sid_raw 0.76 ≈ 0.45 ≈ grl)  <-- the key P-starvation fix
#   grl   0.5 (kept, dann_full_discriminator: reversed gradient at full lambda)
#   grl_p 0.1 (kept)    raising beta drops the P scrub:build ratio 29:1 -> 1.4:1
# The DANN sigmoid ramp keeps scrubbers weak early, so L/P populate first (builds
# ~9-11x recon from step 500) and the adversaries clean them as they ramp in.
#
#   0  soft   soft masks (train+test), tau->0.5
#   1  hard   hard STE,                tau->0.1
# SAE held at lr 3e-5 (features re-sorted, not reshaped) — contrast with the
# separate routing_B_joint run (hard, SAE unfrozen at 1e-4).

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
mkdir -p "${DIS_DIR}/logs/train/stage2/routing_hiTask_dann"
cd "${DIS_DIR}"

i="${SLURM_ARRAY_TASK_ID:-0}"
# ---- per-experiment config (only the routing mode differs) ----
case "$i" in
  0) NAME="hiTask_soft"; MASK_ARGS=(--no-hard_gumbel_routing); TAU_END=0.5 ;;
  1) NAME="hiTask_hard"; MASK_ARGS=(--hard_gumbel_routing);    TAU_END=0.1 ;;
  *) echo "bad array index $i" >&2; exit 1 ;;
esac

SEED="${SEED:-42}"
STAGE2_STEPS="${STAGE2_STEPS:-8000}"
PROBE_STEPS="${PROBE_STEPS:-4000}"   # ceiling; dev early-stopping usually ends sooner
RUN_NAME="routing_${NAME}"
STAGE1_CKPT="${DIS_DIR}/checkpoints/ln_sae/stage1_best.pt"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"

echo "=== Routing experiment ${i}: ${NAME} (high-task + DANN) ==="
echo "Job/Array  : ${SLURM_JOB_ID} / ${i}"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"
echo "mask       : ${MASK_ARGS[*]}   tau_end=${TAU_END}"
echo "tasks      : alpha=0.8 beta=0.6   adversary grl=0.5 (dann) grl_p=0.1"
echo "SAE lr     : 3e-5 (re-sort, not reshape)   rho=0.001 spec=0.01"

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
    --rho                0.001 \
    --alpha              0.8 \
    --beta               0.6 \
    --grl_weight         0.5 \
    --grl_phoneme_weight 0.1 \
    --grl_delay_steps    0 \
    --dann_full_discriminator \
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
    --probe_val_every    "${PROBE_VAL_EVERY:-250}" \
    --probe_patience     "${PROBE_PATIENCE:-5}" \
    --seed               "${SEED}" \
    --pr_max_examples    0 \
    --pr_probe_lr        5e-4 \
    --sid_probe_lr       1e-3 \
    --probe_warmup_steps 0

echo; echo "Finished   : $(date)"
