#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=routing_retest
#SBATCH --array=0-3%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/routing_retest/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/routing_retest/%x_%A_%a.err

# Routing re-test: does routing specialize now that (a) the heads are
# capacity-ordered, (b) the adversary is DANN-fixed, (c) lr_routing is raised,
# (d) init is random, and (e) a per-unit specialization loss (MI objective) is on?
# Routing mode (NO projection), n_routes=3, reuses the LayerNorm stage-1 SAE.
# Sweeps lr_routing x routing_spec_weight.  STE=1 / DYNAMIC=1 to toggle those.
# Watch: Hu (should fall), lstd (should rise), spec<.5 (should rise), then the probe.

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

mkdir -p "${DIS_DIR}/logs/train/stage2/routing_retest"
cd "${DIS_DIR}"

# ---- sweep grid: (lr_routing, spec_weight) ----
LR_ROUTINGS=(1e-3 1e-3 1e-2 1e-2)
SPEC_WEIGHTS=(0.01 0.05 0.01 0.05)
i="${SLURM_ARRAY_TASK_ID:-0}"
LR_ROUTING="${LR_ROUTING:-${LR_ROUTINGS[$i]}}"
SPEC_WEIGHT="${SPEC_WEIGHT:-${SPEC_WEIGHTS[$i]}}"

STE="${STE:-0}";        STE_ARGS=();  STE_TAG=""
DYNAMIC="${DYNAMIC:-0}"; DYN_ARGS=(); DYN_TAG=""
[[ "${STE}" == "1" ]]     && { STE_ARGS=(--ste_routing);    STE_TAG="_ste"; }
[[ "${DYNAMIC}" == "1" ]] && { DYN_ARGS=(--routing_dynamic); DYN_TAG="_dyn"; }

STAGE2_STEPS="${STAGE2_STEPS:-8000}"
SEED="${SEED:-42}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
LR_TAG="lr$(echo "${LR_ROUTING}" | tr '.-' 'pm')"
SPEC_TAG="spec$(echo "${SPEC_WEIGHT}" | tr '.' 'p')"
RUN_NAME="${RUN_NAME:-routing_${LR_TAG}_${SPEC_TAG}${STE_TAG}${DYN_TAG}}"

STAGE1_CKPT="${DIS_DIR}/checkpoints/ln_sae/stage1_best.pt"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"

echo "=== Routing re-test (reuse ln_sae) ==="
echo "Job/Array         : ${SLURM_JOB_ID} / ${i}"
echo "Node / GPU        : $(hostname) / $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started           : $(date)"
echo "run_name          : ${RUN_NAME}"
echo "lr_routing        : ${LR_ROUTING}    spec_weight=${SPEC_WEIGHT}"
echo "ste / dynamic     : ${STE} / ${DYNAMIC}    (hard Gumbel + random init are config defaults)"

if [[ ! -f "${STAGE1_CKPT}" ]]; then echo "ERROR: missing ${STAGE1_CKPT}" >&2; exit 2; fi

# ----------------------------- Stage 2 (routing) -----------------------------
${PYTHON} -u run.py \
    --stage              2 \
    --stage1_ckpt        "${STAGE1_CKPT}" \
    --spear_layernorm \
    --n_routes           3 \
    --dann_full_discriminator \
    --alpha              0.02 \
    --beta               0.03 \
    --grl_weight         0.3 \
    --grl_phoneme_weight 0.1 \
    --grl_delay_steps    0 \
    --rho                0.001 \
    --lr_routing         "${LR_ROUTING}" \
    --routing_spec_weight "${SPEC_WEIGHT}" \
    "${STE_ARGS[@]}" \
    "${DYN_ARGS[@]}" \
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

if [[ ! -f "${STAGE2_CKPT}" ]]; then echo "ERROR: stage 2 finished but checkpoint missing" >&2; exit 3; fi

# ----------------------------- Probe (unified) -----------------------------
echo; echo "----- probe -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage1_ckpt        "${STAGE1_CKPT}" \
    --stage2_ckpt        "${STAGE2_CKPT}" \
    --run_name           "diag_probe_${RUN_NAME}" \
    --spear_layernorm \
    --sources            "z_t,z_L,z_P" \
    --tasks              "pr,sid" \
    --probe_steps        "${PROBE_STEPS}" \
    --seed               "${SEED}" \
    --pr_max_examples    0 \
    --pr_probe_lr        5e-4 \
    --sid_probe_lr       1e-3 \
    --probe_warmup_steps 0

echo; echo "Finished          : $(date)"
