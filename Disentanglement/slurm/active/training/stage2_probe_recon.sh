#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=recon_grl
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection_reconstruct/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/projection_reconstruct/%x_%A_%a.err

# Reconstructive-projection stage-2 + probe, REUSING the existing LayerNorm SAE
# (checkpoints/ln_sae) — no stage-1 retrain.  Sweeps the SPEAKER GRL only.
#   - Job A (sweep):  sbatch --array=0-2 stage2_probe_recon.sh           -> grl ∈ {0.02,0.04,0.1}
#   - Job B (frame):  sbatch --export=ALL,GRL_FRAME_LEVEL=1,GRL_WEIGHT=0.01 stage2_probe_recon.sh
# Probe runs BOTH label sets: dis (41) and superb (74).

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

# ---- speaker-GRL weight: array picks from the list; otherwise env / default ----
GRL_WEIGHTS=(0.02 0.04 0.1)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    GRL_WEIGHT="${GRL_WEIGHTS[$SLURM_ARRAY_TASK_ID]}"
fi
GRL_WEIGHT="${GRL_WEIGHT:-0.01}"
GRL_FRAME_LEVEL="${GRL_FRAME_LEVEL:-0}"

PROJECTION_DIM="${PROJECTION_DIM:-256}"
STAGE2_STEPS="${STAGE2_STEPS:-8000}"
SEED="${SEED:-42}"
ALPHA="${ALPHA:-0.02}"
BETA="${BETA:-0.03}"
GRL_P_WEIGHT="${GRL_P_WEIGHT:-0.01}"
PROBE_STEPS="${PROBE_STEPS:-2000}"

GRL_TAG="grl$(echo "${GRL_WEIGHT}" | tr '.' 'p')"
FRAME_ARGS=()
FRAME_TAG=""
if [[ "${GRL_FRAME_LEVEL}" == "1" ]]; then FRAME_ARGS=(--grl_frame_level); FRAME_TAG="_frame"; fi
RUN_NAME="${RUN_NAME:-proj_recon_ln_d${PROJECTION_DIM}_${GRL_TAG}${FRAME_TAG}}"

STAGE1_CKPT="${DIS_DIR}/checkpoints/ln_sae/stage1_best.pt"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"

echo "=== Reconstructive projection stage2 + probe (reuse ln_sae) ==="
echo "Job/Array         : ${SLURM_JOB_ID} / ${SLURM_ARRAY_TASK_ID:-none}"
echo "Node / GPU        : $(hostname) / $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started           : $(date)"
echo "run_name          : ${RUN_NAME}"
echo "grl_weight (SID)  : ${GRL_WEIGHT}    frame_level=${GRL_FRAME_LEVEL}"
echo "grl_p (phoneme)   : ${GRL_P_WEIGHT}  (frozen)"
echo "stage1_ckpt       : ${STAGE1_CKPT}"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
    echo "ERROR: missing LayerNorm stage-1 checkpoint: ${STAGE1_CKPT} (run the e2e / stage1_ln_sae job first)" >&2
    exit 2
fi

# ----------------------------- Stage 2 -----------------------------
${PYTHON} -u run.py \
    --stage                  2 \
    --stage1_ckpt            "${STAGE1_CKPT}" \
    --spear_layernorm \
    --projection_disentanglement \
    --projection_reconstruct \
    --projection_dim         "${PROJECTION_DIM}" \
    --stage2_steps           "${STAGE2_STEPS}" \
    --warmup_steps           500 \
    --alpha                  "${ALPHA}" \
    --beta                   "${BETA}" \
    --grl_weight             "${GRL_WEIGHT}" \
    --grl_delay_steps        0 \
    --grl_phoneme_weight     "${GRL_P_WEIGHT}" \
    "${FRAME_ARGS[@]}" \
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

# ----------------------- Probe: BOTH label sets -----------------------
for LABELS in dis superb; do
    echo; echo "----- probe (${LABELS}) -----"; date
    ${PYTHON} -u diag_probe/run.py \
        --stage1_ckpt        "${STAGE1_CKPT}" \
        --stage2_ckpt        "${STAGE2_CKPT}" \
        --run_name           "diag_probe_${RUN_NAME}_${LABELS}" \
        --spear_layernorm \
        --pr_label_set       "${LABELS}" \
        --sources            "z_t,z_L,z_P" \
        --tasks              "pr,sid" \
        --probe_steps        "${PROBE_STEPS}" \
        --seed               "${SEED}" \
        --pr_max_examples    0 \
        --pr_probe_lr        5e-4 \
        --sid_probe_lr       1e-3 \
        --probe_warmup_steps 0
done

echo; echo "Finished          : $(date)"
