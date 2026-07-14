#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_qfrz
#SBATCH --array=0-2%3
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.err

# MSP quota-freeze follow-up sweep.
#
# These are intentionally only three runs.  They build from the best current MSP
# direction: hard learned routing, freeze at 4k, calibrate route-local TopK quotas,
# and continue to 12k with reduced prosody/emotion adversaries.
#
# Task 0: stronger cleanup after freeze.
#   - doubles z_L speaker GRL normalized target: 2e-4 -> 4e-4
#   - strengthens anti-content pressure on z_P: grl_p 0.15 -> 0.25
#
# Task 1: no GRL normalization.
#   - keeps the adversary, but uses plain gradient reversal instead of per-frame
#     normalized GRL.  This tests whether normalization itself is causing odd
#     MSP speaker dynamics.
#
# Task 2: emotion-only P factor.
#   - keeps emotion supervision/adversary, removes prosody supervision/adversary.
#   - tests whether z_P is overloaded by prosody + emotion together.
#
# Submit:
#   cd /rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
#   sbatch msp/slurm/sweep_msp_qfreeze_followups.sh
#
# Submit only one case:
#   sbatch --array=1 msp/slurm/sweep_msp_qfreeze_followups.sh

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

mkdir -p "${DIS_DIR}/msp/logs"
cd "${DIS_DIR}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"

# Shared defaults.
STEPS=12000
FREEZE_STEP=4000
WARMUP_STEPS=500
DANN_RAMP_STEPS=12000
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
NUM_WORKERS=8
SEED=42

LR_SAE=1e-4
LR_MIN=1e-5
LR_HEADS=1e-4
LR_DISC=1e-3
LR_ROUTING=1e-3
N_DISC_STEPS=3
GRAD_CLIP=1.0

ROUTING_INIT_STD=0.5
ROUTING_SPEC_WEIGHT=0.01
ROUTING_TAU=1.0
ROUTE_TOPK_CALIB_BATCHES=20
PCGRAD_TASKS="recon,pr,sid,prosody,emotion"

PR_WEIGHT=0.8
RECON_WEIGHT=1.0
SID_WEIGHT=0.6
SPEAKER_GRL_WEIGHT=1.0
PHONEME_GRL_WEIGHT=0.15
PROSODY_WEIGHT=0.5
EMOTION_WEIGHT=0.5
PROSODY_GRL_WEIGHT=0.10
EMOTION_GRL_WEIGHT=0.10
INVARIANCE_WEIGHT=0.0

GRL_NORM_MODE=on
GRL_NORM_TARGET=0.0002

# Human-readable compact logs every 500 steps, detailed gradient diagnostics
# every 1000 steps.
LOG_EVERY=500
GRAD_LOG_EVERY=1000
CKPT_EVERY=1000

MANIFEST="${DIS_DIR}/data/msp_subset"
AUDIO_ROOT="${DIS_DIR}/data/msp_audio"
TRANSCRIPTS="/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0/Transcripts.zip"

case "${TASK_ID}" in
  0)
    RUN_NAME=msp_hardqfreeze4000_strongclean_dann12000_advpe010_s42
    GRL_NORM_TARGET=0.0004
    PHONEME_GRL_WEIGHT=0.25
    ;;
  1)
    RUN_NAME=msp_hardqfreeze4000_nognorm_dann12000_advpe010_s42
    GRL_NORM_MODE=off
    ;;
  2)
    RUN_NAME=msp_hardqfreeze4000_emoonly_dann12000_advemo010_s42
    PROSODY_WEIGHT=0.0
    PROSODY_GRL_WEIGHT=0.0
    PCGRAD_TASKS="recon,pr,sid,emotion"
    ;;
  *)
    echo "Unknown TASK_ID=${TASK_ID}. Expected 0,1,2." >&2
    exit 2
    ;;
esac

CHECKPOINT_DIR="${DIS_DIR}/msp/checkpoints/${RUN_NAME}"

GRL_NORM_ARGS=(--grl_grad_norm --grl_grad_norm_target "${GRL_NORM_TARGET}")
if [[ "${GRL_NORM_MODE}" == "off" ]]; then
  GRL_NORM_ARGS=(--no-grl_grad_norm)
fi

echo "=== MSP quota-freeze follow-up sweep ==="
echo "started    : $(date)"
echo "task_id    : ${TASK_ID}"
echo "gpu        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "run_name   : ${RUN_NAME}"
echo "steps      : ${STEPS}"
echo "dann_ramp  : ${DANN_RAMP_STEPS}"
echo "routing    : hard binary L/P, tau=${ROUTING_TAU}, init_std=${ROUTING_INIT_STD}, spec=${ROUTING_SPEC_WEIGHT}"
echo "learn/freeze: yes  freeze_step=${FREEZE_STEP}  quota_batches=${ROUTE_TOPK_CALIB_BATCHES}"
echo "learning   : sae=${LR_SAE} heads=${LR_HEADS} disc=${LR_DISC} routing=${LR_ROUTING} min=${LR_MIN}"
echo "disc       : steps=${N_DISC_STEPS} grad_clip=${GRAD_CLIP}"
echo "weights    : recon=${RECON_WEIGHT} pr=${PR_WEIGHT} sid=${SID_WEIGHT} grl=${SPEAKER_GRL_WEIGHT} grl_p=${PHONEME_GRL_WEIGHT}"
echo "weights    : pros=${PROSODY_WEIGHT}/${PROSODY_GRL_WEIGHT} emo=${EMOTION_WEIGHT}/${EMOTION_GRL_WEIGHT} inv=${INVARIANCE_WEIGHT}"
echo "pcgrad     : ${PCGRAD_TASKS}"
echo "zL_grl_norm: mode=${GRL_NORM_MODE} target=${GRL_NORM_TARGET}"
echo "logs       : compact_every=${LOG_EVERY} grad_every=${GRAD_LOG_EVERY}"

COMMON_ARGS=(
  --run_name "${RUN_NAME}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --manifest "${MANIFEST}" --audio_root "${AUDIO_ROOT}" --transcripts "${TRANSCRIPTS}"
  --steps "${STEPS}"
  --warmup_steps "${WARMUP_STEPS}" --dann_ramp_steps "${DANN_RAMP_STEPS}"
  --batch_size "${BATCH_SIZE}" --eval_batch "${EVAL_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}" --seed "${SEED}"
  --lr "${LR_SAE}" --lr_min "${LR_MIN}"
  --lr_heads "${LR_HEADS}" --lr_disc "${LR_DISC}" --lr_routing "${LR_ROUTING}"
  --n_disc_steps "${N_DISC_STEPS}" --grad_clip "${GRAD_CLIP}"
  --routing_init_std "${ROUTING_INIT_STD}"
  --routing_spec_weight "${ROUTING_SPEC_WEIGHT}" --routing_tau "${ROUTING_TAU}"
  --pcgrad_tasks "${PCGRAD_TASKS}"
  --recon_weight "${RECON_WEIGHT}"
  --alpha "${PR_WEIGHT}" --beta "${SID_WEIGHT}"
  --grl_weight "${SPEAKER_GRL_WEIGHT}" --grl_phoneme_weight "${PHONEME_GRL_WEIGHT}"
  --prosody_weight "${PROSODY_WEIGHT}" --grl_prosody_weight "${PROSODY_GRL_WEIGHT}"
  --emotion_weight "${EMOTION_WEIGHT}" --grl_emotion_weight "${EMOTION_GRL_WEIGHT}"
  --inv_weight "${INVARIANCE_WEIGHT}"
  --no_invariance
  "${GRL_NORM_ARGS[@]}"
  --log_every "${LOG_EVERY}" --grad_log_every "${GRAD_LOG_EVERY}" --ckpt_every "${CKPT_EVERY}"
)

echo "[phase 1] learn routing until step ${FREEZE_STEP}"
"${PYTHON}" -u -m msp.run "${COMMON_ARGS[@]}" \
  --resume none \
  --segment_steps "${FREEZE_STEP}" \
  --resume_every "${CKPT_EVERY}"

echo "[phase 2] exact resume, freeze learned routes, calibrate route-local TopK, continue"
"${PYTHON}" -u -m msp.run "${COMMON_ARGS[@]}" \
  --resume "${CHECKPOINT_DIR}/latest-resume.pt" \
  --freeze_learned_routing_on_resume \
  --freeze_route_topk_on_resume \
  --route_topk_calib_batches "${ROUTE_TOPK_CALIB_BATCHES}" \
  --resume_every "${CKPT_EVERY}"

echo "finished   : $(date)"
