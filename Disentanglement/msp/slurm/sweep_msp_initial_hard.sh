#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_hard_init
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%A_%a.err

# Initial MSP hard-routing sweep.
#
# Purpose:
#   Learn the basic MSP dynamics before doing more niche learned/freeze variants.
#   No soft routing here. Each array task keeps hard ST-Gumbel routing and changes
#   one main hypothesis about early P starvation / adversary onset.
#
# Submit one at a time:
#   sbatch --array=0-3%1 msp/slurm/sweep_msp_initial_hard.sh
#
# Submit only one case:
#   sbatch --array=2 msp/slurm/sweep_msp_initial_hard.sh

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
WARMUP_STEPS=500
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
NUM_WORKERS=8
SEED=42

LR_SAE=1e-4
LR_MIN=1e-6
LR_HEADS=1e-4
LR_DISC=1e-3
LR_ROUTING=1e-3
N_DISC_STEPS=3
GRAD_CLIP=1.0

ROUTING_INIT_STD=0.5
ROUTING_SPEC_WEIGHT=0.01
ROUTING_TAU=1.0
PCGRAD_TASKS="recon,pr,sid,prosody,emotion"

PR_WEIGHT=0.8
SID_WEIGHT=0.6
SPEAKER_GRL_WEIGHT=1.0
PHONEME_GRL_WEIGHT=0.15
PROSODY_WEIGHT=0.5
PROSODY_GRL_WEIGHT=0.5
EMOTION_WEIGHT=0.5
EMOTION_GRL_WEIGHT=0.5
INVARIANCE_WEIGHT=0.0
GRL_NORM_TARGET=0.0002

LOG_EVERY=100
CKPT_EVERY=1000

MANIFEST="${DIS_DIR}/data/msp_subset"
AUDIO_ROOT="${DIS_DIR}/data/msp_audio"
TRANSCRIPTS="/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0/Transcripts.zip"

case "${TASK_ID}" in
  0)
    RUN_NAME=msp_hard_dann6000_spec001_s42
    DANN_RAMP_STEPS=6000
    ;;
  1)
    RUN_NAME=msp_hard_dann12000_spec001_s42
    DANN_RAMP_STEPS=12000
    ;;
  2)
    RUN_NAME=msp_hard_dann6000_spec0003_s42
    DANN_RAMP_STEPS=6000
    ROUTING_SPEC_WEIGHT=0.003
    ;;
  3)
    RUN_NAME=msp_hard_dann6000_spec0003_pboost_s42
    DANN_RAMP_STEPS=6000
    ROUTING_SPEC_WEIGHT=0.003
    SID_WEIGHT=0.8
    PROSODY_WEIGHT=0.7
    EMOTION_WEIGHT=0.7
    ;;
  *)
    echo "Unknown TASK_ID=${TASK_ID}. Expected 0,1,2,3." >&2
    exit 2
    ;;
esac

echo "=== MSP hard-routing initial sweep ==="
echo "started   : $(date)"
echo "task_id   : ${TASK_ID}"
echo "gpu       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "run_name  : ${RUN_NAME}"
echo "steps     : ${STEPS}"
echo "dann_ramp : ${DANN_RAMP_STEPS}"
echo "routing   : hard binary L/P, tau=${ROUTING_TAU}, init_std=${ROUTING_INIT_STD}, spec=${ROUTING_SPEC_WEIGHT}"
echo "learning  : sae=${LR_SAE} heads=${LR_HEADS} disc=${LR_DISC} routing=${LR_ROUTING} min=${LR_MIN}"
echo "disc      : steps=${N_DISC_STEPS} grad_clip=${GRAD_CLIP}"
echo "weights   : pr=${PR_WEIGHT} sid=${SID_WEIGHT} grl=${SPEAKER_GRL_WEIGHT} grl_p=${PHONEME_GRL_WEIGHT}"
echo "weights   : pros=${PROSODY_WEIGHT}/${PROSODY_GRL_WEIGHT} emo=${EMOTION_WEIGHT}/${EMOTION_GRL_WEIGHT} inv=${INVARIANCE_WEIGHT}"
echo "zL_grl_norm: ${GRL_NORM_TARGET}"

${PYTHON} -u -m msp.run \
    --run_name "${RUN_NAME}" \
    --manifest "${MANIFEST}" --audio_root "${AUDIO_ROOT}" --transcripts "${TRANSCRIPTS}" \
    --steps "${STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" --dann_ramp_steps "${DANN_RAMP_STEPS}" \
    --batch_size "${BATCH_SIZE}" --eval_batch "${EVAL_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
    --lr "${LR_SAE}" --lr_min "${LR_MIN}" \
    --lr_heads "${LR_HEADS}" --lr_disc "${LR_DISC}" --lr_routing "${LR_ROUTING}" \
    --n_disc_steps "${N_DISC_STEPS}" --grad_clip "${GRAD_CLIP}" \
    --routing_init_std "${ROUTING_INIT_STD}" \
    --routing_spec_weight "${ROUTING_SPEC_WEIGHT}" --routing_tau "${ROUTING_TAU}" \
    --pcgrad_tasks "${PCGRAD_TASKS}" \
    --alpha "${PR_WEIGHT}" --beta "${SID_WEIGHT}" \
    --grl_weight "${SPEAKER_GRL_WEIGHT}" --grl_phoneme_weight "${PHONEME_GRL_WEIGHT}" \
    --prosody_weight "${PROSODY_WEIGHT}" --grl_prosody_weight "${PROSODY_GRL_WEIGHT}" \
    --emotion_weight "${EMOTION_WEIGHT}" --grl_emotion_weight "${EMOTION_GRL_WEIGHT}" \
    --inv_weight "${INVARIANCE_WEIGHT}" \
    --no_invariance \
    --grl_grad_norm --grl_grad_norm_target "${GRL_NORM_TARGET}" \
    --log_every "${LOG_EVERY}" --ckpt_every "${CKPT_EVERY}"

echo "finished  : $(date)"
