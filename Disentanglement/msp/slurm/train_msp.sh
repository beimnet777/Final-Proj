#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.err

# Standalone MSP-Podcast disentanglement: content + speaker + prosody + emotion in
# one dataset, per-batch, with PCGrad over the cooperative tasks.

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

# Complete experiment specification. Submit with no environment variables:
#     sbatch msp/slurm/train_msp.sh
RUN_NAME=msp_v3
STEPS=12000
WARMUP_STEPS=500
# Keep LR warmup short, but let the adversaries turn on more gradually.
# The initial MSP logs show active P capacity collapsing very early when the
# adversary reaches full strength by step 500.
DANN_RAMP_STEPS=6000
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
LEXICON_PATH="${DIS_DIR}/../Probing/data/librispeech-lexicon.txt"
[[ -f "${LEXICON_PATH}" ]] || { echo "Missing lexicon: ${LEXICON_PATH}" >&2; exit 8; }

echo "=== MSP standalone disentanglement ==="
echo "started   : $(date)"
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
    --manifest "${MANIFEST}" --audio_root "${AUDIO_ROOT}" --transcripts "${TRANSCRIPTS}" --lexicon_path "${LEXICON_PATH}" \
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
