#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_wpre_ret
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.err

# MSP binary L/P qfreeze run: retention-dominant variant.
#
# Compared with msp_weakpre_retain_affect_20k.sh:
#   - speaker GRL norm target: 3e-4 -> 4e-4
#   - stronger positive retention: emotion 0.75 -> 1.0, prosody 0.60 -> 0.75
#   - keeps pre-freeze adversaries ON at 0.10; no no-adversary pre-freeze phase
#   - post-freeze affect cleanup is moderate: 0.15, not SC++-harsh
#
# Submit on HPC:
#   cd /rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
#   sbatch msp/slurm/msp_weakpre_retainmax_gn0004_20k.sh

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

RUN_NAME=msp_hardqfreeze4000_weakpre_retainmax_gn0004_grlp010to025_advpe010to015_20k_s42
CHECKPOINT_DIR="${DIS_DIR}/msp/checkpoints/${RUN_NAME}"

# Data paths on Cambridge HPC.
MANIFEST="${DIS_DIR}/data/msp_subset"
AUDIO_ROOT="${DIS_DIR}/data/msp_audio"
TRANSCRIPTS="/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0/Transcripts.zip"
LEXICON_PATH="${DIS_DIR}/../Probing/data/librispeech-lexicon.txt"
[[ -f "${LEXICON_PATH}" ]] || { echo "Missing lexicon: ${LEXICON_PATH}" >&2; exit 8; }

# Schedule.
STEPS=20000
FREEZE_STEP=4000
WARMUP_STEPS=500
DANN_RAMP_STEPS=12000
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
NUM_WORKERS=8
SEED=42

# Optimisation.
LR_SAE=1e-4
LR_MIN=1e-5
LR_HEADS=1e-4
LR_DISC=1e-3
LR_ROUTING=1e-3
N_DISC_STEPS=3
GRAD_CLIP=1.0

# Routing.
ROUTING_INIT_STD=0.5
ROUTING_SPEC_WEIGHT=0.01
ROUTING_TAU=1.0
ROUTE_TOPK_CALIB_BATCHES=20

# Cooperative weights: make z_P a stronger home for speaker + affect.
RECON_WEIGHT=1.0
PR_WEIGHT=0.8
SID_WEIGHT=1.0
PROSODY_WEIGHT=0.75
EMOTION_WEIGHT=1.0
INVARIANCE_WEIGHT=0.0
PCGRAD_TASKS="recon,pr,sid,prosody,emotion"

# Speaker cleanup: return to the historically safe strong-clean target.
SPEAKER_GRL_WEIGHT=1.0
SPEAKER_GRL_NORM_TARGET=0.0004

# Pre-freeze adversaries stay on; this is not a no-adversary pre-freeze run.
PRE_PHONEME_GRL_WEIGHT=0.10
PRE_PROSODY_GRL_WEIGHT=0.10
PRE_EMOTION_GRL_WEIGHT=0.10

# Post-freeze cleanup: stronger than pre-freeze, gentler than SC++.
POST_PHONEME_GRL_WEIGHT=0.25
POST_PROSODY_GRL_WEIGHT=0.15
POST_EMOTION_GRL_WEIGHT=0.15

# Logging/checkpoints.
LOG_EVERY=500
GRAD_LOG_EVERY=1000
CKPT_EVERY=1000

BASE_ARGS=(
  --run_name "${RUN_NAME}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --manifest "${MANIFEST}" --audio_root "${AUDIO_ROOT}" --transcripts "${TRANSCRIPTS}" --lexicon_path "${LEXICON_PATH}"
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
  --grl_weight "${SPEAKER_GRL_WEIGHT}"
  --prosody_weight "${PROSODY_WEIGHT}"
  --emotion_weight "${EMOTION_WEIGHT}"
  --inv_weight "${INVARIANCE_WEIGHT}"
  --no_invariance
  --grl_grad_norm --grl_grad_norm_target "${SPEAKER_GRL_NORM_TARGET}"
  --log_every "${LOG_EVERY}" --grad_log_every "${GRAD_LOG_EVERY}" --ckpt_every "${CKPT_EVERY}"
)

echo "=== MSP weak-pre / retain-max qfreeze 20k ==="
echo "started    : $(date)"
echo "gpu        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "run_name   : ${RUN_NAME}"
echo "steps      : ${STEPS}"
echo "dann_ramp  : ${DANN_RAMP_STEPS}"
echo "routing    : hard binary L/P, tau=${ROUTING_TAU}, init_std=${ROUTING_INIT_STD}, spec=${ROUTING_SPEC_WEIGHT}"
echo "learn/freeze: yes  freeze_step=${FREEZE_STEP}  quota_batches=${ROUTE_TOPK_CALIB_BATCHES}"
echo "learning   : sae=${LR_SAE} heads=${LR_HEADS} disc=${LR_DISC} routing=${LR_ROUTING} min=${LR_MIN}"
echo "disc       : steps=${N_DISC_STEPS} grad_clip=${GRAD_CLIP}"
echo "positive   : recon=${RECON_WEIGHT} pr=${PR_WEIGHT} sid=${SID_WEIGHT} prosody=${PROSODY_WEIGHT} emotion=${EMOTION_WEIGHT}"
echo "speaker GN : grl=${SPEAKER_GRL_WEIGHT} target=${SPEAKER_GRL_NORM_TARGET}"
echo "pre adv    : grl_p=${PRE_PHONEME_GRL_WEIGHT} pros=${PRE_PROSODY_GRL_WEIGHT} emo=${PRE_EMOTION_GRL_WEIGHT}"
echo "post adv   : grl_p=${POST_PHONEME_GRL_WEIGHT} pros=${POST_PROSODY_GRL_WEIGHT} emo=${POST_EMOTION_GRL_WEIGHT}"
echo "pcgrad     : ${PCGRAD_TASKS}"
echo "logs       : compact_every=${LOG_EVERY} grad_every=${GRAD_LOG_EVERY}"

echo "[phase 1] learn routing until step ${FREEZE_STEP} with moderate pre-freeze adversaries"
"${PYTHON}" -u -m msp.run "${BASE_ARGS[@]}" \
  --grl_phoneme_weight "${PRE_PHONEME_GRL_WEIGHT}" \
  --grl_prosody_weight "${PRE_PROSODY_GRL_WEIGHT}" \
  --grl_emotion_weight "${PRE_EMOTION_GRL_WEIGHT}" \
  --resume none \
  --segment_steps "${FREEZE_STEP}" \
  --resume_every "${CKPT_EVERY}"

echo "[phase 2] freeze learned routes + quota TopK, then continue with retention-dominant cleanup"
"${PYTHON}" -u -m msp.run "${BASE_ARGS[@]}" \
  --grl_phoneme_weight "${POST_PHONEME_GRL_WEIGHT}" \
  --grl_prosody_weight "${POST_PROSODY_GRL_WEIGHT}" \
  --grl_emotion_weight "${POST_EMOTION_GRL_WEIGHT}" \
  --resume "${CHECKPOINT_DIR}/latest-resume.pt" \
  --freeze_learned_routing_on_resume \
  --freeze_route_topk_on_resume \
  --route_topk_calib_batches "${ROUTE_TOPK_CALIB_BATCHES}" \
  --resume_every "${CKPT_EVERY}"

echo "finished   : $(date)"
