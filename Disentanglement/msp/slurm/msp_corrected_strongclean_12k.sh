#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_sc_fix12k
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.err

# Corrected MSP strong-clean experiment.
#
# One controlled run implementing the six conclusions from the MSP trajectory
# analysis:
#   1. clip SAE, router, positive heads, and discriminator heads separately;
#   2. train adversaries with a separate optimizer on detached representations;
#   3. unit-balance raw cooperative SAE gradients, then apply task weights/PCGrad;
#   4. revive dead units with valid-frame AuxK-64 residual reconstruction;
#   5. restore moderate strong-clean positive weights;
#   6. stop at 12k, retaining 8k/10k/12k checkpoints for comparison.
#
# Submit from the HPC Disentanglement directory:
#   sbatch msp/slurm/msp_corrected_strongclean_12k.sh

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module purge 2>/dev/null || true
  module load rhel8/default-amp 2>/dev/null || true
fi

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
DIS_DIR="${DIS_DIR:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement}"
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"

mkdir -p "${DIS_DIR}/msp/logs"
cd "${DIS_DIR}"

# Companion scripts may override these variables before executing this common
# recipe. Defaults preserve the already-run corrected experiment exactly.
RUN_NAME="${MSP_RUN_NAME:-msp_hardqfreeze4000_sc_optfix_aux64_gn0004_grlp025_dann12000_12k_s42}"
PCGRAD_BALANCE="${MSP_PCGRAD_BALANCE:-unit}"
ADVERSARY_BALANCE="${MSP_ADVERSARY_BALANCE:-none}"
case "${PCGRAD_BALANCE}" in
  unit|none) ;;
  *) echo "Unsupported MSP_PCGRAD_BALANCE=${PCGRAD_BALANCE}; expected unit or none" >&2; exit 2 ;;
esac
case "${ADVERSARY_BALANCE}" in
  none|unit_preserve_bundle) ;;
  *) echo "Unsupported MSP_ADVERSARY_BALANCE=${ADVERSARY_BALANCE}" >&2; exit 2 ;;
esac
CHECKPOINT_DIR="${DIS_DIR}/msp/checkpoints/${RUN_NAME}"

MANIFEST="${DIS_DIR}/data/msp_subset"
AUDIO_ROOT="${DIS_DIR}/data/msp_audio"
TRANSCRIPTS=/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0/Transcripts.zip
LEXICON_PATH="${DIS_DIR}/../Probing/data/librispeech-lexicon.txt"

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

RECON_WEIGHT=1.0
PR_WEIGHT=0.8
SID_WEIGHT=0.6
SPEAKER_GRL_WEIGHT=1.0
SPEAKER_GRL_NORM_TARGET=0.0004
PHONEME_GRL_WEIGHT=0.25
PROSODY_WEIGHT=0.5
PROSODY_GRL_WEIGHT=0.10
EMOTION_WEIGHT=0.5
EMOTION_GRL_WEIGHT=0.10
INVARIANCE_WEIGHT=0.0

AUX_K=64
AUX_K_COEF=0.03125
DEAD_STEPS_THRESHOLD=256
PCGRAD_TASKS=recon,pr,sid,prosody,emotion,aux

LOG_EVERY=500
GRAD_LOG_EVERY=1000
CKPT_EVERY=1000

echo "=== MSP corrected strong-clean 12k ==="
echo "started     : $(date)"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "gpu         : dry-run"
else
  echo "gpu         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
fi
echo "run_name    : ${RUN_NAME}"
echo "schedule    : total=${STEPS} freeze=${FREEZE_STEP} DANN=${DANN_RAMP_STEPS}"
echo "optimization: separate_disc=yes separate_clip=yes PCGrad=${PCGRAD_TASKS} balance=${PCGRAD_BALANCE} adversary_balance=${ADVERSARY_BALANCE}"
echo "positive    : recon=${RECON_WEIGHT} pr=${PR_WEIGHT} sid=${SID_WEIGHT} pros=${PROSODY_WEIGHT} emo=${EMOTION_WEIGHT}"
echo "adversarial : speaker=${SPEAKER_GRL_WEIGHT}/GN${SPEAKER_GRL_NORM_TARGET} phone=${PHONEME_GRL_WEIGHT} pros=${PROSODY_GRL_WEIGHT} emo=${EMOTION_GRL_WEIGHT}"
echo "dead revival: AuxK=${AUX_K} coef=${AUX_K_COEF} threshold=${DEAD_STEPS_THRESHOLD} valid_frames=yes"
echo "checkpoints : every ${CKPT_EVERY}; compare steps 8000, 10000, 12000"

COMMON_ARGS=(
  --run_name "${RUN_NAME}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --manifest "${MANIFEST}" --audio_root "${AUDIO_ROOT}"
  --transcripts "${TRANSCRIPTS}" --lexicon_path "${LEXICON_PATH}"
  --steps "${STEPS}" --warmup_steps "${WARMUP_STEPS}"
  --dann_ramp_steps "${DANN_RAMP_STEPS}"
  --batch_size "${BATCH_SIZE}" --eval_batch "${EVAL_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}" --seed "${SEED}"
  --lr "${LR_SAE}" --lr_min "${LR_MIN}"
  --lr_heads "${LR_HEADS}" --lr_disc "${LR_DISC}" --lr_routing "${LR_ROUTING}"
  --n_disc_steps "${N_DISC_STEPS}" --grad_clip "${GRAD_CLIP}"
  --separate_discriminator_optimizer --separate_grad_clip
  --routing_init_std "${ROUTING_INIT_STD}"
  --routing_spec_weight "${ROUTING_SPEC_WEIGHT}" --routing_tau "${ROUTING_TAU}"
  --pcgrad_tasks "${PCGRAD_TASKS}" --pcgrad_balance "${PCGRAD_BALANCE}"
  --adversary_balance "${ADVERSARY_BALANCE}"
  --recon_weight "${RECON_WEIGHT}" --alpha "${PR_WEIGHT}" --beta "${SID_WEIGHT}"
  --grl_weight "${SPEAKER_GRL_WEIGHT}"
  --grl_phoneme_weight "${PHONEME_GRL_WEIGHT}"
  --prosody_weight "${PROSODY_WEIGHT}"
  --grl_prosody_weight "${PROSODY_GRL_WEIGHT}"
  --emotion_weight "${EMOTION_WEIGHT}"
  --grl_emotion_weight "${EMOTION_GRL_WEIGHT}"
  --inv_weight "${INVARIANCE_WEIGHT}" --no_invariance
  --grl_grad_norm --grl_grad_norm_target "${SPEAKER_GRL_NORM_TARGET}"
  --aux_k "${AUX_K}" --aux_k_coef "${AUX_K_COEF}"
  --dead_steps_threshold "${DEAD_STEPS_THRESHOLD}" --valid_frame_dead_count
  --log_every "${LOG_EVERY}" --grad_log_every "${GRAD_LOG_EVERY}"
  --ckpt_every "${CKPT_EVERY}"
)

run_or_print() {
  echo
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

echo "[phase 1] learn routing to step ${FREEZE_STEP}"
run_or_print "${PYTHON}" -u -m msp.run "${COMMON_ARGS[@]}" \
  --resume none \
  --segment_steps "${FREEZE_STEP}" \
  --resume_every "${CKPT_EVERY}"

echo "[phase 2] exact resume, freeze learned routes and learned active quotas"
run_or_print "${PYTHON}" -u -m msp.run "${COMMON_ARGS[@]}" \
  --resume "${CHECKPOINT_DIR}/latest-resume.pt" \
  --freeze_learned_routing_on_resume \
  --freeze_route_topk_on_resume \
  --route_topk_calib_batches "${ROUTE_TOPK_CALIB_BATCHES}" \
  --resume_every "${CKPT_EVERY}"

echo "finished    : $(date)"
