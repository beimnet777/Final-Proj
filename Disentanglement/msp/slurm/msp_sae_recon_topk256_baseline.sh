#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_sae256_base
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.err

# MSP pure-SAE reconstruction baseline.
#
# Purpose:
#   Train the MSP SAE with the standard top-k=256 objective, using reconstruction
#   only, then probe z_t with independent frozen probes for:
#     PR/PER, SID, emotion UAR, and prosody.
#
# Submit from the HPC repo:
#   cd /rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
#   sbatch msp/slurm/msp_sae_recon_topk256_baseline.sh

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

mkdir -p "${DIS_DIR}/msp/logs" "${DIS_DIR}/msp/probe_results"
cd "${DIS_DIR}"

RUN_NAME=msp_sae_recon_topk256_12k_s42
CHECKPOINT_DIR="${DIS_DIR}/msp/checkpoints/${RUN_NAME}"
PROBE_JSON="${DIS_DIR}/msp/probe_results/${RUN_NAME}_zt_pr_sid_emo_prosody_probe_5k_seed42.json"

MANIFEST="${DIS_DIR}/data/msp_subset"
AUDIO_ROOT="${DIS_DIR}/data/msp_audio"
TRANSCRIPTS="/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0/Transcripts.zip"

STEPS=12000
WARMUP_STEPS=500
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
NUM_WORKERS=8
SEED=42

LR_SAE=1e-4
LR_MIN=1e-5
LR_HEADS=1e-4
LR_DISC=1e-3
LR_ROUTING=1e-3
N_DISC_STEPS=1
GRAD_CLIP=1.0

ROUTING_INIT_STD=0.5
ROUTING_SPEC_WEIGHT=0.0
ROUTING_TAU=1.0

LOG_EVERY=500
GRAD_LOG_EVERY=1000
CKPT_EVERY=1000

echo "=== MSP pure SAE reconstruction baseline ==="
echo "started    : $(date)"
echo "gpu        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "run_name   : ${RUN_NAME}"
echo "topk       : 256 (DISConfig default)"
echo "steps      : ${STEPS}"
echo "objective  : reconstruction only; all supervised/adversarial weights set to 0"
echo "checkpoint : ${CHECKPOINT_DIR}/final.pt"
echo "probe      : z_t only; PR/SID/emotion/prosody; steps=5000"

"${PYTHON}" -u -m msp.run \
  --run_name "${RUN_NAME}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --manifest "${MANIFEST}" --audio_root "${AUDIO_ROOT}" --transcripts "${TRANSCRIPTS}" \
  --steps "${STEPS}" \
  --warmup_steps "${WARMUP_STEPS}" --dann_ramp_steps "${WARMUP_STEPS}" \
  --batch_size "${BATCH_SIZE}" --eval_batch "${EVAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
  --lr "${LR_SAE}" --lr_min "${LR_MIN}" \
  --lr_heads "${LR_HEADS}" --lr_disc "${LR_DISC}" --lr_routing "${LR_ROUTING}" \
  --n_disc_steps "${N_DISC_STEPS}" --grad_clip "${GRAD_CLIP}" \
  --routing_init_std "${ROUTING_INIT_STD}" \
  --routing_spec_weight "${ROUTING_SPEC_WEIGHT}" --routing_tau "${ROUTING_TAU}" \
  --no_pcgrad --pcgrad_tasks recon \
  --recon_weight 1.0 \
  --alpha 0.0 --beta 0.0 \
  --grl_weight 0.0 --grl_phoneme_weight 0.0 \
  --prosody_weight 0.0 --grl_prosody_weight 0.0 \
  --emotion_weight 0.0 --grl_emotion_weight 0.0 \
  --inv_weight 0.0 \
  --no_invariance \
  --log_every "${LOG_EVERY}" --grad_log_every "${GRAD_LOG_EVERY}" --ckpt_every "${CKPT_EVERY}"

if [[ ! -f "${CHECKPOINT_DIR}/final.pt" ]]; then
  echo "Missing final checkpoint: ${CHECKPOINT_DIR}/final.pt" >&2
  exit 4
fi

echo
echo "=== MSP z_t independent baseline probes ==="
echo "started probe: $(date)"
echo "output       : ${PROBE_JSON}"

"${PYTHON}" -u -m msp.probe \
  --checkpoint "${CHECKPOINT_DIR}/final.pt" \
  --manifest "${MANIFEST}" --audio_root "${AUDIO_ROOT}" --transcripts "${TRANSCRIPTS}" \
  --steps 5000 --val_every 500 \
  --batch_size 16 --eval_batch 32 --num_workers 8 \
  --lr 5e-4 --seed "${SEED}" \
  --sources z_t \
  --tasks pr,sid,emotion,prosody \
  --output "${PROBE_JSON}"

echo "finished   : $(date)"
