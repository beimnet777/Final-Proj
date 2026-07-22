#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=msp_fix210L46P
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/msp/logs/%x_%j.err

# MSP fixed-routing control based on the healthy strong-clean route allocation.
# Fixed dictionary membership: 2130 L / 2990 P; fixed active budget: 210 L / 46 P.
# This is a single 12k run: there is no learned-routing or freeze continuation.
#
# Submit from the HPC Disentanglement directory:
#   sbatch msp/slurm/msp_fixed_empirical_210L46P_12k.sh

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

RUN_NAME=msp_fixed2130L2990P_topk210L46P_sc_optfix_aux64_gn0004_grlp025_dann12000_12k_s42
CHECKPOINT_DIR="${DIS_DIR}/msp/checkpoints/${RUN_NAME}"

MANIFEST="${DIS_DIR}/data/msp_subset"
AUDIO_ROOT="${DIS_DIR}/data/msp_audio"
TRANSCRIPTS=/rds/project/rds-xyBFuSj0hm0/dataset/MSP-Podcast-2.0/Transcripts.zip
LEXICON_PATH="${DIS_DIR}/../Probing/data/librispeech-lexicon.txt"

CMD=(
  "${PYTHON}" -u -m msp.run
  --run_name "${RUN_NAME}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --manifest "${MANIFEST}" --audio_root "${AUDIO_ROOT}"
  --transcripts "${TRANSCRIPTS}" --lexicon_path "${LEXICON_PATH}"
  --steps 12000 --warmup_steps 500 --dann_ramp_steps 12000
  --batch_size 16 --eval_batch 32 --num_workers 8 --seed 42
  --lr 1e-4 --lr_min 1e-5 --lr_heads 1e-4 --lr_disc 1e-3 --lr_routing 0
  --n_disc_steps 3 --grad_clip 1.0
  --separate_discriminator_optimizer --separate_grad_clip
  --fixed_blocks --K_L 2130 --K_P 2990 --K_U 0
  --per_block_topk --topk_L 210 --topk_P 46 --topk_U 0
  --routing_spec_weight 0
  --pcgrad_tasks recon,pr,sid,prosody,emotion,aux
  --pcgrad_balance none --adversary_balance none
  --recon_weight 1.0 --alpha 0.8 --beta 0.6
  --grl_weight 1.0 --grl_phoneme_weight 0.25
  --prosody_weight 0.5 --grl_prosody_weight 0.10
  --emotion_weight 0.5 --grl_emotion_weight 0.10
  --inv_weight 0 --no_invariance
  --grl_grad_norm --grl_grad_norm_target 0.0004
  --aux_k 64 --aux_k_coef 0.03125
  --dead_steps_threshold 256 --valid_frame_dead_count
  --log_every 500 --grad_log_every 1000 --ckpt_every 1000
  --resume none
)

echo "=== MSP empirical fixed-routing run ==="
echo "started     : $(date)"
echo "run_name    : ${RUN_NAME}"
echo "routing     : fixed blocks L/P/U=2130/2990/0"
echo "active      : fixed TopK L/P/U=210/46/0"
echo "schedule    : 12000 steps; sigmoid DANN ramp=12000"
echo "optimization: separate discriminator/clip; PCGrad cooperative, balance=none; AuxK=64"
printf '+ %q ' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "gpu         : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
  "${CMD[@]}"
fi

echo "finished    : $(date)"
