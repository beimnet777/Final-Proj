#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=libri_zL_asr_rerun
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/blackwell/logs/%x_%j.err

# Rerun only z_L ASR for the post-freeze gp=0.30 learned-qfreeze checkpoint.
#
# Submit from the HPC repo root:
#   sbatch Disentanglement/blackwell/slurm/rerun_postgp030_zL_asr_hpc.sh

set -euo pipefail

if [[ -f /etc/profile.d/modules.sh ]]; then
  . /etc/profile.d/modules.sh
  module purge 2>/dev/null || true
  module load rhel8/default-amp 2>/dev/null || true
fi

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
REPO_ROOT="${REPO_ROOT:-/rds/user/bbg25/hpc-work/Thesis/Final-Proj}"
DIS_DIR="${DIS_DIR:-${REPO_ROOT}/Disentanglement}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/runs/blackwell_followups}"
LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:-${REPO_ROOT}/Probing/data/LibriSpeech}"
LEXICON_PATH="${LEXICON_PATH:-${REPO_ROOT}/Probing/data/librispeech-lexicon.txt}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-43}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/Probing/data/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/Probing/data/datasets_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/Probing/data/hub_cache}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

RUN="libri_advlearn_hardqfreeze4000_postgp030_fromgp02base_dann6800_gn0002_aux64_20k_s42"
CKPT_DIR="${RUNS_ROOT}/${RUN}/checkpoints"
if [[ -f "${CKPT_DIR}/stage2_step20000.pt" ]]; then
  CKPT="${CKPT_DIR}/stage2_step20000.pt"
elif [[ -f "${CKPT_DIR}/final.pt" ]]; then
  CKPT="${CKPT_DIR}/final.pt"
else
  echo "[error] missing checkpoint: expected stage2_step20000.pt or final.pt under ${CKPT_DIR}" >&2
  exit 1
fi

LOG_ROOT="${DIS_DIR}/blackwell/logs/postgp030_zL_asr_rerun"
JSON_ROOT="${LOG_ROOT}/json/${RUN}"
mkdir -p "${JSON_ROOT}"

cd "${REPO_ROOT}"

RUN_ID="diag_${RUN}_postgp030_zL_asr_lstm_10k_rerun_seed${SEED}"
OUT_JSON="${JSON_ROOT}/${RUN_ID}.json"

echo "[rerun] repo=${REPO_ROOT}"
echo "[rerun] checkpoint=${CKPT}"
echo "[rerun] output_json=${OUT_JSON}"
echo "[rerun] source=z_L task=asr seed=${SEED}"

"${PYTHON}" -u Disentanglement/diag_probe/run.py \
  --stage2_ckpt "${CKPT}" \
  --stage1_ckpt "${CKPT}" \
  --run_name "${RUN_ID}" \
  --output_json "${OUT_JSON}" \
  --local_data \
  --librispeech_root "${LIBRISPEECH_ROOT}" \
  --lexicon_path "${LEXICON_PATH}" \
  --spear_layernorm \
  --topk 256 \
  --n_routes 2 \
  --hard_gumbel_routing \
  --gumbel_tau_end 0.1 \
  --probe_steps 10000 \
  --probe_val_every 250 \
  --probe_patience 0 \
  --probe_warmup_steps 0 \
  --seed "${SEED}" \
  --num_workers "${NUM_WORKERS}" \
  --sources z_L \
  --tasks asr \
  --asr_probe_arch lstm \
  --asr_probe_lr 5e-4 \
  --asr_probe_warmup_steps 500 \
  --asr_probe_proj_dim 1024 \
  --asr_lstm_hidden 1024 \
  --asr_lstm_layers 2 \
  --asr_time_mask_param 50 \
  --asr_freq_mask_param 64 \
  --asr_probe_dropout 0.1 \
  --asr_max_examples 0

echo "[rerun] done"
