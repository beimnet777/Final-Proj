#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=probe_zU_dense
#SBATCH --array=0-1
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/jobs_zU_probe/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/jobs_zU_probe/%x_%A_%a.err

# Probe z_U (the un-probed residual) for the two z_L-success runs:
#   0  dense_gradnorm  (dense-context + grad-norm speaker GRL)
#   1  inv_dense       (invariance + dense-context speaker GRL)
# Probes z_L,z_P,z_U for PR + SID (stats probe) — re-confirms z_L=0.010 under the
# honest probe AND reveals where the removed speaker went (z_U is the suspect sink).
# Fixed-block carve must match training: per-block topk 160/64/32, spear_layernorm.

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
mkdir -p "${DIS_DIR}/logs/diag/jobs_zU_probe"
cd "${DIS_DIR}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) RUN_NAME=job2_dense_gradnorm ;;
  1) RUN_NAME=job1_inv_dense ;;
  *) echo "bad array index" >&2; exit 1 ;;
esac
CKPT="${DIS_DIR}/checkpoints/${RUN_NAME}/stage2_best.pt"
[[ -f "${CKPT}" ]] || { echo "ERROR: missing ${CKPT}" >&2; exit 2; }

BLOCKS=(--fixed_blocks --per_block_topk --K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== Probe z_U: ${RUN_NAME} (z_L,z_P,z_U : PR + SID-stats) ===  $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader

${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${CKPT}" --stage1_ckpt "${CKPT}" \
    --run_name "diag_zU_${RUN_NAME}" "${BLOCKS[@]}" --spear_layernorm \
    --sources "z_U" --tasks "pr,sid" --sid_probe_arch stats \
    --probe_steps 10000 --probe_val_every 250 --probe_patience 8 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 --seed 42

echo "Finished probe: $(date)"
