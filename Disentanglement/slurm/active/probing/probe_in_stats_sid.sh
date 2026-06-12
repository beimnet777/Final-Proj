#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=in_stats_sid
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/probes/projection/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/probes/projection/%x_%j.err

# Decisive validation of the instance-norm result.  The plain SID probe
# (linear -> mean-pool) is structurally blind to IN'd z_L (IN zeroes the
# per-utterance mean a linear projector commutes with), so its z_L->SID=0.002
# proves only first-order removal.  This re-probe uses the pooling-robust
# stats probe (projector -> ReLU -> mean+std pool) on the SAME checkpoint:
#   z_L -> SID stays near chance  => IN result is real (higher-order too)
#   z_L -> SID bounces back up    => IN removed only first-order stats
# z_t / z_P are in-run controls (must stay ~0.99: the new probe is not weaker).

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

mkdir -p "${DIS_DIR}/logs/probes/projection"
cd "${DIS_DIR}"

MODEL_NAME="${MODEL_NAME:-proj_recon_ln_d256_grl0_in}"
STAGE1_CKPT="${STAGE1_CKPT:-${DIS_DIR}/checkpoints/ln_sae/stage1_best.pt}"
STAGE2_CKPT="${STAGE2_CKPT:-${DIS_DIR}/checkpoints/${MODEL_NAME}/stage2_best.pt}"
RUN_NAME="${RUN_NAME:-diag_probe_${MODEL_NAME}_stats_sid}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
SEED="${SEED:-42}"

echo "=== Pooling-robust (stats) SID re-probe of the IN checkpoint ==="
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Node / GPU        : $(hostname) / $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started           : $(date)"
echo "model             : ${MODEL_NAME}"
echo "stage2_ckpt       : ${STAGE2_CKPT}"
echo "sid_probe_arch    : stats (projector -> ReLU -> mean+std pool)"

if [[ ! -f "${STAGE1_CKPT}" ]]; then echo "ERROR: missing stage1 ckpt: ${STAGE1_CKPT}" >&2; exit 2; fi
if [[ ! -f "${STAGE2_CKPT}" ]]; then echo "ERROR: missing stage2 ckpt: ${STAGE2_CKPT}" >&2; exit 3; fi

${PYTHON} -u diag_probe/run.py \
    --stage1_ckpt        "${STAGE1_CKPT}" \
    --stage2_ckpt        "${STAGE2_CKPT}" \
    --run_name           "${RUN_NAME}" \
    --spear_layernorm \
    --instance_norm_zL \
    --sid_probe_arch     stats \
    --pr_label_set       dis \
    --sources            "z_t,z_L,z_P" \
    --tasks              "sid" \
    --probe_steps        "${PROBE_STEPS}" \
    --seed               "${SEED}" \
    --sid_probe_lr       1e-3 \
    --probe_warmup_steps 0

echo "Finished          : $(date)"
