#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=reprobe
#SBATCH --array=0-1%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/reprobe_tests/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/reprobe_tests/%x_%A_%a.err

# Two MEASUREMENT tests on existing checkpoints (no retraining):
#   0  exp2: z_L->SID with the IN-robust STATS probe (and LINEAR for contrast) —
#            confirm whether the 0.002 is real removal or a linear-probe artifact.
#   1  exp3: probe z_U on BOTH tasks — is the 175-unit residual genuine junk
#            (low PR + low SID) or an entangled blob (high on both)?

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
mkdir -p "${DIS_DIR}/logs/train/stage2/reprobe_tests"
cd "${DIS_DIR}"

CK_EXP2="${DIS_DIR}/checkpoints/fxblk_exp2_perblock_IN/stage2_best.pt"
CK_EXP3="${DIS_DIR}/checkpoints/fxblk_exp3_global_equal/stage2_best.pt"
COMMON=(--probe_steps 4000 --probe_val_every 250 --probe_patience 5 --pr_max_examples 0 \
        --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 --probe_warmup_steps 0 --spear_layernorm)

i="${SLURM_ARRAY_TASK_ID:-0}"
echo "=== reprobe test ${i} ===  $(date)  GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"

if [[ "$i" == "0" ]]; then
  # ---- Test 1: exp2 z_L->SID, robust (stats) vs blinded (linear) ----
  for ARCH in stats linear; do
    echo; echo "##### exp2 z_L->SID  arch=${ARCH} #####"
    ${PYTHON} -u diag_probe/run.py \
        --stage2_ckpt "${CK_EXP2}" --stage1_ckpt "${CK_EXP2}" \
        --run_name "reprobe_exp2_zL_sid_${ARCH}" \
        --fixed_blocks --per_block_topk \
        --K_L 3072 --K_P 1024 --K_U 1024 --topk_L 160 --topk_P 64 --topk_U 32 \
        --instance_norm_zL \
        --sources "z_L" --tasks "sid" --sid_probe_arch "${ARCH}" \
        "${COMMON[@]}"
  done
else
  # ---- Test 2: exp3 probe z_U (+ z_t/z_L/z_P) on both tasks ----
  echo; echo "##### exp3 probe z_t,z_L,z_P,z_U  (pr,sid) #####"
  ${PYTHON} -u diag_probe/run.py \
      --stage2_ckpt "${CK_EXP3}" --stage1_ckpt "${CK_EXP3}" \
      --run_name "reprobe_exp3_zU" \
      --fixed_blocks --no-per_block_topk \
      --K_L 1706 --K_P 1707 --K_U 1707 \
      --sources "z_t,z_L,z_P,z_U" --tasks "pr,sid" --sid_probe_arch stats \
      "${COMMON[@]}"
fi
echo; echo "Finished ${i}: $(date)"
