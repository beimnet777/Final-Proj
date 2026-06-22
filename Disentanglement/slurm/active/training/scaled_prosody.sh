#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=scaled_prosody
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/scaled_prosody/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/scaled_prosody/%x_%j.err

# Scaled, EMERGENT activation (fixed-block MEMBERSHIP + global top-k, NO per-block
# budget) — identical to scaled_3partition case 0 (preact_fixed), which STARVED
# z_P to ~3 active units (pooled SID gives no per-frame select signal).  The ONLY
# change: add the PROSODY task on z_P (per-frame log-F0 + log-energy regression).
# Hypothesis: prosody is a PER-FRAME task → gives z_P a frame-level reason to win
# the emergent top-k, so the paralinguistic block survives WITHOUT a forced budget.
# Anti-prosody adversaries on z_L/z_U push F0/energy into z_P.  Clean A/B vs the
# starved baseline: does sid lift off chance and does actP rise above ~3?

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
mkdir -p "${DIS_DIR}/logs/train/stage2/scaled_prosody"
cd "${DIS_DIR}"

SEED="${SEED:-42}"
STAGE2_STEPS="${STAGE2_STEPS:-26000}"   # 4 passes/utt on 360h
PROBE_STEPS="${PROBE_STEPS:-8000}"
RUN_NAME="scaled_prosody"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
STAGE2_CKPT="${CKPT_DIR}/stage2_best.pt"
# Fixed MEMBERSHIP, equal blocks, GLOBAL top-k (emergent allocation) — matches
# scaled_3partition case 0 so the only new variable is prosody.
BLOCKS=(--fixed_blocks --no-per_block_topk --K_L 5462 --K_P 5461 --K_U 5461)

echo "=== scaled + prosody (emergent top-k, no forced budget) ===  $(date)"
echo "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "blocks: ${BLOCKS[*]}"

# ----------------------------- Unified training -----------------------------
${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch \
    --local_data --train_split_dir train-clean-360 \
    --spear_layernorm \
    --K 16384 --topk 64 \
    "${BLOCKS[@]}" --rho 0.0 \
    --aux_k 512 --aux_k_coef 0.03125 --dead_steps_threshold 256 \
    --geom_median_bias --renorm_decoder \
    --alpha 0.8 --beta 0.6 \
    --grl_weight 0.7 --grl_phoneme_weight 0.7 \
    --grl_u_weight 0.5 --grl_phoneme_u_weight 0.5 \
    --prosody --prosody_weight 0.5 \
    --grl_prosody_weight 0.5 --grl_prosody_u_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator \
    --lr_disc 1e-3 --n_disc_steps 3 \
    --stage2_steps "${STAGE2_STEPS}" --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" --seed "${SEED}"

[[ -f "${STAGE2_CKPT}" ]] || { echo "ERROR: checkpoint missing" >&2; exit 3; }

# ----------------------------- Probe (incl. prosody on every bucket) -----------------------------
echo; echo "----- probe -----"; date
${PYTHON} -u diag_probe/run.py \
    --stage2_ckpt "${STAGE2_CKPT}" --stage1_ckpt "${STAGE2_CKPT}" \
    --run_name "diag_probe_${RUN_NAME}" \
    "${BLOCKS[@]}" \
    --topk 64 --spear_layernorm \
    --sources "z_t,z_L,z_P,z_U" --tasks "pr,sid" --prosody --sid_probe_arch stats \
    --probe_steps "${PROBE_STEPS}" --probe_val_every 250 --probe_patience 6 \
    --pr_max_examples 0 --pr_probe_lr 5e-4 --sid_probe_lr 1e-3 --probe_warmup_steps 0 \
    --seed "${SEED}"

echo; echo "Finished: $(date)"
