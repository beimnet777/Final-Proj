#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=24:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=inv_dense_robust
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/inv_dense_robust_stronger/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/inv_dense_robust_stronger/%x_%j.err

# Stronger inv_dense variant:
#   - fixed L/P/U blocks, stage-2 from scratch
#   - slightly stronger pitch/formant invariance on z_L
#   - robust branched z_L speaker GRL:
#       linear mean branch + GELU stats branch + dense-context frame branch
#   - z_P phone GRL kept on, to control phone leakage into the paralinguistic bucket
#   - probe final checkpoint only, not stage2_best.pt

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null || true
module load rhel8/default-amp 2>/dev/null || true

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"

RUN_NAME="inv_dense_robust_stronger"
STAGE2_STEPS="${STAGE2_STEPS:-12000}"
PROBE_STEPS="${PROBE_STEPS:-10000}"
PROBE_VAL_EVERY="${PROBE_VAL_EVERY:-250}"
PROBE_SEEDS_STR="${PROBE_SEEDS:-42 7}"

TRAIN_LOG_DIR="${DIS_DIR}/logs/train/stage2/${RUN_NAME}"
DIAG_LOG_DIR="${DIS_DIR}/logs/diag/${RUN_NAME}"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
RUNS_DIR="${DIS_DIR}/runs/${RUN_NAME}"
mkdir -p "${TRAIN_LOG_DIR}" "${DIAG_LOG_DIR}" "${CKPT_DIR}" "${RUNS_DIR}"

cd "${DIS_DIR}"

BLOCKS=(
    --fixed_blocks --per_block_topk
    --K_L 3072 --K_P 1024 --K_U 1024
    --topk_L 160 --topk_P 64 --topk_U 32
)

echo "=== inv_dense_robust_stronger: stronger invariance + branched dense GRL ==="
echo "started          : $(date)"
echo "node             : ${SLURMD_NODENAME:-unknown}"
echo "gpu              : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "run_name         : ${RUN_NAME}"
echo "stage2_steps     : ${STAGE2_STEPS}"
echo "invariance       : w=4.0, no ramp, f0=[0.6,1.7], formant=[0.75,1.55]"
echo "z_L grl          : robust branches = linear mean + GELU stats + dense context"
echo "z_P grl          : PR-GRL weight=0.5"
echo "probe seeds      : ${PROBE_SEEDS_STR}"
echo "probe final ckpt : stage2_step${STAGE2_STEPS}.pt"
echo

"${PYTHON}" -u run.py \
    --stage 2 --stage2_from_scratch "${BLOCKS[@]}" \
    --invariance --inv_weight 4.0 --inv_ramp_end 0 \
    --inv_f0_low 0.6 --inv_f0_high 1.7 \
    --inv_formant_low 0.75 --inv_formant_high 1.55 \
    --grl_robust_sid --grl_robust_activation gelu \
    --grl_dense_context --grl_context_kernel 31 \
    --local_data --train_split_dir train-clean-100 --spear_layernorm --num_workers 12 \
    --alpha 0.8 --beta 0.6 --grl_weight 2.0 --grl_phoneme_weight 0.5 \
    --grl_delay_steps 0 --dann_full_discriminator --lr_disc 1e-3 --n_disc_steps 3 --rho 0.0 \
    --stage2_steps "${STAGE2_STEPS}" --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" --runs_dir "${RUNS_DIR}" \
    --log_dir "${DIS_DIR}/logs" --seed 42

FINAL_CKPT="${CKPT_DIR}/stage2_step${STAGE2_STEPS}.pt"
if [[ ! -f "${FINAL_CKPT}" ]]; then
    echo "ERROR: final checkpoint missing: ${FINAL_CKPT}" >&2
    exit 3
fi

echo
echo "----- diagnostic probes from FINAL checkpoint only -----"
echo "final_ckpt        : ${FINAL_CKPT}"
echo "reported metrics  : TEST, with val shown in parentheses inside diag_probe"
echo

for PROBE_SEED in ${PROBE_SEEDS_STR}; do
    for SID_HEAD in linear mlp stats; do
        echo
        echo ">>> probe z_L -> SID | head=${SID_HEAD} | seed=${PROBE_SEED}"
        "${PYTHON}" -u diag_probe/run.py \
            --stage2_ckpt "${FINAL_CKPT}" --stage1_ckpt "${FINAL_CKPT}" \
            --run_name "diag_${RUN_NAME}_final_zL_sid_${SID_HEAD}_seed${PROBE_SEED}" \
            "${BLOCKS[@]}" --spear_layernorm \
            --sources "z_L" --tasks "sid" --sid_probe_arch "${SID_HEAD}" \
            --probe_steps "${PROBE_STEPS}" --probe_val_every "${PROBE_VAL_EVERY}" --probe_patience 0 \
            --sid_probe_lr 1e-3 --seed "${PROBE_SEED}" \
            2>&1 | tee "${DIAG_LOG_DIR}/zL_sid_${SID_HEAD}_seed${PROBE_SEED}_${SLURM_JOB_ID:-local}.out"
    done

    for PR_HEAD in linear mlp; do
        echo
        echo ">>> probe z_P -> PR | head=${PR_HEAD} | seed=${PROBE_SEED}"
        "${PYTHON}" -u diag_probe/run.py \
            --stage2_ckpt "${FINAL_CKPT}" --stage1_ckpt "${FINAL_CKPT}" \
            --run_name "diag_${RUN_NAME}_final_zP_pr_${PR_HEAD}_seed${PROBE_SEED}" \
            "${BLOCKS[@]}" --spear_layernorm \
            --sources "z_P" --tasks "pr" --pr_probe_arch "${PR_HEAD}" \
            --probe_steps "${PROBE_STEPS}" --probe_val_every "${PROBE_VAL_EVERY}" --probe_patience 0 \
            --pr_max_examples 0 --pr_probe_lr 5e-4 --seed "${PROBE_SEED}" \
            2>&1 | tee "${DIAG_LOG_DIR}/zP_pr_${PR_HEAD}_seed${PROBE_SEED}_${SLURM_JOB_ID:-local}.out"
    done
done

echo
echo "finished         : $(date)"
