#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=30:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=grlgelupem
#SBATCH --array=0-1%2
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/grl_gelu_stats_prosody_emotion_8to1/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/grl_gelu_stats_prosody_emotion_8to1/%x_%A_%a.err

# Non-invariance PEM variant.
#
# Same binary learned L/P routing and Libri:IEMOCAP cadence as the invariance PEM
# run, but the pitch/formant perturbed-pair invariance loss is OFF.  Speaker
# removal from z_L comes from a robust speaker-GRL instead:
#   branch A: signed linear masked mean
#   branch B: GELU masked mean+std stats pooling
#
# No fixed blocks and no z_U: --n_routes 2 only.  This isolates whether the
# reconstruction instability came from invariance while keeping the prosody,
# emotion, and phoneme-leakage controls active.

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

mkdir -p "${DIS_DIR}/logs/train/stage2/grl_gelu_stats_prosody_emotion_8to1"
cd "${DIS_DIR}"

ROUTINGS=(soft hard)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#ROUTINGS[@]} )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
    exit 2
fi

ROUTING="${ROUTINGS[${TASK_ID}]}"
TRAIN_SEED="${TRAIN_SEED:-42}"
PROBE_SEEDS=(${PROBE_SEEDS:-42 7})
STAGE2_STEPS="${STAGE2_STEPS:-12000}"
PROBE_STEPS="${PROBE_STEPS:-10000}"
PROSODY_PROBE_STEPS="${PROSODY_PROBE_STEPS:-4000}"
EMOTION_PROBE_STEPS="${EMOTION_PROBE_STEPS:-4000}"
PROBE_PATIENCE="${PROBE_PATIENCE:-0}"
EMOTION_EVERY="${EMOTION_EVERY:-8}"
IEMOCAP_ROOT="${IEMOCAP_ROOT:-/rds/project/rds-xyBFuSj0hm0/dataset/IEMOCAP_full_release}"
IEMOCAP_FOLD="${IEMOCAP_FOLD:-5}"
IEMOCAP_BATCH_SIZE="${IEMOCAP_BATCH_SIZE:-8}"

if [[ ! -d "${IEMOCAP_ROOT}" ]]; then
    echo "ERROR: IEMOCAP_ROOT not found: ${IEMOCAP_ROOT}" >&2
    echo "Set IEMOCAP_ROOT to the extracted IEMOCAP_full_release directory." >&2
    exit 3
fi

ROUTING_ARGS=()
case "${ROUTING}" in
    soft)
        ROUTING_ARGS=(--no-hard_gumbel_routing --gumbel_tau_start 1.0 --gumbel_tau_end 0.5)
        ;;
    hard)
        ROUTING_ARGS=(--hard_gumbel_routing --gumbel_tau_start 1.0 --gumbel_tau_end 0.1)
        ;;
    *)
        echo "ERROR: unknown routing=${ROUTING}" >&2
        exit 4
        ;;
esac

RUN_NAME="grl_gelu_stats_prosody_emotion_8to1_${ROUTING}_seed${TRAIN_SEED}"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
FINAL_CKPT="${CKPT_DIR}/stage2_step${STAGE2_STEPS}.pt"

echo "=== GRL-GELU-stats + prosody + emotion, binary learned routing ==="
echo "started          : $(date)"
echo "array_task       : ${TASK_ID}"
echo "gpu              : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "run_name         : ${RUN_NAME}"
echo "routing          : ${ROUTING} (${ROUTING_ARGS[*]})"
echo "train_seed       : ${TRAIN_SEED}"
echo "stage2_steps     : ${STAGE2_STEPS}"
echo "ratio            : Libri every step; IEMOCAP every ${EMOTION_EVERY} steps"
echo "batch ratio      : Libri batch=16, IEMOCAP batch=${IEMOCAP_BATCH_SIZE} (default utterance ratio ~16:1)"
echo "iemocap_root     : ${IEMOCAP_ROOT}"
echo "iemocap_fold     : ${IEMOCAP_FOLD}"
echo "no_z_U           : --n_routes 2, no z_U adversaries"
echo "invariance       : OFF"
echo "z_L removal      : robust speaker-GRL, linear mean + GELU mean/std stats, grl_weight=1.0"
echo "speaker grl_norm : target=0.0005"
echo "z_P tasks        : SID + prosody + emotion, with phoneme-GRL"
echo "phoneme grl_norm : target=0.0005"
echo "final_ckpt       : ${FINAL_CKPT}"

${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch \
    "${ROUTING_ARGS[@]}" \
    --n_routes 2 \
    --routing_init_std 0.5 --routing_spec_weight 0.01 --lr_routing 1e-3 \
    --local_data --train_split_dir train-clean-100 --spear_layernorm \
    --num_workers 8 \
    --grl_robust_sid --grl_robust_activation gelu \
    --grl_grad_norm --grl_grad_norm_target 0.0005 \
    --alpha 0.8 --beta 0.6 \
    --grl_weight 1.0 --grl_phoneme_weight 0.5 \
    --grl_p_grad_norm --grl_p_grad_norm_target 0.0005 \
    --prosody --prosody_weight 0.5 --grl_prosody_weight 0.5 --grl_prosody_u_weight 0.0 \
    --emotion --emotion_weight 0.5 --grl_emotion_weight 0.2 \
    --emotion_every "${EMOTION_EVERY}" --emotion_grl_ramp_end 2000 --emotion_aux_loss_clip 5.0 \
    --iemocap_root "${IEMOCAP_ROOT}" --iemocap_fold "${IEMOCAP_FOLD}" \
    --iemocap_batch_size "${IEMOCAP_BATCH_SIZE}" --iemocap_eval_batch_size 16 \
    --grl_delay_steps 0 --dann_full_discriminator --lr_disc 1e-3 \
    --n_disc_steps 3 --rho 0.0 --grad_clip 1.0 \
    --stage2_steps "${STAGE2_STEPS}" --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" \
    --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" \
    --seed "${TRAIN_SEED}"

[[ -f "${FINAL_CKPT}" ]] || {
    echo "ERROR: final checkpoint missing: ${FINAL_CKPT}" >&2
    echo "This script intentionally probes the final checkpoint, not stage2_best.pt." >&2
    exit 5
}

echo
echo "=== Targeted diagnostic probes on FINAL checkpoint ==="
echo "probe_started    : $(date)"
echo "stage1_ckpt      : ${FINAL_CKPT}"
echo "stage2_ckpt      : ${FINAL_CKPT}"

for PROBE_SEED in "${PROBE_SEEDS[@]}"; do
    for SID_HEAD in linear mlp stats; do
        echo
        echo "--- probe: z_L -> SID | sid_head=${SID_HEAD} | seed=${PROBE_SEED} ---"
        ${PYTHON} -u diag_probe/run.py \
            --stage2_ckpt "${FINAL_CKPT}" \
            --stage1_ckpt "${FINAL_CKPT}" \
            --run_name "diag_${RUN_NAME}_final_zL_sid_${SID_HEAD}_seed${PROBE_SEED}" \
            "${ROUTING_ARGS[@]}" --n_routes 2 --spear_layernorm \
            --sources "z_L" --tasks "sid" --sid_probe_arch "${SID_HEAD}" \
            --probe_steps "${PROBE_STEPS}" --probe_val_every 250 \
            --probe_patience "${PROBE_PATIENCE}" \
            --pr_max_examples 0 --sid_probe_lr 1e-3 \
            --probe_warmup_steps 0 --seed "${PROBE_SEED}"
    done

    for PR_HEAD in linear mlp; do
        echo
        echo "--- probe: z_P -> PR | pr_head=${PR_HEAD} | seed=${PROBE_SEED} ---"
        ${PYTHON} -u diag_probe/run.py \
            --stage2_ckpt "${FINAL_CKPT}" \
            --stage1_ckpt "${FINAL_CKPT}" \
            --run_name "diag_${RUN_NAME}_final_zP_pr_${PR_HEAD}_seed${PROBE_SEED}" \
            "${ROUTING_ARGS[@]}" --n_routes 2 --spear_layernorm \
            --sources "z_P" --tasks "pr" --pr_probe_arch "${PR_HEAD}" \
            --probe_steps "${PROBE_STEPS}" --probe_val_every 250 \
            --probe_patience "${PROBE_PATIENCE}" \
            --pr_max_examples 0 --pr_probe_lr 5e-4 \
            --probe_warmup_steps 0 --seed "${PROBE_SEED}"
    done

    echo
    echo "--- probe: z_L,z_P -> prosody | seed=${PROBE_SEED} ---"
    ${PYTHON} -u diag_probe/run.py \
        --stage2_ckpt "${FINAL_CKPT}" \
        --stage1_ckpt "${FINAL_CKPT}" \
        --run_name "diag_${RUN_NAME}_final_prosody_seed${PROBE_SEED}" \
        "${ROUTING_ARGS[@]}" --n_routes 2 --spear_layernorm \
        --sources "z_L,z_P" --tasks "prosody" --prosody \
        --probe_steps "${PROSODY_PROBE_STEPS}" --probe_val_every 250 \
        --probe_patience "${PROBE_PATIENCE}" \
        --prosody_probe_lr 5e-4 --prosody_max_train 2000 \
        --probe_warmup_steps 0 --seed "${PROBE_SEED}"

    echo
    echo "--- probe: z_L,z_P -> emotion | seed=${PROBE_SEED} ---"
    ${PYTHON} -u diag_probe/run.py \
        --stage2_ckpt "${FINAL_CKPT}" \
        --stage1_ckpt "${FINAL_CKPT}" \
        --run_name "diag_${RUN_NAME}_final_emotion_seed${PROBE_SEED}" \
        "${ROUTING_ARGS[@]}" --n_routes 2 --spear_layernorm \
        --sources "z_L,z_P" --tasks "emotion" --emotion \
        --iemocap_root "${IEMOCAP_ROOT}" --iemocap_fold "${IEMOCAP_FOLD}" \
        --iemocap_batch_size "${IEMOCAP_BATCH_SIZE}" --iemocap_eval_batch_size 16 \
        --probe_steps "${EMOTION_PROBE_STEPS}" --probe_val_every 250 \
        --probe_patience "${PROBE_PATIENCE}" \
        --emotion_probe_lr 5e-4 --probe_warmup_steps 0 --seed "${PROBE_SEED}"
done

echo
echo "finished         : $(date)"
