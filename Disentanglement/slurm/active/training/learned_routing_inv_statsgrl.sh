#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=24:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=lr_invrobust
#SBATCH --array=0-3%4
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/learned_routing_inv_statsgrl/%x_%A_%a.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/learned_routing_inv_statsgrl/%x_%A_%a.err

# Learned-routing comparison for the two strongest current objectives.
#
# Four array tasks:
#   0: invariance_only_w4 + soft binary L/P routing
#   1: invariance_only_w4 + hard binary L/P ST-Gumbel routing
#   2: robust-GRL gp02   + soft binary L/P routing
#   3: robust-GRL gp02   + hard binary L/P ST-Gumbel routing
#
# Each task trains stage2 from scratch, then probes the FINAL checkpoint only:
#   z_L -> SID with SID probe heads: linear, mlp, stats
#   z_P -> PR  with PR  probe heads: linear, mlp
# across two probe seeds.  We intentionally do not probe stage2_best.pt because
# previous GRL runs showed that the proxy-selected checkpoint can be misleading.

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

mkdir -p "${DIS_DIR}/logs/train/stage2/learned_routing_inv_statsgrl"
cd "${DIS_DIR}"

METHODS=(invariance_only_w4 invariance_only_w4 robustgrl_gp02 robustgrl_gp02)
ROUTINGS=(soft hard soft hard)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#METHODS[@]} )); then
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
    exit 2
fi

METHOD="${METHODS[${TASK_ID}]}"
ROUTING="${ROUTINGS[${TASK_ID}]}"
TRAIN_SEED="${TRAIN_SEED:-42}"
PROBE_SEEDS=(${PROBE_SEEDS:-42 7})
STAGE2_STEPS="${STAGE2_STEPS:-12000}"
PROBE_STEPS="${PROBE_STEPS:-10000}"
PROBE_PATIENCE="${PROBE_PATIENCE:-0}"

RUN_NAME="lr_${METHOD}_${ROUTING}_seed${TRAIN_SEED}"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
FINAL_CKPT="${CKPT_DIR}/stage2_step${STAGE2_STEPS}.pt"

ROUTING_ARGS=()
ROUTING_LABEL=""
case "${ROUTING}" in
    soft)
        ROUTING_ARGS=(--no-hard_gumbel_routing)
        ROUTING_LABEL="soft fractional routing; eval uses deterministic soft masks"
        ;;
    hard)
        ROUTING_ARGS=(--hard_gumbel_routing)
        ROUTING_LABEL="hard ST-Gumbel routing; eval uses deterministic argmax masks"
        ;;
    *)
        echo "ERROR: unknown routing=${ROUTING}" >&2
        exit 3
        ;;
esac

METHOD_ARGS=()
METHOD_LABEL=""
case "${METHOD}" in
    invariance_only_w4)
        METHOD_ARGS=(
            --invariance --inv_weight 4.0 --inv_ramp_end 0
            --alpha 0.8 --beta 0.6
            --grl_weight 0.0 --grl_phoneme_weight 0.5
            --grl_delay_steps 0
            --dann_full_discriminator --lr_disc 1e-3 --n_disc_steps 3
            --rho 0.0
        )
        METHOD_LABEL="invariance-only w4, no z_L speaker GRL, z_P phone GRL=0.5"
        ;;
    robustgrl_gp02)
        METHOD_ARGS=(
            --grl_robust_sid --grl_robust_activation gelu
            --grl_grad_norm --grl_grad_norm_target 0.001
            --alpha 0.8 --beta 0.6
            --grl_weight 1.0 --grl_phoneme_weight 0.15
            --grl_delay_steps 0
            --dann_full_discriminator --lr_disc 1e-3 --n_disc_steps 3
            --rho 0.0 --grad_clip 1.0
        )
        METHOD_LABEL="robust two-branch z_L GRL: signed linear mean + GELU mean/std, grad-norm target=0.001, z_P phone GRL=0.15"
        ;;
    *)
        echo "ERROR: unknown method=${METHOD}" >&2
        exit 4
        ;;
esac

echo "=== Learned-routing invariance/robust-GRL experiment ==="
echo "started          : $(date)"
echo "array_task       : ${TASK_ID}"
echo "gpu              : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "run_name         : ${RUN_NAME}"
echo "method           : ${METHOD}"
echo "method_label     : ${METHOD_LABEL}"
echo "routing          : ${ROUTING}"
echo "routing_label    : ${ROUTING_LABEL}"
echo "train_seed       : ${TRAIN_SEED}"
echo "probe_seeds      : ${PROBE_SEEDS[*]}"
echo "stage2_steps     : ${STAGE2_STEPS}"
echo "final_ckpt       : ${FINAL_CKPT}"
echo "probe_targets    : z_L->SID and z_P->PR"
echo "sid_probe_heads  : linear mlp stats"
echo "pr_probe_heads   : linear mlp"
echo "probe_steps      : ${PROBE_STEPS}"
echo "probe_patience   : ${PROBE_PATIENCE}"
echo "note             : no fixed_blocks; learned binary L/P routing; n_routes=2, no z_U"

${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch \
    "${ROUTING_ARGS[@]}" \
    "${METHOD_ARGS[@]}" \
    --n_routes 2 \
    --local_data --train_split_dir train-clean-100 --spear_layernorm \
    --num_workers 8 \
    --stage2_steps "${STAGE2_STEPS}" --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" \
    --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" \
    --seed "${TRAIN_SEED}"

[[ -f "${FINAL_CKPT}" ]] || {
    echo "ERROR: final checkpoint missing: ${FINAL_CKPT}" >&2
    echo "This script intentionally refuses to probe stage2_best.pt." >&2
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
done

echo
echo "finished         : $(date)"
