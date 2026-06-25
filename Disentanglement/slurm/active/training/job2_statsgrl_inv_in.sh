#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=72G
#SBATCH --time=16:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=statsgrl_inv_in
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/statsgrl_inv_in/%x_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/train/stage2/statsgrl_inv_in/%x_%j.err

# Job2 z_L leakage fix:
#   Keep the successful fixed-block Job2 recipe that made z_P strong:
#     - z_P positive SID task
#     - weak phoneme adversary on z_P
#     - stats-pool speaker adversary on z_L with gradient-normalized reversal
#   Add two structural z_L constraints:
#     - instance_norm_zL removes utterance-level mean/scale speaker statistics;
#     - perturbation invariance gives z_L a positive dense signal to ignore
#       speaker-like pitch/formant changes.
#
# This is not "more GRL"; it tests whether z_L needs structural invariance,
# while z_P keeps the previously successful setup.

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

mkdir -p "${DIS_DIR}/logs/train/stage2/statsgrl_inv_in"
cd "${DIS_DIR}"

RUN_NAME="job2_statsgrl_inv_in_w4_r3000"
CKPT_DIR="${DIS_DIR}/checkpoints/${RUN_NAME}"
BLOCKS=(--fixed_blocks --per_block_topk
        --K_L 3072 --K_P 1024 --K_U 1024
        --topk_L 160 --topk_P 64 --topk_U 32)

echo "=== ${RUN_NAME}: Job2 + stats-GRL + z_L instance-norm + ramped invariance ==="
echo "started       : $(date)"
echo "gpu           : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "purpose       : fix z_L speaker leakage without changing the successful z_P recipe"
echo "fixed blocks  : L/P/U = 3072/1024/1024 ; active = 160/64/32"
echo "z_L cleanup   : instance_norm_zL + invariance"
echo "inv_weight    : 4.0"
echo "inv_ramp_end  : 3000"
echo "grl_head      : stats-pool SID adversary on z_L"
echo "grl_grad_norm : target=0.001"
echo "grl_p_weight  : 0.2"
echo "grad_clip     : 1.0"

${PYTHON} -u run.py \
    --stage 2 --stage2_from_scratch "${BLOCKS[@]}" \
    --instance_norm_zL \
    --invariance --inv_weight 4.0 --inv_ramp_end 3000 \
    --grl_stats_pool \
    --grl_grad_norm --grl_grad_norm_target 0.001 \
    --local_data --train_split_dir train-clean-100 --spear_layernorm \
    --num_workers 12 \
    --alpha 0.8 --beta 0.6 --grl_weight 1.0 --grl_phoneme_weight 0.2 \
    --grl_delay_steps 0 --dann_full_discriminator --lr_disc 1e-3 \
    --n_disc_steps 3 --rho 0.0 --grad_clip 1.0 \
    --stage2_steps 12000 --warmup_steps 500 \
    --lr 1e-4 --lr_min 1e-6 --lr_heads 1e-4 --grad_log_every 500 \
    --checkpoint_dir "${CKPT_DIR}" \
    --runs_dir "${DIS_DIR}/runs/${RUN_NAME}" \
    --log_dir "${DIS_DIR}/logs" \
    --seed 42

[[ -f "${CKPT_DIR}/stage2_best.pt" ]] || {
    echo "ERROR: training finished but ${CKPT_DIR}/stage2_best.pt is missing" >&2
    exit 3
}

echo
echo "Training finished. Submit the matching diagnostic probe with:"
echo "  sbatch slurm/active/probing/probe_job2_statsgrl_inv_in.sh"
echo
echo "Finished: $(date)"
