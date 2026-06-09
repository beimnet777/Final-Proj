#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#
# Generic stage-2 worker — called by stage2_sweep.sh / stage2_abl.sh.
# Parameters injected via --export: RUN_NAME ALPHA BETA GRL_WEIGHT RHO GRL_DELAY_STEPS
# Optional: EXTRA_ARGS (e.g. "--no_routing" or "--n_routes 2")

. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1

mkdir -p "${DIS_DIR}/logs"
cd "${DIS_DIR}"

echo "=== Stage 2: ${RUN_NAME} ==="
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started    : $(date)"
echo "α=${ALPHA}  β=${BETA}  grl=${GRL_WEIGHT}  ρ=${RHO}  grl_delay=${GRL_DELAY_STEPS}"

${PYTHON} -u run.py \
    --stage             2                                              \
    --stage1_ckpt       "${DIS_DIR}/checkpoints/best.pt"              \
    --stage2_steps      8000                                           \
    --warmup_steps      500                                            \
    --alpha             "${ALPHA}"                                     \
    --beta              "${BETA}"                                      \
    --grl_weight        "${GRL_WEIGHT}"                                \
    --grl_delay_steps   "${GRL_DELAY_STEPS}"                          \
    --rho               "${RHO}"                                       \
    --lr                3e-5                                           \
    --lr_min            1e-6                                           \
    --lr_routing        5e-6                                           \
    --lr_heads          1e-4                                           \
    --max_train_examples 0                                             \
    --max_val_examples   500                                           \
    --grad_log_every    500                                            \
    --checkpoint_dir    "${DIS_DIR}/checkpoints/${RUN_NAME}"           \
    --runs_dir          "${DIS_DIR}/runs/${RUN_NAME}"                  \
    --log_dir           "${DIS_DIR}/logs"                              \
    ${EXTRA_ARGS:-}

echo "Finished : $(date)"
