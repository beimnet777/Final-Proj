#!/bin/bash
# Submit one weaker dual-GRL stage-2 run, then probe it after training succeeds.
#
# Usage:
#   bash slurm/stage2_dual_grl_weak.sh
#
# Optional overrides:
#   GRL_P_WEIGHT=0.005 NAME=dual_grl_03_gp005 bash slurm/stage2_dual_grl_weak.sh
#   GRL_WEIGHT=0.02 GRL_P_WEIGHT=0.002 NAME=dual_grl_03_gs02_gp002 bash slurm/stage2_dual_grl_weak.sh

set -euo pipefail

DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
TRAIN_SCRIPT="${DIS_DIR}/slurm/stage2_run.sh"
PROBE_SCRIPT="${DIS_DIR}/slurm/probe_run.sh"

NAME="${NAME:-dual_grl_03_gp002}"
ALPHA="${ALPHA:-0.02}"
BETA="${BETA:-0.03}"
GRL_WEIGHT="${GRL_WEIGHT:-0.01}"
GRL_P_WEIGHT="${GRL_P_WEIGHT:-0.002}"
RHO="${RHO:-0.001}"
GRL_DELAY_STEPS="${GRL_DELAY_STEPS:-0}"

mkdir -p "${DIS_DIR}/logs/stage2/experiments"
mkdir -p "${DIS_DIR}/logs/probes"

CKPT="${DIS_DIR}/checkpoints/${NAME}/stage2_best.pt"
EXTRA_ARGS="--grl_phoneme_weight ${GRL_P_WEIGHT}"

echo "=== Submitting weaker dual-GRL experiment ==="
echo "name=${NAME}"
echo "alpha=${ALPHA} beta=${BETA} grl=${GRL_WEIGHT} grl_p=${GRL_P_WEIGHT} rho=${RHO} delay=${GRL_DELAY_STEPS}"

TRAIN_ID=$(sbatch --parsable \
    --job-name="s2_${NAME}" \
    --output="${DIS_DIR}/logs/stage2/experiments/${NAME}_%j.out" \
    --error="${DIS_DIR}/logs/stage2/experiments/${NAME}_%j.err" \
    --export="ALL,RUN_NAME=${NAME},ALPHA=${ALPHA},BETA=${BETA},GRL_WEIGHT=${GRL_WEIGHT},RHO=${RHO},GRL_DELAY_STEPS=${GRL_DELAY_STEPS},EXTRA_ARGS=${EXTRA_ARGS}" \
    "${TRAIN_SCRIPT}")

PROBE_ID=$(sbatch --parsable \
    --dependency=afterok:${TRAIN_ID} \
    --job-name="probe_${NAME}" \
    --output="${DIS_DIR}/logs/probes/probe_${NAME}_%j.out" \
    --error="${DIS_DIR}/logs/probes/probe_${NAME}_%j.err" \
    --export="ALL,RUN_NAME=probe_${NAME},STAGE2_CKPT=${CKPT}" \
    "${PROBE_SCRIPT}")

echo "train=${TRAIN_ID}"
echo "probe=${PROBE_ID} afterok:${TRAIN_ID}"
echo "Monitor: squeue -u ${USER}"
echo "Train log: ${DIS_DIR}/logs/stage2/experiments/${NAME}_${TRAIN_ID}.out"
echo "Probe log: ${DIS_DIR}/logs/probes/probe_${NAME}_${PROBE_ID}.out"
