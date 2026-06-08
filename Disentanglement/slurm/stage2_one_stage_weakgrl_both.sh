#!/bin/bash
# Submit two true one-stage/from-scratch weak-GRL runs, then probe each.
#
# Runs:
#   one_stage_weakgrl_x1  α=0.02 β=0.01 grl=0.01 ρ=0.001
#   one_stage_weakgrl_x2  α=0.04 β=0.02 grl=0.02 ρ=0.001
#
# Usage:
#   bash slurm/stage2_one_stage_weakgrl_both.sh

set -euo pipefail

DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
TRAIN_SCRIPT="${DIS_DIR}/slurm/stage2_run.sh"
PROBE_SCRIPT="${DIS_DIR}/slurm/probe_run.sh"

mkdir -p "${DIS_DIR}/logs/stage2/one_stage"
mkdir -p "${DIS_DIR}/logs/probes"

submit() {
    local name=$1 alpha=$2 beta=$3 grl=$4 rho=$5
    local ckpt="${DIS_DIR}/checkpoints/${name}/stage2_best.pt"

    local train_id
    train_id=$(sbatch --parsable \
        --job-name="s2_${name}" \
        --output="${DIS_DIR}/logs/stage2/one_stage/${name}_%j.out" \
        --error="${DIS_DIR}/logs/stage2/one_stage/${name}_%j.err" \
        --export="ALL,RUN_NAME=${name},ALPHA=${alpha},BETA=${beta},GRL_WEIGHT=${grl},RHO=${rho},GRL_DELAY_STEPS=0,EXTRA_ARGS=--stage2_from_scratch" \
        "${TRAIN_SCRIPT}")

    local probe_id
    probe_id=$(sbatch --parsable \
        --dependency=afterok:${train_id} \
        --job-name="probe_${name}" \
        --output="${DIS_DIR}/logs/probes/probe_${name}_%j.out" \
        --error="${DIS_DIR}/logs/probes/probe_${name}_%j.err" \
        --export="ALL,RUN_NAME=probe_${name},STAGE2_CKPT=${ckpt}" \
        "${PROBE_SCRIPT}")

    printf "  %-24s train=%-10s probe=%-10s α=%-5s β=%-5s grl=%-5s\n" \
        "${name}" "${train_id}" "${probe_id}" "${alpha}" "${beta}" "${grl}"
}

echo "=== Submitting true one-stage weak-GRL runs ==="
submit one_stage_weakgrl_x1 0.02 0.01 0.01 0.001
submit one_stage_weakgrl_x2 0.04 0.02 0.02 0.001

echo ""
echo "Monitor: squeue -u ${USER}"
echo "Train logs: ls -lt ${DIS_DIR}/logs/stage2/one_stage/"
echo "Probe logs: ls -lt ${DIS_DIR}/logs/probes/probe_one_stage_weakgrl_*.out"
