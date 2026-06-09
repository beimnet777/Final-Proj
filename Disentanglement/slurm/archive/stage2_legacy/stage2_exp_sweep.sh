#!/bin/bash
# stage2_exp_sweep.sh — submit stage-2 experiments 1, 4, 5 (no new stage 1 needed).
#
# Usage:
#   bash slurm/stage2_exp_sweep.sh
#
# Runs (all use existing best.pt stage-1 checkpoint):
#   dual_grl_03   Exp 1: β=0.03  grl_s=0.01  grl_p=0.01
#   dual_grl_04   Exp 1: β=0.04  grl_s=0.01  grl_p=0.01
#   ub            Exp 4: β=0.01  ub_weight=0.01
#   ste           Exp 5: β=0.01  ste_routing
#   ste_ub        Exp 4+5: β=0.01  ste + ub=0.01
#   combined      Exp 1+4+5: β=0.03  dual_grl + ste + ub=0.01
#
# Each run gets an automatic post-hoc probe queued after training.

DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
TRAIN_SCRIPT="${DIS_DIR}/slurm/stage2_run.sh"
PROBE_SCRIPT="${DIS_DIR}/slurm/probe_run.sh"

mkdir -p "${DIS_DIR}/logs/stage2/experiments"
mkdir -p "${DIS_DIR}/logs/probes"

submit() {
    local name=$1 alpha=$2 beta=$3 grl=$4 rho=$5 delay=$6 extra=${7:-""}
    local ckpt="${DIS_DIR}/checkpoints/${name}/stage2_best.pt"

    local train_id
    train_id=$(sbatch --parsable \
        --job-name="s2_${name}" \
        --output="${DIS_DIR}/logs/stage2/experiments/${name}_%j.out" \
        --error="${DIS_DIR}/logs/stage2/experiments/${name}_%j.err" \
        --export="ALL,RUN_NAME=${name},ALPHA=${alpha},BETA=${beta},GRL_WEIGHT=${grl},RHO=${rho},GRL_DELAY_STEPS=${delay},EXTRA_ARGS=${extra}" \
        "${TRAIN_SCRIPT}")

    local probe_id
    probe_id=$(sbatch --parsable \
        --dependency=afterok:${train_id} \
        --job-name="probe_${name}" \
        --output="${DIS_DIR}/logs/probes/probe_${name}_%j.out" \
        --error="${DIS_DIR}/logs/probes/probe_${name}_%j.err" \
        --export="ALL,RUN_NAME=probe_${name},STAGE2_CKPT=${ckpt}" \
        "${PROBE_SCRIPT}")

    printf "  %-18s  train=%-10s  probe=%-10s\n" "${name}" "${train_id}" "${probe_id}"
}

echo "=== Submitting stage-2 experiments (Exp 1 / 4 / 5) ==="
#          name            α     β     grl   ρ      delay  extra_args
submit  dual_grl_03    0.02  0.03  0.01  0.001  0  "--grl_phoneme_weight 0.01"
submit  dual_grl_04    0.02  0.04  0.01  0.001  0  "--grl_phoneme_weight 0.01"
submit  ub             0.02  0.01  0.01  0.001  0  "--ub_weight 0.01"
submit  ste            0.02  0.01  0.01  0.001  0  "--ste_routing"
submit  ste_ub         0.02  0.01  0.01  0.001  0  "--ste_routing --ub_weight 0.01"
submit  combined       0.02  0.03  0.01  0.001  0  "--grl_phoneme_weight 0.01 --ste_routing --ub_weight 0.01"
echo ""
echo "Monitor:  squeue -u ${USER}"
echo "Logs:     ls -lt ${DIS_DIR}/logs/stage2/experiments/"
