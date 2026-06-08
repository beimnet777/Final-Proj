#!/bin/bash
# stage2_abl.sh — submit ablation training jobs D, E, F in parallel.
#
# Usage:
#   bash /rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/slurm/stage2_abl.sh
#
# All use best sweep config: α=0.02 β=0.01 grl=0.01 ρ=0.001
#
# D  no_routing      — bypass routing, feed full z to all heads
# E  fixed_70_30     — freeze routing at 70% L / 30% P, not learned
# F  two_route       — binary L/P routing, no U bucket

DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
SCRIPT="${DIS_DIR}/slurm/stage2_run.sh"

mkdir -p "${DIS_DIR}/logs"

# submit <name> <alpha> <beta> <grl> <rho> <delay> <extra_args>
submit() {
    local name=$1 alpha=$2 beta=$3 grl=$4 rho=$5 delay=$6 extra=${7:-""}
    local jobid
    jobid=$(sbatch --parsable \
        --job-name="s2_${name}" \
        --output="${DIS_DIR}/logs/stage2_${name}_%j.out" \
        --error="${DIS_DIR}/logs/stage2_${name}_%j.err" \
        --export="ALL,RUN_NAME=${name},ALPHA=${alpha},BETA=${beta},GRL_WEIGHT=${grl},RHO=${rho},GRL_DELAY_STEPS=${delay},EXTRA_ARGS=${extra}" \
        "${SCRIPT}")
    printf "  %-18s  job=%-10s  extra=%s\n" "${name}" "${jobid}" "${extra:-none}"
}

echo "=== Submitting ablation jobs (D / E / F) ==="
#          name            α     β     grl   ρ      delay  extra_args
submit   no_routing      0.02  0.01  0.01  0.001  0      "--no_routing"
submit   fixed_70_30     0.02  0.01  0.01  0.001  0      "--fixed_routing --fixed_routing_split 0.7"
submit   two_route       0.02  0.01  0.01  0.001  0      "--n_routes 2"
echo ""
echo "Monitor:  watch -n 30 'squeue -u ${USER} -o \"%.10i %-22j %.8T %.10M %R\"'"
echo "Logs:     ls -lt ${DIS_DIR}/logs/stage2_{no_routing,fixed,two}*.out"
