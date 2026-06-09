#!/bin/bash
# stage2_sweep.sh — submit all stage-2 variation jobs in parallel.
#
# Usage (from any directory):
#   bash /rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/slurm/stage2_sweep.sh
#
# Each run gets isolated checkpoint_dir and runs_dir under checkpoints/<name>/ and runs/<name>/.
# Logs land in logs/stage2_<name>_<jobid>.out/.err
#
# Runs:
#   0  baseline          α=0.02  β=0.003  grl=0.04  ρ=0.001  delay=0    (already ran as 29880935 — skipped)
#   1  sid1              α=0.02  β=0.01   grl=0.04  ρ=0.001  delay=0
#   2  sid1_weakgrl      α=0.02  β=0.01   grl=0.01  ρ=0.001  delay=0
#   3  sid1_nogrl        α=0.02  β=0.01   grl=0.00  ρ=0.001  delay=0
#   4  sid1_delayedgrl   α=0.02  β=0.01   grl=0.04  ρ=0.001  delay=2000
#   5  sid1_highrho      α=0.02  β=0.01   grl=0.01  ρ=0.005  delay=0

DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
SCRIPT="${DIS_DIR}/slurm/stage2_run.sh"

mkdir -p "${DIS_DIR}/logs"

# submit <name> <alpha> <beta> <grl_weight> <rho> <grl_delay_steps>
submit() {
    local name=$1 alpha=$2 beta=$3 grl=$4 rho=$5 delay=$6
    local jobid
    jobid=$(sbatch --parsable \
        --job-name="s2_${name}" \
        --output="${DIS_DIR}/logs/stage2_${name}_%j.out" \
        --error="${DIS_DIR}/logs/stage2_${name}_%j.err" \
        --export="ALL,RUN_NAME=${name},ALPHA=${alpha},BETA=${beta},GRL_WEIGHT=${grl},RHO=${rho},GRL_DELAY_STEPS=${delay}" \
        "${SCRIPT}")
    printf "  %-22s  job_id=%-10s  α=%-5s β=%-6s grl=%-5s ρ=%-6s delay=%s\n" \
        "${name}" "${jobid}" "${alpha}" "${beta}" "${grl}" "${rho}" "${delay}"
}

echo "=== Submitting stage-2 sweep ==="
#          name                alpha  beta   grl    rho    delay
# run 0 already completed as 29880935 — uncomment to rerun
# submit baseline            0.02   0.003  0.04   0.001  0
submit   sid1               0.02   0.01   0.04   0.001  0
submit   sid1_weakgrl       0.02   0.01   0.01   0.001  0
submit   sid1_nogrl         0.02   0.01   0.00   0.001  0
submit   sid1_delayedgrl    0.02   0.01   0.04   0.001  2000
submit   sid1_highrho       0.02   0.01   0.01   0.005  0
echo ""
echo "Monitor:  watch -n 30 'squeue -u ${USER} -o \"%.10i %-22j %.8T %.10M %.6D %R\"'"
echo "Logs:     ls -lt ${DIS_DIR}/logs/stage2_sid*.out"
