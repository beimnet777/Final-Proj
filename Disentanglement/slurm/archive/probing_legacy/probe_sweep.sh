#!/bin/bash
# probe_sweep.sh — submit probing jobs A, B, C in parallel.
#
# Usage:
#   bash /rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/slurm/probe_sweep.sh
#
# A — probe Run 2 (sid1_weakgrl, best config): z_L, z_P, z_t, h_t
# B — baselines only (h_t, z_t); no stage-2 checkpoint needed
# C — probe Run 3 (sid1_nogrl): z_L, z_P, z_t, h_t

DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
SCRIPT="${DIS_DIR}/slurm/probe_run.sh"

mkdir -p "${DIS_DIR}/logs"

submit() {
    local name=$1 ckpt=${2:-""}
    local jobid
    jobid=$(sbatch --parsable \
        --job-name="probe_${name}" \
        --output="${DIS_DIR}/logs/probe_${name}_%j.out" \
        --error="${DIS_DIR}/logs/probe_${name}_%j.err" \
        --export="ALL,RUN_NAME=${name},STAGE2_CKPT=${ckpt}" \
        "${SCRIPT}")
    printf "  %-15s  job=%s  ckpt=%s\n" "${name}" "${jobid}" "${ckpt:-none}"
}

echo "=== Submitting probe jobs (A / B / C) ==="
submit probe_B  ""
submit probe_A  "${DIS_DIR}/checkpoints/sid1_weakgrl/stage2_best.pt"
submit probe_C  "${DIS_DIR}/checkpoints/sid1_nogrl/stage2_best.pt"
echo ""
echo "Logs:  ls -lt ${DIS_DIR}/logs/probe_*.out"
