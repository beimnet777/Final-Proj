#!/bin/bash
# Resubmit 4 failed/missing stage-2 runs + paired probe jobs.
#
# Runs covered:
#   1. K10240_t128       — K=10240 topk=128, stage-2 crashed (K mismatch); stage-1 OK
#   2. K10240_t128_dc    — K=10240 topk=128 + decorr, stage-2 crashed; stage-1 OK
#   3. ste_ub            — β=0.01 grl=0.01 + STE routing + UB weight=0.01; never ran
#   4. combined          — β=0.03 grl=0.01 + dual-GRL phoneme=0.01 + STE + UB=0.01; never ran
#
# Hyperparams match the original sweep scripts:
#   K10240 stage-2   : stage1_sweep.sh  — ALPHA=0.02 BETA=0.01 GRL=0.01 RHO=0.001 DELAY=0
#   ste_ub/combined  : stage2_exp_sweep.sh — same base, β differs for combined

set -euo pipefail

DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
SLURM_DIR="${DIS_DIR}/slurm"
mkdir -p "${DIS_DIR}/logs/stage2" "${DIS_DIR}/logs/probes"

echo "=== Submitting 8 jobs (4 stage-2 + 4 probes) ==="

# ── 1. K10240_t128 ─────────────────────────────────────────────────────────
S2_1=$(sbatch --parsable \
    --job-name=s2_K10240_t128 \
    --output="${DIS_DIR}/logs/stage2/K10240_t128_%j.out" \
    --error="${DIS_DIR}/logs/stage2/K10240_t128_%j.err" \
    --export="ALL,RUN_NAME=K10240_t128,ALPHA=0.02,BETA=0.01,GRL_WEIGHT=0.01,RHO=0.001,GRL_DELAY_STEPS=0,EXTRA_ARGS=--stage1_ckpt ${DIS_DIR}/checkpoints/K10240_t128/stage1_best.pt --n_routes 2 --K 10240 --topk 128" \
    "${SLURM_DIR}/stage2_run.sh")
echo "  [1/8] K10240_t128 stage-2 → job ${S2_1}"

P_1=$(sbatch --parsable \
    --job-name=probe_K10240_t128 \
    --output="${DIS_DIR}/logs/probes/probe_K10240_t128_%j.out" \
    --error="${DIS_DIR}/logs/probes/probe_K10240_t128_%j.err" \
    --dependency=afterok:${S2_1} \
    --export="ALL,RUN_NAME=probe_K10240_t128,STAGE2_CKPT=${DIS_DIR}/checkpoints/K10240_t128/stage2_best.pt,STAGE1_CKPT=${DIS_DIR}/checkpoints/K10240_t128/stage1_best.pt,PROBE_EXTRA=--topk 128" \
    "${SLURM_DIR}/probe_run.sh")
echo "  [2/8] K10240_t128 probe   → job ${P_1}  (afterok:${S2_1})"

# ── 2. K10240_t128_dc ──────────────────────────────────────────────────────
S2_2=$(sbatch --parsable \
    --job-name=s2_K10240_dc \
    --output="${DIS_DIR}/logs/stage2/K10240_t128_dc_%j.out" \
    --error="${DIS_DIR}/logs/stage2/K10240_t128_dc_%j.err" \
    --export="ALL,RUN_NAME=K10240_t128_dc,ALPHA=0.02,BETA=0.01,GRL_WEIGHT=0.01,RHO=0.001,GRL_DELAY_STEPS=0,EXTRA_ARGS=--stage1_ckpt ${DIS_DIR}/checkpoints/K10240_t128_dc/stage1_best.pt --n_routes 2 --K 10240 --topk 128" \
    "${SLURM_DIR}/stage2_run.sh")
echo "  [3/8] K10240_t128_dc stage-2 → job ${S2_2}"

P_2=$(sbatch --parsable \
    --job-name=probe_K10240_dc \
    --output="${DIS_DIR}/logs/probes/probe_K10240_t128_dc_%j.out" \
    --error="${DIS_DIR}/logs/probes/probe_K10240_t128_dc_%j.err" \
    --dependency=afterok:${S2_2} \
    --export="ALL,RUN_NAME=probe_K10240_t128_dc,STAGE2_CKPT=${DIS_DIR}/checkpoints/K10240_t128_dc/stage2_best.pt,STAGE1_CKPT=${DIS_DIR}/checkpoints/K10240_t128_dc/stage1_best.pt,PROBE_EXTRA=--topk 128" \
    "${SLURM_DIR}/probe_run.sh")
echo "  [4/8] K10240_t128_dc probe   → job ${P_2}  (afterok:${S2_2})"

# ── 3. ste_ub ──────────────────────────────────────────────────────────────
S2_3=$(sbatch --parsable \
    --job-name=s2_ste_ub \
    --output="${DIS_DIR}/logs/stage2/ste_ub_%j.out" \
    --error="${DIS_DIR}/logs/stage2/ste_ub_%j.err" \
    --export="ALL,RUN_NAME=ste_ub,ALPHA=0.02,BETA=0.01,GRL_WEIGHT=0.01,RHO=0.001,GRL_DELAY_STEPS=0,EXTRA_ARGS=--ste_routing --ub_weight 0.01" \
    "${SLURM_DIR}/stage2_run.sh")
echo "  [5/8] ste_ub stage-2 → job ${S2_3}"

P_3=$(sbatch --parsable \
    --job-name=probe_ste_ub \
    --output="${DIS_DIR}/logs/probes/probe_ste_ub_%j.out" \
    --error="${DIS_DIR}/logs/probes/probe_ste_ub_%j.err" \
    --dependency=afterok:${S2_3} \
    --export="ALL,RUN_NAME=probe_ste_ub,STAGE2_CKPT=${DIS_DIR}/checkpoints/ste_ub/stage2_best.pt" \
    "${SLURM_DIR}/probe_run.sh")
echo "  [6/8] ste_ub probe   → job ${P_3}  (afterok:${S2_3})"

# ── 4. combined ────────────────────────────────────────────────────────────
S2_4=$(sbatch --parsable \
    --job-name=s2_combined \
    --output="${DIS_DIR}/logs/stage2/combined_%j.out" \
    --error="${DIS_DIR}/logs/stage2/combined_%j.err" \
    --export="ALL,RUN_NAME=combined,ALPHA=0.02,BETA=0.03,GRL_WEIGHT=0.01,RHO=0.001,GRL_DELAY_STEPS=0,EXTRA_ARGS=--grl_phoneme_weight 0.01 --ste_routing --ub_weight 0.01" \
    "${SLURM_DIR}/stage2_run.sh")
echo "  [7/8] combined stage-2 → job ${S2_4}"

P_4=$(sbatch --parsable \
    --job-name=probe_combined \
    --output="${DIS_DIR}/logs/probes/probe_combined_%j.out" \
    --error="${DIS_DIR}/logs/probes/probe_combined_%j.err" \
    --dependency=afterok:${S2_4} \
    --export="ALL,RUN_NAME=probe_combined,STAGE2_CKPT=${DIS_DIR}/checkpoints/combined/stage2_best.pt" \
    "${SLURM_DIR}/probe_run.sh")
echo "  [8/8] combined probe   → job ${P_4}  (afterok:${S2_4})"

echo ""
echo "=== All 8 jobs submitted ==="
printf "Stage-2 jobs : %s  %s  %s  %s\n" "${S2_1}" "${S2_2}" "${S2_3}" "${S2_4}"
printf "Probe jobs   : %s  %s  %s  %s\n" "${P_1}"  "${P_2}"  "${P_3}"  "${P_4}"
echo ""
echo "Monitor: squeue -u \${USER}"
