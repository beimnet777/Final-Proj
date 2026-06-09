#!/bin/bash
# stage1_sweep.sh — submit stage-1 variants for Exp 2 (decorrelation) and Exp 3 (larger SAE).
#
# Usage:
#   bash slurm/stage1_sweep.sh
#
# Variants:
#   decor_only      K=5120  topk=256  decor=0.01   (Exp 2 at current scale)
#   K10240_t128     K=10240 topk=128  no decor      (Exp 3)
#   K10240_t128_dc  K=10240 topk=128  decor=0.01    (Exp 3 + Exp 2)
#   K10240_t200     K=10240 topk=200  no decor      (Exp 3 alternative topk)
#
# Each stage-2 job is queued with --dependency=afterok on its stage-1 job.

DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
S1_SCRIPT="${DIS_DIR}/slurm/stage1_run.sh"
S2_SCRIPT="${DIS_DIR}/slurm/stage2_run.sh"

mkdir -p "${DIS_DIR}/logs/stage1"

# submit_pair <name> <stage1_extra> <stage2_extra>
submit_pair() {
    local name=$1 s1_extra=$2 s2_extra=${3:-""}
    local ckpt="${DIS_DIR}/checkpoints/${name}/stage1_best.pt"

    local s1_id
    s1_id=$(sbatch --parsable \
        --job-name="s1_${name}" \
        --output="${DIS_DIR}/logs/stage1/stage1_${name}_%j.out" \
        --error="${DIS_DIR}/logs/stage1/stage1_${name}_%j.err" \
        --export="ALL,RUN_NAME=${name},EXTRA_ARGS=${s1_extra}" \
        "${S1_SCRIPT}")

    # Stage 2 uses best config: β=0.01 grl=0.01 ρ=0.001, pointing at new checkpoint
    local s2_id
    s2_id=$(sbatch --parsable \
        --dependency=afterok:${s1_id} \
        --job-name="s2_${name}" \
        --output="${DIS_DIR}/logs/stage2/${name}_%j.out" \
        --error="${DIS_DIR}/logs/stage2/${name}_%j.err" \
        --export="ALL,RUN_NAME=${name},ALPHA=0.02,BETA=0.01,GRL_WEIGHT=0.01,RHO=0.001,GRL_DELAY_STEPS=0,EXTRA_ARGS=--stage1_ckpt ${ckpt} ${s2_extra}" \
        "${S2_SCRIPT}")

    printf "  %-20s  stage1=%-10s  stage2=%-10s (after stage1)\n" "${name}" "${s1_id}" "${s2_id}"
}

mkdir -p "${DIS_DIR}/logs/stage2"

echo "=== Submitting stage-1 sweep (Exp 2 + Exp 3) ==="
#               name             stage1_extra                              stage2_extra
submit_pair  decor_only      "--K 5120  --topk 256 --decor_weight 0.01"   ""
submit_pair  K10240_t128     "--K 10240 --topk 128"                        "--n_routes 2"
submit_pair  K10240_t128_dc  "--K 10240 --topk 128 --decor_weight 0.01"   "--n_routes 2"
submit_pair  K10240_t200     "--K 10240 --topk 200"                        "--n_routes 2"
echo ""
echo "Monitor:  squeue -u ${USER}"
