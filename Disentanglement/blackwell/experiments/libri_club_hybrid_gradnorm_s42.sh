#!/usr/bin/env bash
# Full LibriSpeech CLUB-hybrid run with normalized speaker-CLUB gradients.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

RUN_NAME="libri_club_hybrid_gradnorm_s42"

# Scientific configuration:
#   - soft two-route learned routing
#   - pair-alpha/pair-beta dual invariance with VICReg variance + covariance
#   - speaker CLUB on z_L; speaker GRL off
#   - sign-preserving normalized CLUB gradient, target 0.005
#   - phoneme GRL on z_P; phoneme CLUB off
# The named preset and every override are resolved into RUN_NAME/resolved_config.yaml.
COMMAND=(
    python -u -m Disentanglement.experiment_runner
    --experiment libri_club_hybrid
    --data_root "$BLACKWELL_DATA_ROOT"
    --profile full
    --phase train
    --seed 42
    --effective_batch_size 16
    --microbatch_size auto
    --resume auto
    --segment_steps 0
    --max_runtime_minutes 0
    --resume_every 50
    --precision auto
    --output_dir "$BLACKWELL_OUTPUT_ROOT/$RUN_NAME"
    --set club_grad_norm=true
    --set club_grad_norm_target=0.005
)

# Optional allocation-time validation: it performs the CUDA smoke test and
# resolves the full command/configuration, but does not start training.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    COMMAND+=(--dry_run)
fi

blackwell_run "$RUN_NAME" "${COMMAND[@]}"

