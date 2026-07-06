#!/usr/bin/env bash
# Copy this file, give it a descriptive name, and commit it before launching.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

RUN_NAME="REPLACE_ME"
EXPERIMENT_CONFIGURED=false
if [[ "$RUN_NAME" == "REPLACE_ME" || "$EXPERIMENT_CONFIGURED" != true ]]; then
    echo "ERROR: copy this template, define the command, and set EXPERIMENT_CONFIGURED=true." >&2
    exit 2
fi

# Keep every scientific choice here so Git records the exact experiment.
# Runtime-only state (the manually allocated GPU) is supplied as GPU_ID.
COMMAND=(
    python -u Disentanglement/run.py
    # Add every flag here, including data and output paths. For example:
    # --stage 2
    # --local_data
    # --librispeech_root "$BLACKWELL_DATA_ROOT/LibriSpeech"
    # --lexicon_path "$BLACKWELL_DATA_ROOT/librispeech-lexicon.txt"
    # --checkpoint_dir "$BLACKWELL_OUTPUT_ROOT/$RUN_NAME/checkpoints"
    # --runs_dir "$BLACKWELL_OUTPUT_ROOT/$RUN_NAME/tensorboard"
    # --log_dir "$BLACKWELL_OUTPUT_ROOT/$RUN_NAME/trainer_logs"
)

blackwell_run "$RUN_NAME" "${COMMAND[@]}"
