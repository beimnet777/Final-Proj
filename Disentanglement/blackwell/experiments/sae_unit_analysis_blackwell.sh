#!/usr/bin/env bash
# Standalone Blackwell launcher for SAEUnitAnalysis.
#
# This script does NOT train a model and does NOT run diagnostic probes.  It only
# analyzes an existing routed-SAE checkpoint against an analysis bundle.
#
# Required:
#   GPU_ID=0..7
#   SAE_CHECKPOINT=/path/to/stage2_or_final_checkpoint.pt
#   SAE_DATA_BUNDLE=/path/to/analysis_bundle
#
# Recommended first pass:
#   SAE_ANALYSIS=health,atlas,selectivity,clustering,similarity,geometry
#
# Example:
#   tmux new-session -d -s sae_units \
#     "bash -lc 'cd /scratch/$USER/Final-Proj && \
#       GPU_ID=1 \
#       SAE_RUN_NAME=sae_gn00015_librispeech \
#       SAE_CHECKPOINT=/scratch/$USER/runs/libri_advfb_prrecover_gn00015_240L16P_aux64_12k_s42/checkpoints/stage2_step12000.pt \
#       SAE_DATA_BUNDLE=/scratch/$USER/data/sae_analysis/librispeech_bundle \
#       ./Disentanglement/blackwell/experiments/sae_unit_analysis_blackwell.sh'"
#
# Useful overrides:
#   SAE_RUN_NAME          log/output group name
#   SAE_ANALYSIS          comma-separated analyses, or all
#   SAE_PROFILE           full|quick
#   SAE_SEED              default 42
#   SAE_DEVICE            default cuda
#   SAE_OUTPUT_DIR        default $BLACKWELL_OUTPUT_ROOT/$SAE_RUN_NAME/results
#   SAE_INSTALL_DEPS=1    install SAEUnitAnalysis/requirements.txt in the venv
#   SAE_EXPECT_BLOCK_TOPK=240,16,0
#                         optional hard check for fixed-block extraction.
#                         If set, the script checks the resolved checkpoint
#                         config before analysis and the route active slots
#                         after analysis.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

SAE_ANALYSIS="${SAE_ANALYSIS:-health,atlas,selectivity,clustering,similarity,geometry}"
SAE_PROFILE="${SAE_PROFILE:-full}"
SAE_SEED="${SAE_SEED:-42}"
SAE_DEVICE="${SAE_DEVICE:-cuda}"
SAE_INSTALL_DEPS="${SAE_INSTALL_DEPS:-0}"
SAE_EXPECT_BLOCK_TOPK="${SAE_EXPECT_BLOCK_TOPK:-}"

# SPEAR's cached HuggingFace module currently emits repeated PyTorch
# FutureWarnings for torch.cuda.amp.autocast.  They are not actionable for this
# analysis and can bury real failures in tmux/log output.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

if [[ -z "${SAE_CHECKPOINT:-}" ]]; then
    echo "ERROR: set SAE_CHECKPOINT to the checkpoint .pt file to analyze." >&2
    exit 2
fi
if [[ -z "${SAE_DATA_BUNDLE:-}" ]]; then
    echo "ERROR: set SAE_DATA_BUNDLE to the analysis-bundle directory." >&2
    exit 2
fi
if [[ "$SAE_PROFILE" != "full" && "$SAE_PROFILE" != "quick" ]]; then
    echo "ERROR: SAE_PROFILE must be full or quick (got: $SAE_PROFILE)." >&2
    exit 2
fi
if [[ "$SAE_INSTALL_DEPS" != "0" && "$SAE_INSTALL_DEPS" != "1" ]]; then
    echo "ERROR: SAE_INSTALL_DEPS must be 0 or 1 (got: $SAE_INSTALL_DEPS)." >&2
    exit 2
fi
if [[ -n "$SAE_EXPECT_BLOCK_TOPK" && ! "$SAE_EXPECT_BLOCK_TOPK" =~ ^[0-9]+,[0-9]+,[0-9]+$ ]]; then
    echo "ERROR: SAE_EXPECT_BLOCK_TOPK must look like 240,16,0 (got: $SAE_EXPECT_BLOCK_TOPK)." >&2
    exit 2
fi

abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$REPO_ROOT/$1" ;;
    esac
}

SAE_CHECKPOINT="$(abs_path "$SAE_CHECKPOINT")"
SAE_DATA_BUNDLE="$(abs_path "$SAE_DATA_BUNDLE")"

if [[ ! -f "$SAE_CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $SAE_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -d "$SAE_DATA_BUNDLE" ]]; then
    echo "ERROR: analysis bundle not found: $SAE_DATA_BUNDLE" >&2
    exit 1
fi
if [[ ! -f "$SAE_DATA_BUNDLE/dataset.yaml" ]]; then
    echo "ERROR: analysis bundle is missing dataset.yaml: $SAE_DATA_BUNDLE" >&2
    exit 1
fi

sanitize_name() {
    printf '%s' "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//'
}

if [[ -z "${SAE_RUN_NAME:-}" ]]; then
    ckpt_stem="$(basename "$SAE_CHECKPOINT")"
    ckpt_stem="${ckpt_stem%.pt}"
    data_stem="$(basename "$SAE_DATA_BUNDLE")"
    SAE_RUN_NAME="$(sanitize_name "sae_${data_stem}_${ckpt_stem}_${SAE_PROFILE}")"
fi
SAE_RUN_NAME="$(sanitize_name "$SAE_RUN_NAME")"
if [[ -z "$SAE_RUN_NAME" ]]; then
    echo "ERROR: SAE_RUN_NAME became empty after sanitization." >&2
    exit 2
fi

SAE_OUTPUT_DIR="${SAE_OUTPUT_DIR:-$BLACKWELL_OUTPUT_ROOT/$SAE_RUN_NAME/results}"
SAE_OUTPUT_DIR="$(abs_path "$SAE_OUTPUT_DIR")"

export BLACKWELL_LOG_GROUP="$SAE_RUN_NAME"

if [[ "$SAE_INSTALL_DEPS" == "1" ]]; then
    RUN_DESCRIPTION="Install SAEUnitAnalysis dependencies before standalone unit analysis"
    blackwell_run "${SAE_RUN_NAME}_install_deps" \
        python -m pip install -r SAEUnitAnalysis/requirements.txt
fi

if [[ -n "$SAE_EXPECT_BLOCK_TOPK" ]]; then
    RUN_DESCRIPTION="Preflight SAE extraction-config check: checkpoint=$SAE_CHECKPOINT expected_block_topk=$SAE_EXPECT_BLOCK_TOPK"
    blackwell_run "${SAE_RUN_NAME}_preflight_block_topk" \
        python - "$SAE_CHECKPOINT" "$SAE_EXPECT_BLOCK_TOPK" <<'PY'
import sys
from SAEUnitAnalysis.checkpoint import load_checkpoint

checkpoint, expected_text = sys.argv[1], sys.argv[2]
expected = [int(x) for x in expected_text.split(",")]
resolved = load_checkpoint(checkpoint)
actual = resolved.config.get("block_topk") or resolved.config.get("topk_blocks")
print("resolved topk:", resolved.config.get("topk"))
print("resolved block_topk:", actual)
print("resolved topk_L/P/U:", [resolved.config.get(k) for k in ("topk_L", "topk_P", "topk_U")])
print("warnings:", resolved.warnings)
if list(actual or []) != expected:
    raise SystemExit(f"ERROR: expected block_topk={expected}, got {actual}")
if int(resolved.config.get("topk", sum(expected))) != sum(expected):
    raise SystemExit(
        f"ERROR: expected topk={sum(expected)} from block_topk, "
        f"got {resolved.config.get('topk')}"
    )
PY
fi

RUN_DESCRIPTION="Standalone SAE unit analysis: checkpoint=$SAE_CHECKPOINT data=$SAE_DATA_BUNDLE analysis=$SAE_ANALYSIS profile=$SAE_PROFILE"
COMMAND=(
    python -u -m SAEUnitAnalysis
    --checkpoint "$SAE_CHECKPOINT"
    --data "$SAE_DATA_BUNDLE"
    --analysis "$SAE_ANALYSIS"
    --output-dir "$SAE_OUTPUT_DIR"
    --device "$SAE_DEVICE"
    --seed "$SAE_SEED"
    --profile "$SAE_PROFILE"
)

blackwell_run "$SAE_RUN_NAME" "${COMMAND[@]}"

if [[ -n "$SAE_EXPECT_BLOCK_TOPK" ]]; then
    RUN_DESCRIPTION="Postflight SAE route active-slot check: output=$SAE_OUTPUT_DIR expected_block_topk=$SAE_EXPECT_BLOCK_TOPK"
    blackwell_run "${SAE_RUN_NAME}_postflight_active_slots" \
        python - "$SAE_OUTPUT_DIR" "$SAE_EXPECT_BLOCK_TOPK" <<'PY'
import json
import sys
from pathlib import Path

output_dir, expected_text = Path(sys.argv[1]), sys.argv[2]
expected = [float(x) for x in expected_text.split(",")]
health_path = output_dir / "health.json"
resolved_path = output_dir / "resolved_model.json"
if not health_path.exists():
    raise SystemExit(f"ERROR: missing health.json: {health_path}")
if not resolved_path.exists():
    raise SystemExit(f"ERROR: missing resolved_model.json: {resolved_path}")

resolved = json.load(open(resolved_path))
warnings = resolved.get("warnings") or []
bad_warnings = [w for w in warnings if "Inferred extraction settings" in str(w)]
if bad_warnings:
    raise SystemExit(f"ERROR: analysis used calibration inference unexpectedly: {bad_warnings}")

health = json.load(open(health_path))
by_route = {row["route"]: row for row in health.get("route_summary", [])}
for route, want in zip(("L", "P", "U"), expected):
    row = by_route.get(route)
    got = 0.0 if row is None else float(row.get("active_slots_per_frame", 0.0))
    # U may be absent when K_U=0, so treating missing U as 0 is intended.
    print(f"{route}: active_slots_per_frame={got} expected={want}")
    if abs(got - want) > 1e-6:
        raise SystemExit(
            f"ERROR: route {route} active_slots_per_frame mismatch: expected {want}, got {got}"
        )
print("deadness:", {
    "active_units": health.get("active_units"),
    "unobserved_units": health.get("unobserved_units"),
    "train_like_dead_units": health.get("train_like_dead_units"),
    "deadness_analysis_batches": health.get("deadness_analysis_batches"),
    "deadness_threshold_batches": health.get("deadness_threshold_batches"),
})
PY
fi

echo
echo "SAE unit analysis complete."
echo "Report: ${SAE_OUTPUT_DIR}/report/index.html"
echo "Tables: ${SAE_OUTPUT_DIR}/tables"
echo "Plots:  ${SAE_OUTPUT_DIR}/plots"
