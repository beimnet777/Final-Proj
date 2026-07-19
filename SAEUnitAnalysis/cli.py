from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import run_analysis
from .utils import AnalysisError


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone routed-SAE unit analysis.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path, help="analysis-bundle root")
    p.add_argument("--analysis", required=True,
                   help="comma-separated health,atlas,selectivity,factor_metrics,clustering,similarity,geometry,causal,swap or all")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", choices=("full", "quick"), default="full")
    p.add_argument(
        "--atlas-assets",
        default="none",
        help=(
            "Optional comma-separated atlas media to generate: none, traces, "
            "spectrograms, audio, or all. Default is none to keep outputs small."
        ),
    )
    p.add_argument(
        "--factor-scope",
        choices=("speaker_phone", "broad"),
        default="speaker_phone",
        help=(
            "Factor set for selectivity. Default speaker_phone computes only "
            "phone identity and speaker_id; broad also includes declared "
            "prosody/metadata/phone-property factors."
        ),
    )
    p.add_argument(
        "--score-splits",
        default="train,validation",
        help=(
            "Comma-separated splits used to score/rank units. Default excludes "
            "test: train,validation. Use all only for exploratory diagnostics."
        ),
    )
    p.add_argument(
        "--threshold-percentile",
        type=float,
        default=0.90,
        help="Percentile threshold for high PhoneScore/SpeakerScore categories.",
    )
    p.add_argument(
        "--split-limits",
        default=None,
        help=(
            "Optional deterministic speaker-balanced caps, for example "
            "train=3000,validation=1000,test=1000."
        ),
    )
    return p


def main() -> None:
    args = _parser().parse_args()
    try:
        result = run_analysis(
            args.checkpoint, args.data, args.analysis,
            output_dir=args.output_dir, device=args.device,
            seed=args.seed, profile=args.profile,
            atlas_assets=args.atlas_assets,
            factor_scope=args.factor_scope,
            score_splits=args.score_splits,
            threshold_percentile=args.threshold_percentile,
            split_limits=args.split_limits,
        )
    except AnalysisError as exc:
        print(f"[SAEUnitAnalysis] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(f"[SAEUnitAnalysis] completed: {', '.join(result.completed)}")
    print(f"[SAEUnitAnalysis] report: {result.artifacts['report']}")
