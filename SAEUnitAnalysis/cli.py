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
                   help="comma-separated health,atlas,selectivity,clustering,similarity,geometry,causal,swap or all")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", choices=("full", "quick"), default="full")
    return p


def main() -> None:
    args = _parser().parse_args()
    try:
        result = run_analysis(
            args.checkpoint, args.data, args.analysis,
            output_dir=args.output_dir, device=args.device,
            seed=args.seed, profile=args.profile,
        )
    except AnalysisError as exc:
        print(f"[SAEUnitAnalysis] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(f"[SAEUnitAnalysis] completed: {', '.join(result.completed)}")
    print(f"[SAEUnitAnalysis] report: {result.artifacts['report']}")

