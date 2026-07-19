from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .bundle import AnalysisBundle
from .extraction import FeatureCache
from .factor_metrics import speech_factor_metrics
from .report import build_report, make_plots
from .types import ResolvedModel
from .utils import AnalysisError, write_json


TABLE_FILES = {
    "health": "units.csv",
    "profiles": "unit_profiles.csv",
    "phone_speaker_scores": "unit_phone_speaker_scores.csv",
    "phone_unit_confusion": "phone_selected_unit_confusion.csv",
    "selected_phone_units": "phone_selected_units.csv",
    "phone_embedding": "route_phone_representation_embedding.csv",
    "speaker_embedding": "route_speaker_representation_embedding.csv",
    "representation_separation": "route_representation_separation.csv",
    "probe_confusion": "route_probe_confusion.csv",
    "classifier_free_geometry_summary": "route_classifier_free_geometry_summary.csv",
    "disentanglement": "unit_disentanglement.csv",
    "leaky": "leaky_units.csv",
    "route_summary": "route_disentanglement_summary.csv",
    "swap": "swaps.csv",
    "swap_summary": "swap_mode_summary.csv",
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise AnalysisError(f"Required existing result artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_tables(result_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in TABLE_FILES.items():
        path = result_dir / "tables" / filename
        if path.exists():
            tables[name] = pd.read_csv(path)
    return tables


def add_factor_metrics(
    result_dir: str | Path,
    *,
    max_segments: int = 20_000,
    bootstrap_repetitions: int = 100,
    dci_repeats: int = 5,
    dci_estimators: int = 48,
) -> dict:
    result_dir = Path(result_dir).resolve()
    manifest = _read_json(result_dir / "run_manifest.json")
    resolved_data = _read_json(result_dir / "resolved_model.json")
    summary = _read_json(result_dir / "summary.json")
    cache = FeatureCache.load(Path(manifest["cache"]))
    bundle = AnalysisBundle(Path(manifest["data"]))
    metrics, importance, repeats, metric_summary = speech_factor_metrics(
        cache, bundle, result_dir,
        seed=int(manifest.get("seed", 42)),
        max_segments=max_segments,
        bootstrap_repetitions=bootstrap_repetitions,
        dci_repeats=dci_repeats,
        dci_estimators=dci_estimators,
    )
    tables = _existing_tables(result_dir)
    tables["factor_metrics"] = metrics
    tables["dci_importance"] = importance
    tables["factor_metric_repeats"] = repeats
    # make_plots intentionally prunes stale files, so pass every preserved
    # report table while adding the new metric figure.
    plot_paths = make_plots(result_dir, tables)
    if not any(path.stem == "speech_factor_metrics" for path in plot_paths):
        raise AnalysisError("Factor metrics completed but produced no report plot.")

    resolved = ResolvedModel(
        checkpoint=Path(resolved_data.get("checkpoint", manifest["checkpoint"])),
        state={},
        config=resolved_data.get("config", {}),
        source_format=str(resolved_data.get("format", "unknown")),
        capabilities=resolved_data.get("capabilities", {}),
        warnings=list(resolved_data.get("warnings", [])),
    )
    completed = list(manifest.get("analyses", []))
    if "factor_metrics" not in completed:
        position = completed.index("selectivity") + 1 if "selectivity" in completed else len(completed)
        completed.insert(position, "factor_metrics")
    summary["factor_metrics"] = metric_summary
    report = build_report(
        result_dir, resolved, completed, summary, tables, resolved.warnings, plot_paths,
        profile=str(manifest.get("profile", "full")),
    )
    manifest["analyses"] = completed
    manifest["factor_metrics"] = {
        "max_segments": int(max_segments),
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "dci_repeats": int(dci_repeats),
        "dci_estimators": int(dci_estimators),
    }
    manifest["report"] = str(report)
    write_json(result_dir / "summary.json", summary)
    write_json(result_dir / "run_manifest.json", manifest)
    return metric_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add speech-adapted MIG/DCI/SAP to existing SAE 5k result folders using their caches.",
    )
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--max-segments", type=int, default=20_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100)
    parser.add_argument("--dci-repeats", type=int, default=5)
    parser.add_argument("--dci-estimators", type=int, default=48)
    return parser


def main() -> None:
    args = _parser().parse_args()
    for result in args.results:
        print(f"[SAEUnitAnalysis] adding factor metrics: {result}", flush=True)
        summary = add_factor_metrics(
            result,
            max_segments=args.max_segments,
            bootstrap_repetitions=args.bootstrap_repetitions,
            dci_repeats=args.dci_repeats,
            dci_estimators=args.dci_estimators,
        )
        print(
            f"[SAEUnitAnalysis] completed {result}: "
            f"segments={summary['sampled_segments']} "
            f"contrasts={summary['headline_route_contrasts']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
