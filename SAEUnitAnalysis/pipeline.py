from __future__ import annotations

import platform
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from .analyses import (
    clustering_analysis, geometry_analysis, health_analysis, selectivity_analysis,
    disentanglement_tables, similarity_analysis, top_examples,
)
from .bundle import AnalysisBundle
from .causal import causal_analysis, swap_analysis
from .checkpoint import load_checkpoint
from .evaluators import train_evaluators
from .extraction import calibrate, extract
from .report import build_atlas, build_report, make_plots
from .types import ANALYSES, AnalysisResult
from .utils import AnalysisError, fingerprint, set_seed, write_json


DEPENDENCIES = {
    "atlas": {"health"}, "clustering": {"selectivity"},
    "causal": {"health", "selectivity"}, "swap": {"health", "selectivity"},
}


def _expand(analyses: Sequence[str] | str) -> list[str]:
    if isinstance(analyses, str):
        analyses = [x.strip() for x in analyses.split(",") if x.strip()]
    requested = list(ANALYSES) if "all" in analyses else list(analyses)
    bad = sorted(set(requested) - set(ANALYSES))
    if bad:
        raise AnalysisError(f"Unknown analyses {bad}; valid choices are {list(ANALYSES)} or all.")
    changed = True
    while changed:
        changed = False
        for name in list(requested):
            for dep in DEPENDENCIES.get(name, set()):
                if dep not in requested:
                    requested.insert(0, dep); changed = True
    return [x for x in ANALYSES if x in requested]


def run_analysis(
    checkpoint: str | Path,
    data_root: str | Path,
    analyses: Sequence[str] | str,
    *,
    output_dir: str | Path | None = None,
    device: str | None = None,
    seed: int = 42,
    profile: str = "full",
) -> AnalysisResult:
    set_seed(seed)
    selected = _expand(analyses)
    bundle = AnalysisBundle(data_root)
    resolved = load_checkpoint(checkpoint)
    for name in selected:
        bundle.require(name)
        if name in {"selectivity", "clustering", "similarity", "geometry", "causal", "swap"} and not resolved.capabilities["unit_routes"]:
            raise AnalysisError(f"Checkpoint architecture does not support unit-route analysis '{name}'.")
        if name in {"causal", "swap"} and not resolved.capabilities[name]:
            raise AnalysisError(f"Checkpoint architecture does not support '{name}'.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if profile not in {"full", "quick"}:
        raise AnalysisError("profile must be 'full' or 'quick'.")

    root = Path(__file__).resolve().parent
    dataset_key = fingerprint([bundle.spec.manifest_path], bundle.spec.raw)
    output = Path(output_dir) if output_dir else root / "results" / Path(checkpoint).stem / dataset_key
    output = output.resolve(); (output / "tables").mkdir(parents=True, exist_ok=True)
    cache_dir = root / "cache" / Path(checkpoint).stem / dataset_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle.write_validation_report(output / "data_validation.json")

    model = calibrate(resolved, bundle, device)
    cache = extract(resolved, bundle, model, cache_dir, device, profile)
    write_json(output / "resolved_model.json", {
        "checkpoint": str(resolved.checkpoint), "format": resolved.source_format,
        "config": resolved.config, "capabilities": resolved.capabilities,
        "warnings": resolved.warnings,
    })

    tables: dict[str, pd.DataFrame] = {}
    summaries: dict = {}
    health = profiles = scores = causal_table = None
    if "health" in selected:
        health, summaries["health"] = health_analysis(cache, resolved, output); tables["health"] = health
    if "selectivity" in selected:
        scores, profiles, summaries["selectivity"] = selectivity_analysis(cache, bundle, output)
        if health is not None:
            profiles = profiles.merge(health[["unit", "frame_frequency"]], on="unit", how="left")
        tables["scores"], tables["profiles"] = scores, profiles
        disent, leaky, route_summary, summaries["disentanglement"] = disentanglement_tables(
            health, profiles, scores, output)
        tables["disentanglement"] = disent
        tables["leaky"] = leaky
        tables["route_summary"] = route_summary
    if "clustering" in selected:
        clustered, summaries["clustering"] = clustering_analysis(cache, profiles, output, seed)
        tables["clusters"] = clustered
    if "similarity" in selected:
        summaries["similarity"] = similarity_analysis(cache, bundle, output, seed)
    if "geometry" in selected:
        geometry, summaries["geometry"] = geometry_analysis(cache, resolved, health, output)
        tables["geometry"] = geometry

    suite = None
    if "causal" in selected or "swap" in selected:
        suite = train_evaluators(cache, bundle, cache_dir, seed)
    if "causal" in selected:
        causal_table, summaries["causal"] = causal_analysis(
            cache, bundle, resolved, suite, profiles, output, seed, profile == "quick")
        tables["causal"] = causal_table
    if "swap" in selected:
        swaps, summaries["swap"] = swap_analysis(
            cache, bundle, resolved, suite, output, seed, profile == "quick")
        tables["swap"] = swaps
    if "atlas" in selected:
        examples = top_examples(cache, bundle)
        build_atlas(output, cache, bundle, health, examples, scores, causal_table)
        tables["examples"] = examples

    plots = make_plots(output, tables)
    report = build_report(output, resolved, selected, summaries, tables, resolved.warnings, plots)
    write_json(output / "summary.json", summaries)
    write_json(output / "run_manifest.json", {
        "checkpoint": str(Path(checkpoint).resolve()), "data": str(Path(data_root).resolve()),
        "analyses": selected, "profile": profile, "device": device, "seed": seed,
        "python": platform.python_version(), "torch": torch.__version__,
        "cache": str(cache.path), "report": str(report),
    })
    artifacts = {"report": report, "summary": output / "summary.json", "cache": cache.path}
    return AnalysisResult(output, selected, artifacts, resolved.warnings, summaries)
