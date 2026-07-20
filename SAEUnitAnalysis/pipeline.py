from __future__ import annotations

import platform
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from .analyses import (
    clustering_analysis, deadness_analysis, geometry_analysis, health_analysis, selectivity_analysis,
    disentanglement_tables, phone_speaker_unit_scores, phone_unit_confusion,
    unrouted_unit_summary,
    route_representation_embeddings, classifier_free_route_geometry,
    similarity_analysis, top_examples, _load_umap,
)
from .bundle import AnalysisBundle
from .causal import causal_analysis, swap_analysis
from .checkpoint import load_checkpoint
from .evaluators import train_evaluators
from .extraction import calibrate, extract, parse_split_limits
from .factor_metrics import speech_factor_metrics
from .report import build_atlas, build_report, make_plots
from .types import ANALYSES, AnalysisResult
from .utils import AnalysisError, fingerprint, set_seed, write_json


DEPENDENCIES = {
    "atlas": {"health"}, "clustering": {"selectivity"},
    "geometry": {"health"},
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
    atlas_assets: str = "none",
    factor_scope: str = "speaker_phone",
    score_splits: str = "train,validation",
    threshold_percentile: float = 0.90,
    split_limits: str | dict[str, int] | None = None,
    persist_cache: bool = True,
    swap_pair_manifest: str | Path | None = None,
) -> AnalysisResult:
    set_seed(seed)
    selected = _expand(analyses)
    bundle = AnalysisBundle(data_root)
    resolved = load_checkpoint(checkpoint)
    # UMAP is part of routed representation reports, but an unrouted unit-only
    # report does not construct L/P embeddings.
    if "selectivity" in selected and resolved.capabilities["unit_routes"]:
        _load_umap()
    for name in selected:
        bundle.require(name)
        if name in {"factor_metrics", "clustering", "similarity", "geometry", "causal", "swap"} and not resolved.capabilities["unit_routes"]:
            raise AnalysisError(f"Checkpoint architecture does not support unit-route analysis '{name}'.")
        if name in {"causal", "swap"} and not resolved.capabilities[name]:
            raise AnalysisError(f"Checkpoint architecture does not support '{name}'.")
    device = device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    if profile not in {"full", "quick"}:
        raise AnalysisError("profile must be 'full' or 'quick'.")
    parsed_split_limits = parse_split_limits(split_limits)
    if profile == "quick" and parsed_split_limits:
        raise AnalysisError("--split-limits cannot be combined with --profile quick.")
    factor_scope = str(factor_scope or "speaker_phone").lower().replace("-", "_")
    if factor_scope not in {"speaker_phone", "broad"}:
        raise AnalysisError("--factor-scope must be 'speaker_phone' or 'broad'.")
    atlas_asset_set = {x.strip().lower() for x in atlas_assets.split(",") if x.strip()}
    if "all" in atlas_asset_set:
        atlas_asset_set = {"spectrograms", "audio", "traces"}
    bad_assets = atlas_asset_set - {"none", "spectrograms", "audio", "traces"}
    if bad_assets:
        raise AnalysisError(
            f"Unknown atlas assets {sorted(bad_assets)}; valid choices are none, "
            "spectrograms, audio, traces, all."
        )
    if "none" in atlas_asset_set and len(atlas_asset_set) > 1:
        raise AnalysisError("--atlas-assets=none cannot be combined with other assets.")
    if atlas_asset_set == {"none"}:
        atlas_asset_set = set()

    root = Path(__file__).resolve().parent
    dataset_paths = [bundle.spec.manifest_path]
    if bundle.spec.alignments_path is not None:
        dataset_paths.append(bundle.spec.alignments_path)
    dataset_key = fingerprint(dataset_paths, bundle.spec.raw)
    output = Path(output_dir) if output_dir else root / "results" / Path(checkpoint).stem / dataset_key
    output = output.resolve(); (output / "tables").mkdir(parents=True, exist_ok=True)
    cache_dir = root / "cache" / Path(checkpoint).stem / dataset_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle.write_validation_report(output / "data_validation.json")

    model = calibrate(resolved, bundle, device)
    cache = extract(
        resolved, bundle, model, cache_dir, device, profile, seed,
        split_limits=parsed_split_limits,
        compute_acoustics=factor_scope == "broad",
        persist_cache=bool(persist_cache) and selected != ["deadness"],
    )
    write_json(output / "resolved_model.json", {
        "checkpoint": str(resolved.checkpoint), "format": resolved.source_format,
        "config": resolved.config, "capabilities": resolved.capabilities,
        "warnings": resolved.warnings,
    })

    tables: dict[str, pd.DataFrame] = {}
    summaries: dict = {}
    health = profiles = scores = causal_table = None
    if "deadness" in selected:
        deadness, deadness_summary = deadness_analysis(cache, resolved, output)
        tables["deadness"] = deadness
        summaries["deadness"] = deadness_summary
        # Reuse the established deadness card/caption in the HTML report,
        # without pretending that the full unit-health analysis was run.
        summaries["health"] = deadness_summary
    if "health" in selected:
        health, summaries["health"] = health_analysis(cache, resolved, output); tables["health"] = health
    if "selectivity" in selected:
        scores, profiles, summaries["selectivity"] = selectivity_analysis(
            cache, bundle, output, factor_scope=factor_scope, score_splits=score_splits)
        if health is not None:
            profiles = profiles.merge(health[["unit", "frame_frequency"]], on="unit", how="left")
        tables["scores"], tables["profiles"] = scores, profiles
        unit_scores, summaries["phone_speaker_scores"] = phone_speaker_unit_scores(
            cache, health, profiles, scores, output,
            threshold_percentile=threshold_percentile,
        )
        tables["phone_speaker_scores"] = unit_scores
        score_keep = ["unit", "PhoneScore", "SpeakerScore", "D", "M", "category"]
        profiles = profiles.drop(
            columns=[c for c in score_keep if c != "unit" and c in profiles.columns], errors="ignore"
        ).merge(unit_scores[score_keep], on="unit", how="left")
        tables["profiles"] = profiles
        phone_confusion, selected_phone_units, summaries["phone_unit_confusion"] = phone_unit_confusion(
            cache, bundle, output, selection_splits=score_splits, evaluation_splits="test",
        )
        tables["phone_unit_confusion"] = phone_confusion
        tables["selected_phone_units"] = selected_phone_units
        if resolved.capabilities["unit_routes"]:
            phone_embedding, speaker_embedding, separation, probe_confusion, summaries["route_representation_embeddings"] = (
                route_representation_embeddings(
                    cache, bundle, output, seed=seed,
                    min_utts_per_speaker=2 if profile == "quick" else 20,
                )
            )
            tables["phone_embedding"] = phone_embedding
            tables["speaker_embedding"] = speaker_embedding
            tables["representation_separation"] = separation
            tables["probe_confusion"] = probe_confusion
            geometry_pairs, geometry_summary, summaries["classifier_free_geometry"] = (
                classifier_free_route_geometry(cache, bundle, output, seed=seed)
            )
            tables["classifier_free_geometry_pairs"] = geometry_pairs
            tables["classifier_free_geometry_summary"] = geometry_summary
            disent, leaky, route_summary, summaries["disentanglement"] = disentanglement_tables(
                health, profiles, scores, output,
                focus=str(resolved.config.get("analysis_focus", "speaker_content")),
            )
            tables["disentanglement"] = disent
            tables["leaky"] = leaky
            tables["route_summary"] = route_summary
        else:
            baseline_table, summaries["unrouted_unit_summary"] = unrouted_unit_summary(
                unit_scores, output,
            )
            tables["unrouted_unit_summary"] = baseline_table
    if "clustering" in selected:
        clustered, summaries["clustering"] = clustering_analysis(cache, profiles, output, seed)
        tables["clusters"] = clustered
    if "similarity" in selected:
        summaries["similarity"] = similarity_analysis(cache, bundle, output, seed)
    if "geometry" in selected:
        geometry, summaries["geometry"] = geometry_analysis(cache, resolved, health, output)
        tables["geometry"] = geometry
    if "factor_metrics" in selected:
        factor_metrics, dci_importance, factor_metric_repeats, summaries["factor_metrics"] = (
            speech_factor_metrics(cache, bundle, output, seed=seed, quick=profile == "quick")
        )
        tables["factor_metrics"] = factor_metrics
        tables["dci_importance"] = dci_importance
        tables["factor_metric_repeats"] = factor_metric_repeats

    suite = None
    if "causal" in selected or "swap" in selected:
        suite = train_evaluators(cache, bundle, cache_dir, resolved, seed)
    if "causal" in selected:
        causal_table, summaries["causal"] = causal_analysis(
            cache, bundle, resolved, suite, profiles, output, seed, profile == "quick")
        tables["causal"] = causal_table
    if "swap" in selected:
        swaps, swap_summary, swap_contrasts, swap_grid, summaries["swap"] = swap_analysis(
            cache, bundle, resolved, suite, output, seed, profile == "quick",
            pair_manifest=Path(swap_pair_manifest) if swap_pair_manifest else None,
        )
        tables["swap"] = swaps
        tables["swap_summary"] = swap_summary
        tables["swap_contrasts"] = swap_contrasts
        tables["swap_grid"] = swap_grid
    if "atlas" in selected and atlas_asset_set:
        examples = top_examples(cache, bundle)
        build_atlas(
            output, cache, bundle, health, examples, scores, causal_table,
            include_spectrograms="spectrograms" in atlas_asset_set,
            include_audio="audio" in atlas_asset_set,
            include_traces="traces" in atlas_asset_set,
        )
        tables["examples"] = examples

    plots = make_plots(output, tables)
    report = build_report(
        output, resolved, selected, summaries, tables, resolved.warnings, plots,
        profile=profile,
    )
    write_json(output / "summary.json", summaries)
    write_json(output / "run_manifest.json", {
        "checkpoint": str(Path(checkpoint).resolve()), "data": str(Path(data_root).resolve()),
        "analyses": selected, "profile": profile, "device": device, "seed": seed,
        "atlas_assets": sorted(atlas_asset_set),
        "factor_scope": factor_scope,
        "score_splits": score_splits,
        "threshold_percentile": float(threshold_percentile),
        "split_limits": parsed_split_limits,
        "persist_cache_requested": bool(persist_cache),
        "swap_pair_manifest": str(Path(swap_pair_manifest).resolve()) if swap_pair_manifest else None,
        "python": platform.python_version(), "torch": torch.__version__,
        "cache": str(cache.path) if cache.path.exists() else None,
        "cache_persisted": bool(cache.path.exists()), "report": str(report),
    })
    artifacts = {"report": report, "summary": output / "summary.json"}
    if cache.path.exists():
        artifacts["cache"] = cache.path
    return AnalysisResult(output, selected, artifacts, resolved.warnings, summaries)
