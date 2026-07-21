from __future__ import annotations

import argparse
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .checkpoint import load_checkpoint, route_information
from .extraction import _encode_sparse
from .utils import AnalysisError, write_json


ROUTE_NAMES = {0: "L", 1: "P", 2: "U", -1: "shared"}
ROUTE_COLORS = {"L": "#277da1", "P": "#f8961e", "U": "#adb5bd", "shared": "#6c757d"}
CATEGORY_ORDER = ("other", "phone", "speaker", "entangled")


@dataclass(frozen=True)
class TimelinePoint:
    step: int
    checkpoint: Path
    label: str
    is_final: bool


@dataclass
class SharedSample:
    h: np.ndarray
    phones: np.ndarray
    utterance_index: np.ndarray
    utterance_ids: np.ndarray
    speakers: np.ndarray
    splits: np.ndarray
    source_frames: np.ndarray
    source_cache: Path


def discover_timeline(checkpoint_dir: str | Path, stride: int = 2000) -> list[TimelinePoint]:
    """Select earliest, then fixed-stride snapshots, replacing the last step by final.pt."""
    root = Path(checkpoint_dir).resolve()
    if not root.is_dir():
        raise AnalysisError(f"Checkpoint directory does not exist: {root}")
    by_step: dict[int, Path] = {}
    for path in root.glob("stage2_step*.pt"):
        match = re.fullmatch(r"stage2_step(\d+)\.pt", path.name)
        if match:
            by_step[int(match.group(1))] = path
    if not by_step:
        raise AnalysisError(f"No stage2_step*.pt checkpoints found in {root}.")
    final = root / "final.pt"
    if not final.exists():
        raise AnalysisError(f"Missing final checkpoint: {final}")
    stride = int(stride)
    if stride <= 0:
        raise AnalysisError("Timeline stride must be positive.")
    first, last = min(by_step), max(by_step)
    wanted = range(first, last, stride)
    points = [
        TimelinePoint(step=step, checkpoint=by_step[step], label=f"{step // 1000}k", is_final=False)
        for step in wanted if step in by_step
    ]
    points.append(
        TimelinePoint(step=last, checkpoint=final, label=f"final ({last // 1000}k)", is_final=True)
    )
    return points


def _parse_families(values: Iterable[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise AnalysisError("Each --family must have the form LABEL=CHECKPOINT_DIRECTORY.")
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise AnalysisError("Family labels cannot be empty.")
        parsed.append((label, Path(path).expanduser().resolve()))
    if not parsed:
        raise AnalysisError("At least one --family is required.")
    if len({label for label, _ in parsed}) != len(parsed):
        raise AnalysisError("Family labels must be unique.")
    return parsed


def _manifest_labels(data_root: Path, utterance_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    manifest = data_root / "utterances.csv"
    if not manifest.exists():
        manifest = data_root / "utterances.parquet"
    if not manifest.exists():
        raise AnalysisError(f"No utterance manifest found in {data_root}.")
    table = pd.read_parquet(manifest) if manifest.suffix == ".parquet" else pd.read_csv(manifest)
    required = {"utterance_id", "speaker_id", "split"}
    missing = required - set(table.columns)
    if missing:
        raise AnalysisError(f"Manifest is missing trajectory labels: {sorted(missing)}")
    table["utterance_id"] = table["utterance_id"].astype(str)
    labels = table.set_index("utterance_id")
    requested = pd.Index(utterance_ids.astype(str))
    if not requested.isin(labels.index).all():
        raise AnalysisError("Shared feature cache contains utterances absent from the data manifest.")
    selected = labels.loc[requested]
    return selected["speaker_id"].astype(str).to_numpy(), selected["split"].astype(str).to_numpy()


def load_shared_sample(
    cache_path: str | Path,
    data_root: str | Path,
    *,
    max_frames: int | None = None,
    seed: int = 42,
) -> SharedSample:
    """Read only the raw-SPEAR sample and its labels from a large compressed cache."""
    cache_path = Path(cache_path).resolve()
    data_root = Path(data_root).resolve()
    if not cache_path.exists():
        raise AnalysisError(f"Shared feature cache does not exist: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as archive:
        required = {"h_sample", "h_sample_frames", "phones", "offsets", "lengths", "utterance_ids"}
        missing = required - set(archive.files)
        if missing:
            raise AnalysisError(f"Shared cache is missing arrays: {sorted(missing)}")
        h = archive["h_sample"]
        source_frames = archive["h_sample_frames"].astype(np.int64)
        offsets = archive["offsets"].astype(np.int64)
        lengths = archive["lengths"].astype(np.int64)
        utterance_ids = archive["utterance_ids"].astype(str)
        all_phones = archive["phones"]
        phones = all_phones[source_frames].astype("U32")
    utterance_index = np.searchsorted(offsets, source_frames, side="right") - 1
    valid = (utterance_index >= 0) & (
        source_frames < offsets[np.maximum(utterance_index, 0)] + lengths[np.maximum(utterance_index, 0)]
    )
    if not valid.all():
        raise AnalysisError("h_sample_frames contains positions outside the cached utterances.")
    speakers, splits = _manifest_labels(data_root, utterance_ids)

    if max_frames is not None and 0 < int(max_frames) < len(h):
        # Preserve broad utterance coverage rather than inheriting a contiguous slice.
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(h), size=int(max_frames), replace=False))
        h, phones = h[chosen], phones[chosen]
        source_frames, utterance_index = source_frames[chosen], utterance_index[chosen]
    return SharedSample(
        h=h, phones=phones, utterance_index=utterance_index,
        utterance_ids=utterance_ids, speakers=speakers, splits=splits,
        source_frames=source_frames, source_cache=cache_path,
    )


def _validate_domain(resolved, sample: SharedSample, reference: dict | None) -> dict:
    current = {
        "D": int(resolved.config["D"]),
        "K": int(resolved.config["K"]),
        "spear_model_id": str(resolved.config.get("spear_model_id", "")),
        "spear_layernorm": bool(resolved.config.get("spear_layernorm", False)),
    }
    if sample.h.ndim != 2 or sample.h.shape[1] != current["D"]:
        raise AnalysisError(
            f"Shared SPEAR sample has shape {sample.h.shape}, but {resolved.checkpoint} expects D={current['D']}."
        )
    if reference is not None:
        for key in ("D", "K", "spear_model_id", "spear_layernorm"):
            if current[key] != reference[key]:
                raise AnalysisError(
                    f"Incompatible shared representation domain at {resolved.checkpoint}: "
                    f"{key}={current[key]!r}, expected {reference[key]!r}."
                )
    return current


@torch.inference_mode()
def encode_sample(
    sample: SharedSample,
    checkpoint: str | Path,
    *,
    device: str,
    batch_size: int = 512,
    reference_domain: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    resolved = load_checkpoint(checkpoint)
    domain = _validate_domain(resolved, sample, reference_domain)
    state = resolved.state
    sae = SimpleNamespace(
        enc_weight=state["sae.enc_weight"].detach().to(device=device, dtype=torch.float32),
        b_pre=state["sae.b_pre"].detach().to(device=device, dtype=torch.float32),
    )
    model = SimpleNamespace(sae=sae)
    all_indices: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    for start in range(0, len(sample.h), int(batch_size)):
        h = torch.as_tensor(
            sample.h[start:start + int(batch_size)], device=device, dtype=torch.float32,
        )
        indices, values = _encode_sparse(h, model, resolved)
        all_indices.append(indices.detach().cpu().numpy().astype(np.uint16))
        all_values.append(values.detach().cpu().numpy().astype(np.float16))
    route, route_probability = route_information(resolved)
    del resolved, state, model, sae
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return np.concatenate(all_indices), np.concatenate(all_values), route, {
        **domain,
        "route_probability": route_probability,
    }


def phone_scores(
    indices: np.ndarray,
    phones: np.ndarray,
    frame_mask: np.ndarray,
    K: int,
    *,
    min_count: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    phones = np.asarray(phones).astype(str)
    valid = np.asarray(frame_mask, dtype=bool) & (np.char.upper(phones) != "<UNALIGNED>")
    labels = phones[valid]
    selected = indices[valid]
    levels, counts = np.unique(labels, return_counts=True)
    keep = counts >= int(min_count)
    levels, counts = levels[keep], counts[keep]
    if not len(levels):
        return np.zeros(K), np.zeros(K), np.full(K, "", dtype="U32"), []
    total = np.bincount(selected.reshape(-1), minlength=K).astype(np.float64)
    effects = np.zeros((len(levels), K), dtype=np.float32)
    N = len(selected)
    for i, (level, count) in enumerate(zip(levels, counts)):
        inside = labels == level
        active_in = np.bincount(selected[inside].reshape(-1), minlength=K).astype(np.float64)
        tpr = active_in / float(count)
        fpr = (total - active_in) / float(max(N - int(count), 1))
        effects[i] = np.clip(tpr - fpr, 0.0, 1.0)
    strongest = effects.argmax(axis=0)
    maximum = effects[strongest, np.arange(K)]
    mean = effects.mean(axis=0)
    preferred = levels[strongest].astype("U32")
    preferred[maximum <= 0] = ""
    return (0.5 * maximum + 0.5 * mean), maximum, preferred, levels.astype(str).tolist()


def _utterance_means(
    indices: np.ndarray,
    values: np.ndarray,
    utterance_index: np.ndarray,
    n_utterances: int,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.zeros((n_utterances, K), dtype=np.float32)
    rows = np.repeat(utterance_index.astype(np.int64), indices.shape[1])
    np.add.at(pooled, (rows, indices.reshape(-1).astype(np.int64)), values.reshape(-1).astype(np.float32))
    counts = np.bincount(utterance_index.astype(np.int64), minlength=n_utterances).astype(np.float32)
    pooled /= np.maximum(counts[:, None], 1.0)
    return pooled, counts


def speaker_scores(
    indices: np.ndarray,
    values: np.ndarray,
    sample: SharedSample,
    K: int,
    *,
    score_splits: tuple[str, ...] = ("train", "val"),
    min_count: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    pooled, sample_counts = _utterance_means(
        indices, values, sample.utterance_index, len(sample.utterance_ids), K,
    )
    best = np.zeros(K, dtype=np.float32)
    preferred = np.full(K, "", dtype="U64")
    for split in score_splits:
        subset = (sample.splits == split) & (sample_counts > 0)
        x = pooled[subset]
        y = sample.speakers[subset]
        if len(x) < 2:
            continue
        total_sum = x.sum(axis=0, dtype=np.float64)
        total_sum2 = np.square(x, dtype=np.float64).sum(axis=0, dtype=np.float64)
        std = np.sqrt(np.maximum(total_sum2 / len(x) - (total_sum / len(x)) ** 2, 0.0))
        levels, counts = np.unique(y, return_counts=True)
        for level, count in zip(levels, counts):
            if count < int(min_count) or count == len(x):
                continue
            inside = y == level
            sum_in = x[inside].sum(axis=0, dtype=np.float64)
            mean_in = sum_in / count
            mean_out = (total_sum - sum_in) / (len(x) - count)
            prevalence = count / len(x)
            r = np.divide(
                (mean_in - mean_out) * math.sqrt(prevalence * (1.0 - prevalence)),
                std, out=np.zeros(K, dtype=np.float64), where=std > 1e-12,
            )
            r = np.clip(r, 0.0, 1.0)
            improve = r > best
            best[improve] = r[improve]
            preferred[improve] = str(level)
    return best, preferred


def categorize(phone: np.ndarray, speaker: np.ndarray, percentile: float = 0.90):
    phone_positive = phone[phone > 0]
    speaker_positive = speaker[speaker > 0]
    phone_threshold = float(np.quantile(phone_positive, percentile)) if len(phone_positive) else float("inf")
    speaker_threshold = (
        float(np.quantile(speaker_positive, percentile)) if len(speaker_positive) else float("inf")
    )
    high_phone = (phone > 0) & (phone >= phone_threshold)
    high_speaker = (speaker > 0) & (speaker >= speaker_threshold)
    category = np.full(len(phone), "other", dtype="U16")
    category[high_phone & ~high_speaker] = "phone"
    category[~high_phone & high_speaker] = "speaker"
    category[high_phone & high_speaker] = "entangled"
    return category, phone_threshold, speaker_threshold


def snapshot_scores(
    sample: SharedSample,
    indices: np.ndarray,
    values: np.ndarray,
    route: np.ndarray,
    *,
    score_splits: tuple[str, ...] = ("train", "val"),
    percentile: float = 0.90,
) -> tuple[pd.DataFrame, dict]:
    K = len(route)
    frame_splits = sample.splits[sample.utterance_index]
    frame_mask = np.isin(frame_splits, np.asarray(score_splits))
    phone, phone_max, preferred_phone, phone_levels = phone_scores(
        indices, sample.phones, frame_mask, K,
    )
    speaker, preferred_speaker = speaker_scores(
        indices, values, sample, K, score_splits=score_splits,
    )
    selected_count = np.bincount(indices.reshape(-1), minlength=K)
    observed = selected_count > 0
    category, phone_threshold, speaker_threshold = categorize(phone, speaker, percentile)
    table = pd.DataFrame({
        "unit": np.arange(K, dtype=int),
        "route_id": route.astype(int),
        "route": [ROUTE_NAMES.get(int(value), str(int(value))) for value in route],
        "observed_on_shared_sample": observed,
        "shared_sample_frame_frequency": selected_count / max(len(indices), 1),
        "PhoneScore": phone,
        "phone_positive_max": phone_max,
        "preferred_phone": preferred_phone,
        "SpeakerScore": speaker,
        "preferred_speaker": preferred_speaker,
        "category": category,
    })
    table["state"] = table["route"] + "-" + table["category"]
    table.loc[~table["observed_on_shared_sample"], "state"] = (
        table.loc[~table["observed_on_shared_sample"], "route"] + "-unobserved"
    )
    summary = {
        "phone_threshold": phone_threshold,
        "speaker_threshold": speaker_threshold,
        "phone_levels": phone_levels,
        "observed_units": int(observed.sum()),
        "unobserved_units": int((~observed).sum()),
    }
    return table, summary


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else float("nan")


def trajectory_metrics(tables: list[pd.DataFrame], points: list[TimelinePoint]) -> pd.DataFrame:
    final_route = tables[-1]["route_id"].to_numpy(int)
    rows = []
    previous_route: np.ndarray | None = None
    for table, point in zip(tables, points):
        route = table["route_id"].to_numpy(int)
        observed = table["observed_on_shared_sample"].to_numpy(bool)
        row: dict[str, object] = {
            "step": point.step, "checkpoint_label": point.label,
            "checkpoint": str(point.checkpoint), "is_final": point.is_final,
            "observed_units": int(observed.sum()),
            "observed_fraction": float(observed.mean()),
            "route_match_to_final": float((route == final_route).mean()),
            "route_turnover_from_previous": (
                float((route != previous_route).mean()) if previous_route is not None else float("nan")
            ),
        }
        for route_id, name in ((0, "L"), (1, "P"), (2, "U")):
            members = route == route_id
            active = members & observed
            row[f"{name}_assigned_units"] = int(members.sum())
            row[f"{name}_observed_units"] = int(active.sum())
            row[f"{name}_observed_fraction"] = float(active.sum() / max(members.sum(), 1))
            row[f"{name}_jaccard_to_final"] = _jaccard(members, final_route == route_id)
            row[f"{name}_mean_phone_score"] = float(table.loc[active, "PhoneScore"].mean()) if active.any() else 0.0
            row[f"{name}_mean_speaker_score"] = float(table.loc[active, "SpeakerScore"].mean()) if active.any() else 0.0
            for category in CATEGORY_ORDER:
                row[f"{name}_{category}_units"] = int((active & table["category"].eq(category).to_numpy()).sum())
        row["phone_L_minus_P"] = float(row["L_mean_phone_score"] - row["P_mean_phone_score"])
        row["speaker_P_minus_L"] = float(row["P_mean_speaker_score"] - row["L_mean_speaker_score"])
        rows.append(row)
        previous_route = route
    return pd.DataFrame(rows)


def _plot_scatter(tables: list[pd.DataFrame], points: list[TimelinePoint], path: Path) -> None:
    import matplotlib.pyplot as plt

    columns = 3
    rows = math.ceil(len(tables) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13.2, 4.0 * rows), squeeze=False, sharex=True, sharey=True)
    active_values = pd.concat([t[t.observed_on_shared_sample] for t in tables], ignore_index=True)
    xmax = max(float(active_values.PhoneScore.quantile(.997)) * 1.08, .01)
    ymax = max(float(active_values.SpeakerScore.quantile(.997)) * 1.08, .01)
    for ax, table, point in zip(axes.flat, tables, points):
        active = table[table.observed_on_shared_sample]
        for route_name in ("L", "P", "U", "shared"):
            subset = active[active.route == route_name]
            if len(subset):
                ax.scatter(
                    subset.PhoneScore, subset.SpeakerScore, s=6, alpha=.22,
                    color=ROUTE_COLORS[route_name], edgecolors="none", rasterized=True,
                    label=route_name,
                )
        ax.axvline(float(table.PhoneScore[table.PhoneScore > 0].quantile(.90)), color="#495057", lw=.7, ls=":")
        ax.axhline(float(table.SpeakerScore[table.SpeakerScore > 0].quantile(.90)), color="#495057", lw=.7, ls=":")
        ax.set_title(point.label, fontweight="bold")
        ax.grid(color="#e9ecef", linewidth=.6)
        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    for ax in axes.flat[len(tables):]:
        ax.axis("off")
    for ax in axes[-1]:
        if ax.axison:
            ax.set_xlabel("phone association")
    for ax in axes[:, 0]:
        ax.set_ylabel("speaker association")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(.5, .982),
        ncol=max(len(labels), 1), frameon=False,
    )
    fig.suptitle("Unit organisation on one fixed SPEAR sample", fontsize=16, y=1.012)
    fig.tight_layout(rect=(0, 0, 1, .945))
    fig.savefig(path.with_suffix(".png"), dpi=210, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_metrics(metrics: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    x = metrics.step.to_numpy(float) / 1000.0
    colors = {"L": ROUTE_COLORS["L"], "P": ROUTE_COLORS["P"], "U": ROUTE_COLORS["U"]}
    for name in ("L", "P", "U"):
        if metrics[f"{name}_assigned_units"].max() > 0:
            axes[0, 0].plot(x, metrics[f"{name}_assigned_units"], marker="o", color=colors[name], label=name)
            axes[0, 1].plot(x, metrics[f"{name}_observed_fraction"], marker="o", color=colors[name], label=name)
    axes[1, 0].plot(x, metrics.phone_L_minus_P, marker="o", lw=2.2, color="#277da1", label="phone: L minus P")
    axes[1, 0].plot(x, metrics.speaker_P_minus_L, marker="o", lw=2.2, color="#f8961e", label="speaker: P minus L")
    axes[1, 0].axhline(0, color="#495057", lw=.8)
    for name in ("L", "P", "U"):
        if metrics[f"{name}_assigned_units"].max() > 0:
            axes[1, 1].plot(x, metrics[f"{name}_jaccard_to_final"], marker="o", color=colors[name], label=f"{name} Jaccard")
    axes[1, 1].plot(x, metrics.route_match_to_final, marker="s", color="#6f42c1", ls="--", label="all-unit match")
    titles = (
        "Assigned route capacity", "Observed coverage on shared sample",
        "Desired route-association contrasts", "Route stability relative to final",
    )
    ylabels = ("units", "fraction", "score difference", "similarity")
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("training step (thousands)"); ax.set_ylabel(ylabel)
        ax.grid(color="#e9ecef", linewidth=.7); ax.legend(frameon=False, fontsize=8)
    axes[0, 1].set_ylim(0, 1.03); axes[1, 1].set_ylim(0, 1.03)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_states(tables: list[pd.DataFrame], points: list[TimelinePoint], path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    labels = [
        f"{route}-{category}" for route in ("L", "P", "U", "shared")
        for category in ("unobserved",) + CATEGORY_ORDER
    ]
    present = {state for table in tables for state in table.state.astype(str)}
    labels = [label for label in labels if label in present]
    codes = {label: i for i, label in enumerate(labels)}
    matrix = np.column_stack([
        table.state.astype(str).map(codes).to_numpy(int) for table in tables
    ])
    # Final state is the primary ordering; complete trajectory breaks ties.
    keys = [matrix[:, column] for column in range(matrix.shape[1])]
    order = np.lexsort(keys)
    palette = []
    for label in labels:
        route, category = label.split("-", 1)
        base = {
            "L": {"unobserved": "#e5f1f6", "other": "#b7dbe8", "phone": "#1676a3", "speaker": "#7b2cbf", "entangled": "#d1495b"},
            "P": {"unobserved": "#fff3dc", "other": "#ffe0a6", "phone": "#e9c46a", "speaker": "#e76f00", "entangled": "#d1495b"},
            "U": {"unobserved": "#f1f3f5", "other": "#ced4da", "phone": "#8ecae6", "speaker": "#c77dff", "entangled": "#d1495b"},
            "shared": {"unobserved": "#f1f3f5", "other": "#ced4da", "phone": "#8ecae6", "speaker": "#c77dff", "entangled": "#d1495b"},
        }
        palette.append(base[route][category])
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(-.5, len(labels) + .5), len(labels))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.05 * len(points)), 8.5))
    ax.imshow(matrix[order], aspect="auto", interpolation="nearest", cmap=cmap, norm=norm, rasterized=True)
    ax.set(xticks=np.arange(len(points)), xticklabels=[point.label for point in points],
           xlabel="checkpoint", ylabel="units sorted by final organisation",
           title="Unit fate across training")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(
        handles=[Patch(facecolor=color, label=label) for label, color in zip(labels, palette)],
        loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _sample_phone_coverage(sample: SharedSample, min_count: int = 20) -> dict:
    frame_splits = sample.splits[sample.utterance_index]
    aligned = np.char.upper(sample.phones.astype(str)) != "<UNALIGNED>"
    mask = np.isin(frame_splits, np.asarray(["train", "val"])) & aligned
    levels, counts = np.unique(sample.phones[mask].astype(str), return_counts=True)
    count_map = {str(level): int(count) for level, count in zip(levels, counts)}
    return {
        "observed_phone_levels": len(count_map),
        "minimum_frames": int(min_count),
        "included_phone_levels": int(sum(count >= min_count for count in count_map.values())),
        "excluded_rare_phones": {
            level: count for level, count in count_map.items() if count < min_count
        },
        "train_validation_sample_counts": count_map,
    }


def _phone_coverage_text(sample: SharedSample) -> str:
    coverage = _sample_phone_coverage(sample)
    rare = ", ".join(
        f"{html.escape(phone)} ({count} frames)"
        for phone, count in coverage["excluded_rare_phones"].items()
    ) or "none"
    return (
        f"All {coverage['observed_phone_levels']} phones occur in the shared sample; "
        f"{coverage['included_phone_levels']} meet the minimum of "
        f"{coverage['minimum_frames']} train/validation frames used in PhoneScore. "
        f"Rare phones excluded from that score: {rare}."
    )


def _family_report(label: str, output: Path, metrics: pd.DataFrame, sample: SharedSample) -> Path:
    final = metrics.iloc[-1]
    rows = "".join(
        f"<tr><td>{html.escape(str(row.checkpoint_label))}</td>"
        f"<td>{int(row.observed_units):,}</td>"
        f"<td>{float(row.phone_L_minus_P):.4f}</td>"
        f"<td>{float(row.speaker_P_minus_L):.4f}</td>"
        f"<td>{float(row.route_match_to_final):.3f}</td></tr>"
        for _, row in metrics.iterrows()
    )
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(label)} trajectory</title>
<style>body{{font-family:Inter,system-ui,sans-serif;background:#f4f6fa;color:#172033;margin:0;padding:34px;line-height:1.45}}
.panel{{background:white;border-radius:15px;padding:22px;margin:20px 0;box-shadow:0 3px 15px #1d2b4b16}} img{{max-width:100%;height:auto}}
table{{border-collapse:collapse;width:100%}} th,td{{padding:8px;border-bottom:1px solid #e9ecef;text-align:left}} .note{{color:#495057}}</style></head><body>
<h1>{html.escape(label)}: unit organisation through training</h1>
<p class='note'>All checkpoints are applied to the same {len(sample.h):,} frozen SPEAR frames from {len(sample.utterance_ids):,} utterances. Association labels use train+validation only. {_phone_coverage_text(sample)} “Unobserved” means absent from this shared sample; it is not the training dead-unit definition.</p>
<div class='panel'><h2>Association landscape</h2><img src='../plots/unit_association_snapshots.png'><p>Each point is an observed SAE unit. A cleaner double dissociation moves L units toward stronger phone association and P units toward stronger speaker association.</p></div>
<div class='panel'><h2>Capacity, evidence and route stability</h2><img src='../plots/trajectory_metrics.png'><p>Positive phone L−P and speaker P−L contrasts support the intended organisation. Jaccard and all-unit agreement show when learned membership stabilises.</p></div>
<div class='panel'><h2>Individual unit fate</h2><img src='../plots/unit_fate_heatmap.png'><p>Rows follow unit identity only within this training run. Colour changes show route or selectivity-category transitions.</p></div>
<div class='panel'><h2>Checkpoint summary</h2><table><thead><tr><th>checkpoint</th><th>observed units</th><th>phone L−P</th><th>speaker P−L</th><th>route match to final</th></tr></thead><tbody>{rows}</tbody></table>
<p>Final shared-sample contrasts: phone L−P={float(final.phone_L_minus_P):.4f}; speaker P−L={float(final.speaker_P_minus_L):.4f}.</p></div>
</body></html>"""
    report = output / "report" / "index.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(page, encoding="utf-8")
    return report


def analyze_family(
    label: str,
    checkpoint_dir: Path,
    sample: SharedSample,
    output: Path,
    *,
    device: str,
    stride: int,
    batch_size: int,
    reference_domain: dict | None,
) -> tuple[pd.DataFrame, dict, Path]:
    points = discover_timeline(checkpoint_dir, stride=stride)
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables").mkdir(exist_ok=True); (output / "plots").mkdir(exist_ok=True)
    tables: list[pd.DataFrame] = []
    snapshot_summaries = []
    domain = reference_domain
    for index, point in enumerate(points, start=1):
        print(f"[trajectory] {label}: {point.label} ({index}/{len(points)})", flush=True)
        indices, values, route, metadata = encode_sample(
            sample, point.checkpoint, device=device, batch_size=batch_size,
            reference_domain=domain,
        )
        if domain is None:
            domain = {key: metadata[key] for key in ("D", "K", "spear_model_id", "spear_layernorm")}
        table, summary = snapshot_scores(sample, indices, values, route)
        table.insert(0, "family", label); table.insert(1, "step", point.step)
        table.insert(2, "checkpoint_label", point.label)
        tables.append(table)
        snapshot_summaries.append({"step": point.step, "label": point.label, **summary})
        del indices, values
    metrics = trajectory_metrics(tables, points)
    units = pd.concat(tables, ignore_index=True)
    transition_rows = []
    for previous, current, previous_point, current_point in zip(
        tables[:-1], tables[1:], points[:-1], points[1:],
    ):
        counts = pd.crosstab(previous["state"], current["state"])
        for from_state in counts.index:
            for to_state in counts.columns:
                count = int(counts.loc[from_state, to_state])
                if count:
                    transition_rows.append({
                        "from_step": previous_point.step, "to_step": current_point.step,
                        "from_state": from_state, "to_state": to_state, "units": count,
                    })
    metrics.to_csv(output / "tables" / "trajectory_metrics.csv", index=False)
    units.to_csv(output / "tables" / "unit_trajectories.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(output / "tables" / "state_transitions.csv", index=False)
    _plot_scatter(tables, points, output / "plots" / "unit_association_snapshots")
    _plot_metrics(metrics, output / "plots" / "trajectory_metrics")
    _plot_states(tables, points, output / "plots" / "unit_fate_heatmap")
    write_json(output / "trajectory_manifest.json", {
        "family": label, "checkpoint_dir": str(checkpoint_dir),
        "shared_sample_cache": str(sample.source_cache),
        "shared_sample_frames": len(sample.h), "utterances": len(sample.utterance_ids),
        "score_splits": ["train", "val"], "stride": stride,
        "domain": domain, "snapshots": snapshot_summaries,
        "inactive_label": "unobserved_on_shared_sample_not_training_deadness",
    })
    report = _family_report(label, output, metrics, sample)
    return metrics, domain or {}, report


def _plot_cross_family(results: list[tuple[str, Path, pd.DataFrame]], output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = ("#0b7285", "#7b2cbf", "#e8590c", "#2b8a3e", "#364fc7")
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.5))
    for color, (label, _, metrics) in zip(colors, results):
        x = metrics.step.to_numpy(float) / 1000.0
        style = {"marker": "o", "linewidth": 2.1, "markersize": 5.5, "color": color, "label": label}
        axes[0, 0].plot(x, metrics.observed_fraction, **style)
        axes[0, 1].plot(x, metrics.L_observed_fraction, **style)
        axes[1, 0].plot(x, metrics.phone_L_minus_P, **style)
        axes[1, 1].plot(x, metrics.speaker_P_minus_L, **style)
    panels = (
        (axes[0, 0], "All-unit observed coverage", "observed fraction", (0, 1.03)),
        (axes[0, 1], "L-route observed coverage", "observed fraction", (0, 1.03)),
        (axes[1, 0], "Phone association contrast", "L minus P", None),
        (axes[1, 1], "Speaker association contrast", "P minus L", None),
    )
    for ax, title, ylabel, ylim in panels:
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("training step (thousands)"); ax.set_ylabel(ylabel)
        ax.grid(color="#e9ecef", linewidth=.7)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.axhline(0, color="#495057", linewidth=.8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(.5, -.01),
        ncol=len(labels), frameon=False,
    )
    fig.suptitle("Cross-family unit-organisation trajectories", fontsize=16, y=.995)
    fig.tight_layout(rect=(0, .055, 1, .96))
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / "cross_family_trajectories.png", dpi=220, bbox_inches="tight")
    fig.savefig(plots / "cross_family_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)


def _root_report(results: list[tuple[str, Path, pd.DataFrame]], output: Path, sample: SharedSample) -> Path:
    rows = []
    links = []
    for label, family_dir, metrics in results:
        final = metrics.iloc[-1]
        rows.append({
            "family": label, "first_step": int(metrics.step.min()), "final_step": int(metrics.step.max()),
            "snapshots": len(metrics), "final_observed_units": int(final.observed_units),
            "final_phone_L_minus_P": float(final.phone_L_minus_P),
            "final_speaker_P_minus_L": float(final.speaker_P_minus_L),
        })
        links.append(f"<li><a href='../{html.escape(family_dir.name)}/report/index.html'>{html.escape(label)}</a></li>")
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "tables" / "cross_family_final_summary.csv", index=False)
    _plot_cross_family(results, output)
    table_rows = "".join(
        f"<tr><td>{html.escape(row.family)}</td><td>{row.first_step // 1000}k</td><td>{row.final_step // 1000}k</td>"
        f"<td>{row.snapshots}</td><td>{row.final_observed_units:,}</td>"
        f"<td>{row.final_phone_L_minus_P:.4f}</td><td>{row.final_speaker_P_minus_L:.4f}</td></tr>"
        for row in summary.itertuples(index=False)
    )
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>SAE organisation trajectories</title>
<style>body{{font-family:Inter,system-ui,sans-serif;background:#f4f6fa;color:#172033;margin:0;padding:34px;line-height:1.45}} .panel{{background:white;border-radius:15px;padding:22px;margin:20px 0;box-shadow:0 3px 15px #1d2b4b16}} table{{border-collapse:collapse;width:100%}} th,td{{padding:8px;border-bottom:1px solid #e9ecef;text-align:left}}</style></head><body>
<h1>SAE unit organisation through training</h1><p>This longitudinal diagnostic applies every SAE checkpoint to exactly the same {len(sample.h):,} raw SPEAR frames from {len(sample.utterance_ids):,} utterances. It supports within-run unit tracking without repeating audio extraction. {_phone_coverage_text(sample)}</p>
<div class='panel'><h2>Family reports</h2><ul>{''.join(links)}</ul></div>
<div class='panel'><h2>Cross-family trajectories</h2><img src='../plots/cross_family_trajectories.png' style='max-width:100%;height:auto'><p>Coverage is descriptive on the fixed sample, not the training dead-unit statistic. Positive phone L−P and speaker P−L values indicate the intended route organisation.</p></div>
<div class='panel'><h2>Final shared-sample summary</h2><table><thead><tr><th>family</th><th>first</th><th>final</th><th>snapshots</th><th>observed units</th><th>phone L−P</th><th>speaker P−L</th></tr></thead><tbody>{table_rows}</tbody></table></div>
<div class='panel'><h2>Interpretation boundary</h2><p>Scores use the report’s directional phone AUROC and speaker point-biserial definitions, but only on the fixed 40k-frame diagnostic sample. Full 5k/12k reports remain the source for headline estimates. Unit IDs are comparable through time within one family, never across independently trained families.</p></div>
</body></html>"""
    report = output / "report" / "index.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(page, encoding="utf-8")
    return report


def run_trajectories(
    families: list[tuple[str, Path]],
    *,
    source_cache: Path,
    data_root: Path,
    output: Path,
    device: str,
    stride: int = 2000,
    batch_size: int = 512,
    max_frames: int | None = None,
    seed: int = 42,
) -> Path:
    output = output.resolve()
    (output / "tables").mkdir(parents=True, exist_ok=True)
    sample = load_shared_sample(source_cache, data_root, max_frames=max_frames, seed=seed)
    write_json(output / "trajectory_manifest.json", {
        "shared_sample_cache": str(sample.source_cache),
        "shared_sample_frames": len(sample.h),
        "utterances": len(sample.utterance_ids),
        "score_splits": ["train", "val"],
        "phone_coverage": _sample_phone_coverage(sample),
        "stride": stride,
        "families": [{"label": label, "checkpoint_dir": str(path)} for label, path in families],
        "inactive_label": "unobserved_on_shared_sample_not_training_deadness",
    })
    results = []
    reference_domain: dict | None = None
    for family_index, (label, checkpoint_dir) in enumerate(families, start=1):
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        family_output = output / slug
        print(f"[trajectory] family {family_index}/{len(families)}: {label}", flush=True)
        metrics, domain, _ = analyze_family(
            label, checkpoint_dir, sample, family_output,
            device=device, stride=stride, batch_size=batch_size,
            reference_domain=reference_domain,
        )
        reference_domain = domain
        results.append((label, family_output, metrics))
    report = _root_report(results, output, sample)
    print(f"[trajectory] report: {report}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track routed-SAE unit organisation on one fixed raw-SPEAR sample."
    )
    parser.add_argument("--family", action="append", default=[], metavar="LABEL=CHECKPOINT_DIR")
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--stride", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        run_trajectories(
            _parse_families(args.family), source_cache=args.source_cache,
            data_root=args.data, output=args.output, device=args.device,
            stride=args.stride, batch_size=args.batch_size,
            max_frames=args.max_frames, seed=args.seed,
        )
    except AnalysisError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
