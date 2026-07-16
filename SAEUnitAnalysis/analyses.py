from __future__ import annotations

import html
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bundle import AnalysisBundle
from .extraction import FeatureCache
from .types import ResolvedModel
from .utils import AnalysisError, bh_fdr, bootstrap_ci, write_json, write_rows


ROUTE_NAMES = {-1: "unassigned", 0: "L", 1: "P", 2: "U"}
_UMAP_MODULE = None

_VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
_STOPS = {"P","B","T","D","K","G"}
_FRICATIVES = {"F","V","TH","DH","S","Z","SH","ZH","HH"}
_AFFRICATES = {"CH","JH"}
_NASALS = {"M","N","NG"}
_LIQUIDS = {"L","R"}
_GLIDES = {"W","Y"}
_VOICED = _VOWELS | {"B","D","G","V","DH","Z","ZH","JH","M","N","NG","L","R","W","Y"}
_LABIAL = {"P","B","M","F","V","W"}
_CORONAL = {"T","D","N","S","Z","TH","DH","SH","ZH","CH","JH","L","R"}
_DORSAL = {"K","G","NG","Y"}


def _base_phone(phone: str) -> str:
    return "".join(c for c in str(phone).upper() if not c.isdigit())


def _phone_property(phone: str, prop: str) -> str:
    p = _base_phone(phone)
    if p.startswith("<"): return p
    if prop == "manner":
        for name, group in (("vowel",_VOWELS),("stop",_STOPS),("fricative",_FRICATIVES),
                            ("affricate",_AFFRICATES),("nasal",_NASALS),("liquid",_LIQUIDS),("glide",_GLIDES)):
            if p in group: return name
        return "other"
    if prop == "place":
        if p in _VOWELS: return "vowel"
        if p in _LABIAL: return "labial"
        if p in _CORONAL: return "coronal"
        if p in _DORSAL: return "dorsal"
        return "other"
    if prop == "phonetic_voicing": return "voiced" if p in _VOICED else "unvoiced"
    return p


def cache_metadata(bundle: AnalysisBundle, cache: FeatureCache) -> pd.DataFrame:
    table = bundle.utterances.copy()
    table["utterance_id"] = table["utterance_id"].astype(str)
    table = table.set_index("utterance_id").loc[cache.utterance_ids.tolist()].reset_index()
    return table


def frame_to_utterance(cache: FeatureCache, frame_rows: np.ndarray) -> np.ndarray:
    ends = cache.offsets + cache.lengths
    return np.searchsorted(ends, frame_rows, side="right")


def _normal_p(z: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda x: math.erfc(abs(float(x)) / math.sqrt(2.0)))(z)


def _split_values(bundle: AnalysisBundle, score_splits: str | None) -> set[str] | None:
    if score_splits is None:
        return None
    tokens = [x.strip() for x in str(score_splits).split(",") if x.strip()]
    if not tokens or any(x.lower() == "all" for x in tokens):
        return None
    aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
    values: set[str] = set()
    for token in tokens:
        logical = aliases.get(token.lower(), token)
        values.add(str(bundle.spec.split_map.get(logical, token)))
    return values


def _score_masks(cache: FeatureCache, bundle: AnalysisBundle, score_splits: str | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = cache_metadata(bundle, cache)
    split_values = _split_values(bundle, score_splits)
    if split_values is None:
        utterance_mask = np.ones(len(cache.utterance_ids), dtype=bool)
        selected = sorted(metadata["split"].astype(str).unique().tolist())
        requested = "all"
    else:
        split = metadata["split"].astype(str).to_numpy()
        utterance_mask = np.isin(split, list(split_values))
        selected = sorted(split_values)
        requested = str(score_splits)
        if not utterance_mask.any():
            available = sorted(metadata["split"].astype(str).unique().tolist())
            raise AnalysisError(
                f"--score-splits={score_splits!r} selected no utterances; "
                f"available manifest split values are {available}."
            )
    frame_to_utt = frame_to_utterance(cache, np.arange(cache.n_frames, dtype=np.int64))
    frame_mask = utterance_mask[frame_to_utt]
    return utterance_mask, frame_mask, {
        "requested": requested,
        "selected_split_values": selected,
        "utterances": int(utterance_mask.sum()),
        "frames": int(frame_mask.sum()),
    }


def _training_like_deadness(cache: FeatureCache, resolved: ResolvedModel) -> dict[str, Any]:
    """Approximate the trainer's dead-latent counter on the analysis stream.

    Training deadness is not "never fired in a corpus"; it is a transient
    counter: a unit is dead if it has not fired for dead_steps_threshold
    optimizer batches.  The counter is intentionally not checkpointed, so the
    best post-hoc analogue is to replay the analysis utterances in batches and
    count how many consecutive analysis batches each unit has missed.
    """
    K = cache.K
    cfg = resolved.config
    batch_size = int(cfg.get("batch_size", cfg.get("eval_batch_size", 16)) or 16)
    batch_size = max(batch_size, 1)
    threshold = int(cfg.get("dead_steps_threshold", 256) or 256)
    inactive = np.zeros(K, dtype=np.int32)
    max_inactive = np.zeros(K, dtype=np.int32)
    fired_batches = np.zeros(K, dtype=np.int32)
    n_utterances = len(cache.utterance_ids)
    n_batches = int(math.ceil(n_utterances / batch_size)) if n_utterances else 0
    for start in range(0, n_utterances, batch_size):
        fired = np.zeros(K, dtype=bool)
        for ui in range(start, min(start + batch_size, n_utterances)):
            sl = cache.utterance_slice(ui)
            if sl.stop > sl.start:
                fired[np.unique(cache.indices[sl].reshape(-1).astype(np.int64))] = True
        inactive += 1
        inactive[fired] = 0
        fired_batches[fired] += 1
        max_inactive = np.maximum(max_inactive, inactive)
    raw_train_like_dead = inactive > threshold
    # A stream only barely longer than the threshold is dominated by its tail
    # ordering and is not a stable analogue of shuffled training. Require two
    # full threshold windows before exposing a headline deadness estimate.
    comparable = bool(n_batches >= 2 * threshold)
    train_like_dead = raw_train_like_dead if comparable else np.zeros(K, dtype=bool)
    return {
        "batch_size": batch_size,
        "threshold": threshold,
        "n_batches": n_batches,
        "final_inactive_batches": inactive,
        "max_inactive_batches": max_inactive,
        "fired_batches": fired_batches,
        "train_like_dead": train_like_dead,
        "raw_train_like_dead": raw_train_like_dead,
        "comparable": comparable,
    }


def health_analysis(cache: FeatureCache, resolved: ResolvedModel, output: Path) -> tuple[pd.DataFrame, dict]:
    K = cache.K
    flat_idx = cache.indices.reshape(-1)
    flat_val = cache.values.astype(np.float32).reshape(-1)
    count = np.bincount(flat_idx, minlength=K)
    pos = np.bincount(flat_idx, weights=(flat_val > 0), minlength=K)
    neg = np.bincount(flat_idx, weights=(flat_val < 0), minlength=K)
    total = np.bincount(flat_idx, weights=flat_val, minlength=K)
    abs_total = np.bincount(flat_idx, weights=np.abs(flat_val), minlength=K)
    sq = np.bincount(flat_idx, weights=flat_val ** 2, minlength=K)
    utterance_count = np.zeros(K, dtype=np.int64)
    for i in range(len(cache.utterance_ids)):
        sl = cache.utterance_slice(i)
        if sl.stop > sl.start:
            seen = np.unique(cache.indices[sl].reshape(-1).astype(np.int64))
            utterance_count[seen] += 1
    decoder = resolved.state["sae.dec_weight"].detach().float().cpu().numpy()
    decoder_norm = np.linalg.norm(decoder, axis=0)
    mean = np.divide(total, count, out=np.zeros(K), where=count > 0)
    var = np.divide(sq, count, out=np.zeros(K), where=count > 0) - mean ** 2
    deadness = _training_like_deadness(cache, resolved)
    observed_active = count > 0
    unobserved = ~observed_active
    train_like_dead = deadness["train_like_dead"]
    rows = []
    for j in range(K):
        rows.append({
            "unit": j, "route": ROUTE_NAMES.get(int(cache.route[j]), str(int(cache.route[j]))),
            "route_id": int(cache.route[j]), "route_probability": float(cache.route_probability[j]),
            "active_frames": int(count[j]), "frame_frequency": float(count[j] / max(cache.n_frames, 1)),
            "active_utterances": int(utterance_count[j]),
            "utterance_frequency": float(utterance_count[j] / max(len(cache.utterance_ids), 1)),
            "positive_fraction": float(pos[j] / count[j]) if count[j] else 0.0,
            "negative_fraction": float(neg[j] / count[j]) if count[j] else 0.0,
            "mean_when_active": float(mean[j]), "std_when_active": float(math.sqrt(max(var[j], 0))),
            "mean_abs_contribution": float(abs_total[j] * decoder_norm[j] / max(cache.n_frames, 1)),
            "decoder_norm": float(decoder_norm[j]),
            "observed_active": bool(observed_active[j]),
            "unobserved": bool(unobserved[j]),
            "train_like_dead": bool(train_like_dead[j]),
            # Keep the historical column name, but make it comparable to the
            # trainer: dead now means inactive for > dead_steps_threshold
            # analysis batches, not merely absent from the analysis subset.
            "dead": bool(train_like_dead[j]),
            "final_inactive_batches": int(deadness["final_inactive_batches"][j]),
            "max_inactive_batches": int(deadness["max_inactive_batches"][j]),
            "fired_batches": int(deadness["fired_batches"][j]),
            "rare": bool(0 < count[j] < max(5, cache.n_frames * 1e-5)),
            "ubiquitous": bool(count[j] > 0.5 * cache.n_frames),
        })
    frame = pd.DataFrame(rows)
    route_summary = []
    for rid in sorted(set(cache.route.tolist())):
        subset = frame[frame.route_id == rid]
        route_summary.append({
            "route": ROUTE_NAMES.get(int(rid), str(rid)), "assigned_units": int(len(subset)),
            "active_units": int(subset.observed_active.sum()),
            "unobserved_fraction": float(subset.unobserved.mean()),
            "train_like_dead_fraction": float(subset.train_like_dead.mean()),
            "dead_fraction": float(subset.train_like_dead.mean()),
            "median_frame_frequency": float(subset.frame_frequency.median()),
            "active_slots_per_frame": float(subset.active_frames.sum() / max(cache.n_frames, 1)),
        })
    summary = {
        "frames": cache.n_frames, "utterances": len(cache.utterance_ids), "K": K,
        "active_units": int(observed_active.sum()),
        "unobserved_units": int(unobserved.sum()),
        "dead_units": int(train_like_dead.sum()),
        "train_like_dead_units": int(train_like_dead.sum()),
        "deadness_batch_size": int(deadness["batch_size"]),
        "deadness_threshold_batches": int(deadness["threshold"]),
        "deadness_analysis_batches": int(deadness["n_batches"]),
        "deadness_comparable_to_training": bool(deadness["comparable"]),
        "raw_tail_inactive_units": int(deadness["raw_train_like_dead"].sum()),
        "route_summary": route_summary,
    }
    frame.to_csv(output / "tables" / "units.csv", index=False)
    try:
        frame.to_parquet(output / "tables" / "units.parquet", index=False)
    except Exception:
        pass
    write_json(output / "health.json", summary)
    return frame, summary


def top_examples(
    cache: FeatureCache, bundle: AnalysisBundle, n: int = 10, *, splits: str = "test",
) -> pd.DataFrame:
    heaps: list[list[tuple[float, int, float]]] = [[] for _ in range(cache.K)]
    reservoirs: list[list[tuple[float, int, float]]] = [[] for _ in range(cache.K)]
    seen = np.zeros(cache.K, dtype=np.int64)
    rng = np.random.default_rng(42)
    import heapq
    _, frame_mask, _ = _score_masks(cache, bundle, splits)
    for frame in np.flatnonzero(frame_mask):
        for unit, value in zip(cache.indices[frame], cache.values[frame]):
            unit = int(unit); seen[unit] += 1
            item = (abs(float(value)), frame, float(value))
            heap = heaps[unit]
            if len(heap) < n:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
            reservoir = reservoirs[unit]
            if len(reservoir) < 64:
                reservoir.append(item)
            else:
                replace = int(rng.integers(seen[unit]))
                if replace < len(reservoir): reservoir[replace] = item
    metadata = cache_metadata(bundle, cache)
    rows = []
    for unit, heap in enumerate(heaps):
        chosen = [("top", rank, item) for rank, item in enumerate(sorted(heap, reverse=True), 1)]
        if reservoirs[unit]:
            sample = sorted(reservoirs[unit])
            for q in (.25, .50, .75):
                chosen.append((f"quantile_{int(q*100)}", 0, sample[round(q*(len(sample)-1))]))
        for example_type, rank, (_, frame, value) in chosen:
            ui = int(frame_to_utterance(cache, np.asarray([frame]))[0])
            row = metadata.iloc[ui]
            local = frame - int(cache.offsets[ui])
            duration = float(cache.lengths[ui])
            rows.append({
                "unit": unit, "rank": rank, "example_type": example_type,
                "utterance_id": cache.utterance_ids[ui],
                "frame": int(frame), "local_frame": int(local), "activation": value,
                "phone": str(cache.phones[frame]), "time_fraction": float((local + .5) / max(duration, 1)),
                "audio_path": str(bundle.audio_path(row)), "transcript": str(row.get("transcript", "")),
                "speaker_id": row.get("speaker_id", ""), "emotion": row.get("emotion", ""),
            })
    return pd.DataFrame(rows)


def _categorical_scores(
    idx: np.ndarray, val: np.ndarray, labels: np.ndarray, K: int,
    factor: str, family: str, min_count: int = 10,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels).astype(str)
    valid = labels != "<missing>"
    idx, val, labels = idx[valid], val[valid], labels[valid]
    total_active = np.bincount(idx.reshape(-1), minlength=K).astype(float)
    total_sum = np.bincount(idx.reshape(-1), weights=val.reshape(-1), minlength=K)
    rows = []
    import heapq
    levels, level_counts = np.unique(labels, return_counts=True)
    high_cardinality = len(levels) > 64
    strongest: list[list[tuple[float, int, dict[str, Any]]]] = [[] for _ in range(K)]
    serial = 0
    N = len(labels)
    for level, n_level in zip(levels, level_counts):
        if n_level < min_count or n_level == N:
            continue
        mask = labels == level
        active_in = np.bincount(idx[mask].reshape(-1), minlength=K).astype(float)
        sum_in = np.bincount(idx[mask].reshape(-1), weights=val[mask].reshape(-1), minlength=K)
        prevalence = n_level / N
        precision = np.divide(active_in, total_active, out=np.full(K, prevalence), where=total_active > 0)
        recall = active_in / n_level
        active_out = total_active - active_in
        false_positive_rate = active_out / (N - n_level)
        # AUROC for the binary Top-K-selection indicator.  This is not the
        # full continuous-amplitude AUROC from the proposal; it treats a unit's
        # selected-vs-unselected state as the score, which is exact for a binary
        # sparse activation indicator and cheap enough to compute for all units.
        active_auroc = 0.5 + 0.5 * (recall - false_positive_rate)
        active_auroc = np.clip(active_auroc, 0.0, 1.0)
        # Direction matters. Positive values mean the unit is selected more
        # often for this level; negative values are anti-associations and must
        # not be ranked as phone/speaker units.
        active_auroc_signed = 2.0 * (active_auroc - 0.5)
        active_auroc_positive = np.clip(active_auroc_signed, 0.0, 1.0)
        active_auroc_magnitude = np.abs(active_auroc_signed)
        # Exact AP for a two-level score (active vs inactive).
        ap = precision * recall + prevalence * (1.0 - recall)
        mean_in = sum_in / n_level
        mean_out = (total_sum - sum_in) / (N - n_level)
        effect = mean_in - mean_out
        expected = total_active * prevalence
        variance = np.maximum(total_active * prevalence * (1 - prevalence), 1.0)
        z = (active_in - expected) / np.sqrt(variance)
        p = _normal_p(z)
        q = bh_fdr(p)
        lift = np.divide(precision, prevalence, out=np.ones(K), where=prevalence > 0)
        for unit in np.flatnonzero(total_active):
            row = {
                "unit": int(unit), "factor": factor, "family": family, "level": str(level),
                "metric": "binary_activation", "prevalence": float(prevalence),
                "auprc": float(ap[unit]), "precision": float(precision[unit]),
                "recall": float(recall[unit]), "lift": float(lift[unit]),
                "active_auroc": float(active_auroc[unit]),
                "active_auroc_signed": float(active_auroc_signed[unit]),
                "active_auroc_positive": float(active_auroc_positive[unit]),
                "active_auroc_magnitude": float(active_auroc_magnitude[unit]),
                "effect": float(effect[unit]), "z": float(z[unit]), "p": float(p[unit]),
                "q": float(q[unit]), "score": float(max(z[unit], 0.0)),
                "magnitude_score": float(abs(z[unit])),
            }
            if high_cardinality:
                serial += 1
                item = (row["score"], serial, row)
                heap = strongest[int(unit)]
                if len(heap) < 5: heapq.heappush(heap, item)
                elif item[0] > heap[0][0]: heapq.heapreplace(heap, item)
            else:
                rows.append(row)
    if high_cardinality:
        for heap in strongest:
            rows.extend(item[2] for item in sorted(heap, reverse=True))
    return rows


def _categorical_dense_scores(
    active: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    factor: str,
    family: str,
    min_count: int = 3,
) -> list[dict[str, Any]]:
    """Categorical scores for utterance-level unit activity.

    ``values`` contains each unit's mean activation amplitude over the
    utterance.  The directional point-biserial correlation with a label is the
    headline association. ``active`` (whether the unit fired at least once) is
    retained only for coverage diagnostics; it saturates on long utterances
    and is therefore unsuitable as the speaker-selectivity score.
    """
    active = np.asarray(active, dtype=bool)
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    valid = labels != "<missing>"
    active, values, labels = active[valid], values[valid], labels[valid]
    if not len(labels):
        return []
    K = active.shape[1]
    total_active = active.sum(axis=0).astype(float)
    total_sum = values.sum(axis=0, dtype=np.float64)
    total_sum2 = np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
    total_mean = total_sum / len(labels)
    total_std = np.sqrt(np.maximum(total_sum2 / len(labels) - total_mean ** 2, 0.0))
    levels, level_counts = np.unique(labels, return_counts=True)
    high_cardinality = len(levels) > 64
    strongest: list[list[tuple[float, int, dict[str, Any]]]] = [[] for _ in range(K)]
    rows: list[dict[str, Any]] = []
    serial = 0
    import heapq
    N = len(labels)
    for level, n_level in zip(levels, level_counts):
        if n_level < min_count or n_level == N:
            continue
        mask = labels == level
        active_in = active[mask].sum(axis=0).astype(float)
        sum_in = values[mask].sum(axis=0, dtype=np.float64)
        prevalence = float(n_level / N)
        precision = np.divide(active_in, total_active, out=np.full(K, prevalence), where=total_active > 0)
        recall = active_in / n_level
        false_positive_rate = (total_active - active_in) / (N - n_level)
        active_auroc = np.clip(0.5 + 0.5 * (recall - false_positive_rate), 0.0, 1.0)
        signed = 2.0 * (active_auroc - 0.5)
        positive = np.clip(signed, 0.0, 1.0)
        magnitude = np.abs(signed)
        ap = precision * recall + prevalence * (1.0 - recall)
        mean_in = sum_in / n_level
        mean_out = (total_sum - sum_in) / (N - n_level)
        effect = mean_in - mean_out
        amplitude_r = np.divide(
            effect * np.sqrt(prevalence * (1.0 - prevalence)),
            total_std,
            out=np.zeros(K, dtype=np.float64),
            where=total_std > 1e-12,
        )
        amplitude_r = np.clip(amplitude_r, -1.0, 1.0)
        amplitude_positive = np.clip(amplitude_r, 0.0, 1.0)
        amplitude_magnitude = np.abs(amplitude_r)
        z = amplitude_r * np.sqrt(
            np.maximum(N - 2, 1) / np.maximum(1.0 - amplitude_r ** 2, 1e-12)
        )
        p = _normal_p(z)
        q = bh_fdr(p)
        lift = np.divide(precision, prevalence, out=np.ones(K), where=prevalence > 0)
        for unit in np.flatnonzero(total_std > 1e-12):
            row = {
                "unit": int(unit), "factor": factor, "family": family, "level": str(level),
                "metric": "utterance_mean_activation", "prevalence": prevalence,
                "auprc": float(ap[unit]), "precision": float(precision[unit]),
                "recall": float(recall[unit]), "lift": float(lift[unit]),
                "active_auroc": float(active_auroc[unit]),
                "active_auroc_signed": float(signed[unit]),
                "active_auroc_positive": float(positive[unit]),
                "active_auroc_magnitude": float(magnitude[unit]),
                "amplitude_r_signed": float(amplitude_r[unit]),
                "amplitude_r_positive": float(amplitude_positive[unit]),
                "amplitude_r_magnitude": float(amplitude_magnitude[unit]),
                "effect": float(effect[unit]), "z": float(z[unit]),
                "p": float(p[unit]), "q": float(q[unit]),
                "score": float(amplitude_positive[unit]),
                "magnitude_score": float(amplitude_magnitude[unit]),
            }
            if high_cardinality:
                serial += 1
                item = (row["score"], serial, row)
                heap = strongest[int(unit)]
                if len(heap) < 5:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)
            else:
                rows.append(row)
    if high_cardinality:
        for heap in strongest:
            rows.extend(item[2] for item in sorted(heap, reverse=True))
    return rows


def _continuous_scores(
    idx: np.ndarray, val: np.ndarray, y: np.ndarray, K: int,
    factor: str, family: str,
) -> list[dict[str, Any]]:
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    idx, val, y = idx[valid], val[valid].astype(np.float32), y[valid]
    if len(y) < 10 or np.nanstd(y) < 1e-12:
        return []
    # Spearman association: rank y, while unit values retain their signed sparse amplitude.
    order = np.argsort(np.argsort(y)).astype(float)
    order = (order - order.mean()) / (order.std() + 1e-12)
    flat_i, flat_v = idx.reshape(-1), val.reshape(-1)
    sum_x = np.bincount(flat_i, weights=flat_v, minlength=K)
    sum_x2 = np.bincount(flat_i, weights=flat_v ** 2, minlength=K)
    sum_xy = np.bincount(flat_i, weights=(val * order[:, None]).reshape(-1), minlength=K)
    N = len(y)
    mean_x = sum_x / N
    var_x = np.maximum(sum_x2 / N - mean_x ** 2, 1e-12)
    rho = (sum_xy / N) / np.sqrt(var_x)
    rho = np.clip(rho, -1, 1)
    z = rho * np.sqrt(np.maximum(N - 2, 1) / np.maximum(1 - rho ** 2, 1e-12))
    p, q = _normal_p(z), bh_fdr(_normal_p(z))
    return [{
        "unit": int(j), "factor": factor, "family": family, "level": "",
        "metric": "spearman", "rho": float(rho[j]), "r2": float(rho[j] ** 2),
        "z": float(z[j]), "p": float(p[j]), "q": float(q[j]), "score": float(abs(z[j])),
    } for j in np.flatnonzero(sum_x2)]


def selectivity_analysis(
    cache: FeatureCache, bundle: AnalysisBundle, output: Path,
    *,
    factor_scope: str = "speaker_phone",
    score_splits: str | None = "train,validation",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    factor_scope = str(factor_scope or "speaker_phone").lower().replace("-", "_")
    if factor_scope not in {"speaker_phone", "broad"}:
        raise ValueError(f"Unknown factor scope: {factor_scope}")
    metadata = cache_metadata(bundle, cache)
    utterance_mask, frame_mask, split_summary = _score_masks(cache, bundle, score_splits)
    rows: list[dict[str, Any]] = []
    factors = {f.name: f for f in bundle.spec.factors}
    if "phone" in factors:
        # MFA leaves silence/gaps as "<unaligned>".  Treating that as an
        # ordinary phone makes silence/timing units look like phone-identity
        # units, which badly inflates "phone leakage" in route summaries.
        phones = np.asarray(cache.phones).astype("U32")
        aligned = (np.char.upper(phones) != "<UNALIGNED>") & frame_mask
        phone_idx = cache.indices[aligned]
        phone_val = cache.values[aligned].astype(np.float32)
        phone_labels = phones[aligned]
        rows += _categorical_scores(phone_idx, phone_val, phone_labels,
                                    cache.K, "phone", "linguistic", min_count=20)
        if factor_scope == "broad":
            for prop in ("manner", "place", "phonetic_voicing"):
                labels = np.asarray([_phone_property(p, prop) for p in phone_labels], dtype="U32")
                rows += _categorical_scores(phone_idx, phone_val, labels,
                                            cache.K, prop, "linguistic", min_count=20)
            starts = np.zeros(len(phones), dtype=bool)
            starts[cache.offsets] = True
            previous = np.roll(phones, 1)
            previous_aligned = np.roll(aligned, 1)
            previous_aligned[starts] = False
            aligned_pair = aligned & previous_aligned
            boundary = np.where(phones[aligned_pair] != previous[aligned_pair], "boundary", "inside")
            transition = np.char.add(np.char.add(previous[aligned_pair].astype("U32"), ">"),
                                     phones[aligned_pair].astype("U32"))
            transition[boundary == "inside"] = "<none>"
            rows += _categorical_scores(cache.indices[aligned_pair],
                                        cache.values[aligned_pair].astype(np.float32), boundary,
                                        cache.K, "phone_boundary", "linguistic", min_count=20)
            rows += _categorical_scores(cache.indices[aligned_pair],
                                        cache.values[aligned_pair].astype(np.float32), transition,
                                        cache.K, "phone_transition", "linguistic", min_count=20)
    if factor_scope == "broad":
        for name in ("f0", "energy", "voicing"):
            if name in factors:
                rows += _continuous_scores(cache.indices[frame_mask], cache.values[frame_mask].astype(np.float32), getattr(cache, name)[frame_mask],
                                           cache.K, name, "paralinguistic")
    # Utterance factors use mean activation amplitudes plus an explicit
    # ever-active indicator. Do not impose a second arbitrary pooled Top-K.
    pooled = cache.pooled_z[utterance_mask].astype(np.float32)
    metadata_scored = metadata[utterance_mask].reset_index(drop=True)
    scored_utt_ids = np.flatnonzero(utterance_mask)
    pooled_active = np.zeros((len(scored_utt_ids), cache.K), dtype=bool)
    for local, ui in enumerate(scored_utt_ids):
        sl = cache.utterance_slice(int(ui))
        if sl.stop > sl.start:
            pooled_active[local, np.unique(cache.indices[sl].reshape(-1).astype(int))] = True
    for factor in bundle.spec.factors:
        if factor.level != "utterance" or factor.source.startswith("computed:"):
            continue
        if factor_scope == "speaker_phone" and factor.name != "speaker_id":
            continue
        if factor.source not in metadata_scored:
            continue
        y = metadata_scored[factor.source].fillna("<missing>").to_numpy()
        if factor.kind == "categorical":
            # Libri speakers are disjoint across train/dev/test. Score speaker
            # identity within each split so split-level recording differences
            # cannot masquerade as speaker-selective units.
            if factor.name == "speaker_id" and "split" in metadata_scored:
                split_labels = metadata_scored["split"].astype(str).to_numpy()
                for split_value in sorted(np.unique(split_labels).tolist()):
                    split_mask = split_labels == split_value
                    split_rows = _categorical_dense_scores(
                        pooled_active[split_mask], pooled[split_mask], y[split_mask],
                        factor.name, factor.family, min_count=3,
                    )
                    for item in split_rows:
                        item["subset"] = str(split_value)
                    rows += split_rows
            else:
                rows += _categorical_dense_scores(
                    pooled_active, pooled, y, factor.name, factor.family, min_count=3,
                )
        else:
            dense_idx = np.broadcast_to(np.arange(cache.K, dtype=np.int32), pooled.shape)
            rows += _continuous_scores(
                dense_idx, pooled,
                pd.to_numeric(metadata_scored[factor.source], errors="coerce"),
                cache.K, factor.name, factor.family,
            )
    scores = pd.DataFrame(rows)
    if scores.empty:
        profiles = pd.DataFrame({"unit": np.arange(cache.K), "linguistic_score": 0., "paralinguistic_score": 0.})
    else:
        scores["profile_score"] = pd.to_numeric(scores["score"], errors="coerce").fillna(0.0)
        frame_categorical = scores["metric"].eq("binary_activation")
        if "active_auroc_positive" in scores:
            scores.loc[frame_categorical, "profile_score"] = pd.to_numeric(
                scores.loc[frame_categorical, "active_auroc_positive"], errors="coerce",
            ).fillna(0.0)
        utterance_categorical = scores["metric"].eq("utterance_mean_activation")
        if "amplitude_r_positive" in scores:
            scores.loc[utterance_categorical, "profile_score"] = pd.to_numeric(
                scores.loc[utterance_categorical, "amplitude_r_positive"], errors="coerce",
            ).fillna(0.0)
        by_factor = scores.groupby(["unit", "family", "factor"], as_index=False)["profile_score"].max()
        by_factor = by_factor.rename(columns={"profile_score": "score"})
        signatures = by_factor.pivot(index="unit", columns="factor", values="score").fillna(0)
        signatures.columns = [f"{c}__score" for c in signatures.columns]
        best = by_factor.groupby(["unit", "family"], as_index=False)["score"].max()
        profiles = best.pivot(index="unit", columns="family", values="score").fillna(0)
        profiles = profiles.join(signatures, how="outer").fillna(0).reset_index()
        for col in ("linguistic", "paralinguistic"):
            if col not in profiles:
                profiles[col] = 0.0
        profiles = profiles.rename(columns={"linguistic": "linguistic_score", "paralinguistic": "paralinguistic_score"})
        profiles["delta_L_minus_P"] = profiles.linguistic_score - profiles.paralinguistic_score
    profiles = pd.DataFrame({"unit": np.arange(cache.K)}).merge(profiles, on="unit", how="left").fillna(0)
    for col in ("linguistic_score", "paralinguistic_score"):
        if col not in profiles:
            profiles[col] = 0.0
    if "delta_L_minus_P" not in profiles:
        profiles["delta_L_minus_P"] = profiles["linguistic_score"] - profiles["paralinguistic_score"]
    profiles["route_id"] = cache.route
    profiles["route"] = [ROUTE_NAMES.get(int(x), str(x)) for x in cache.route]
    L = profiles[profiles.route_id == 0].delta_L_minus_P.to_numpy()
    P = profiles[profiles.route_id == 1].delta_L_minus_P.to_numpy()
    effect = float(L.mean() - P.mean()) if len(L) and len(P) else float("nan")
    if len(L) and len(P):
        # Mann-Whitney interpretation: probability that a random L unit has a
        # larger linguistic-minus-paralinguistic score than a random P unit.
        alignment_auc = float((L[:, None] > P[None, :]).mean() + .5 * (L[:, None] == P[None, :]).mean())
    else:
        alignment_auc = float("nan")
    ci_L, ci_P = bootstrap_ci(L), bootstrap_ci(P)
    summary = {
        "rows": int(len(scores)), "route_alignment_effect": effect, "route_alignment_auc": alignment_auc,
        "factor_scope": factor_scope,
        "score_splits": split_summary,
        "L_delta_mean": float(L.mean()) if len(L) else None,
        "P_delta_mean": float(P.mean()) if len(P) else None,
        "L_delta_ci": ci_L, "P_delta_ci": ci_P,
    }
    scores.to_csv(output / "tables" / "unit_factor_scores.csv", index=False)
    profiles.to_csv(output / "tables" / "unit_profiles.csv", index=False)
    try:
        scores.to_parquet(output / "tables" / "unit_factor_scores.parquet", index=False)
    except Exception:
        pass
    write_json(output / "selectivity.json", summary)
    return scores, profiles, summary


def _metric_summary(
    scores: pd.DataFrame,
    factor: str,
    K: int,
    prefix: str,
    *,
    association: str = "active_auroc",
) -> pd.DataFrame:
    base = pd.DataFrame({"unit": np.arange(K, dtype=int)})
    positive_source = f"{association}_positive"
    signed_source = f"{association}_signed"
    positive_max = f"{prefix}_{positive_source}_max"
    positive_mean = f"{prefix}_{positive_source}_mean"
    signed_output = f"{prefix}_{signed_source}"
    raw_source = "active_auroc" if association == "active_auroc" else None
    raw_output = f"{prefix}_{raw_source}" if raw_source else None

    def empty() -> pd.DataFrame:
        values: dict[str, Any] = {
            f"preferred_{prefix}": "",
            signed_output: 0.0,
            positive_max: 0.0,
            positive_mean: 0.0,
            f"{prefix}_levels_evaluated": 0,
        }
        if raw_output:
            values[raw_output] = np.nan
        return base.assign(**values)

    if scores is None or scores.empty:
        return empty()
    association_values = scores.get(positive_source, pd.Series(np.nan, index=scores.index))
    sub = scores[(scores["factor"] == factor) & association_values.notna()].copy()
    if sub.empty:
        return empty()
    sub["unit"] = sub["unit"].astype(int)
    positive = sub[sub[positive_source] > 0].copy()
    order = positive.sort_values(
        ["unit", positive_source, "score"], ascending=[True, False, False])
    best_columns = ["unit", "level", signed_source, positive_source, "prevalence", "q"]
    if raw_source:
        best_columns.insert(2, raw_source)
    best = (
        order.groupby("unit", as_index=False).first()[best_columns]
        if len(order) else pd.DataFrame(columns=best_columns)
    )
    renames = {
        "level": f"preferred_{prefix}",
        signed_source: signed_output,
        positive_source: positive_max,
        "prevalence": f"preferred_{prefix}_prevalence",
        "q": f"{prefix}_best_q",
    }
    if raw_source and raw_output:
        renames[raw_source] = raw_output
    best = best.rename(columns=renames)
    mean = sub.groupby("unit", as_index=False).agg(
        **{
            positive_mean: (positive_source, "mean"),
        }
    ) if len(sub) else pd.DataFrame(columns=["unit", positive_mean])
    levels = sub.groupby("unit", as_index=False).agg(
        **{f"{prefix}_levels_evaluated": ("level", "nunique")}
    )
    out = base.merge(best, on="unit", how="left").merge(mean, on="unit", how="left").merge(
        levels, on="unit", how="left")
    defaults = {
        f"preferred_{prefix}": "",
        signed_output: 0.0,
        positive_max: 0.0,
        f"preferred_{prefix}_prevalence": 0.0,
        f"{prefix}_best_q": np.nan,
        positive_mean: 0.0,
        f"{prefix}_levels_evaluated": 0,
    }
    if raw_output:
        defaults[raw_output] = np.nan
    for col, value in defaults.items():
        if col not in out:
            out[col] = value
        out[col] = out[col].fillna(value)
    return out


def phone_speaker_unit_scores(
    cache: FeatureCache,
    health: pd.DataFrame | None,
    profiles: pd.DataFrame,
    scores: pd.DataFrame,
    output: Path,
    *,
    phone_weight_max: float = 0.5,
    phone_weight_mean: float = 0.5,
    speaker_weight_max: float = 1.0,
    speaker_weight_mean: float = 0.0,
    threshold_percentile: float = 0.90,
) -> tuple[pd.DataFrame, dict]:
    """Create the thesis-facing per-unit phone-vs-speaker score table.

    Phone evidence is the binary frame-level Top-K indicator. Speaker evidence
    is the unit's mean activation amplitude per utterance. The default
    composite is deliberately simple and reported:

      PhoneScore = .5 * max positive phone AUROC + .5 * mean positive phone AUROC
      SpeakerScore = max positive speaker/mean-activation correlation
      D = (PhoneScore - SpeakerScore) / (PhoneScore + SpeakerScore + eps)
      M = PhoneScore + SpeakerScore

    Thresholds are empirical percentiles over units with positive evidence for
    the corresponding score, so zero-evidence units remain ``other``.
    """
    K = cache.K
    units = pd.DataFrame({
        "unit": np.arange(K, dtype=int),
        "route_id": cache.route.astype(int),
        "route": [ROUTE_NAMES.get(int(x), str(int(x))) for x in cache.route],
        "route_probability": cache.route_probability.astype(float),
    })
    if health is not None and len(health):
        keep = [c for c in (
            "unit", "frame_frequency", "utterance_frequency", "observed_active",
            "unobserved", "train_like_dead", "dead", "mean_abs_contribution",
        ) if c in health.columns]
        units = units.merge(health[keep], on="unit", how="left")
    if profiles is not None and len(profiles):
        keep = [c for c in ("unit", "linguistic_score", "paralinguistic_score", "delta_L_minus_P") if c in profiles.columns]
        units = units.merge(profiles[keep], on="unit", how="left")

    phone = _metric_summary(scores, "phone", K, "phone")
    speaker = _metric_summary(
        scores, "speaker_id", K, "speaker", association="amplitude_r",
    )
    units = units.merge(phone, on="unit", how="left").merge(speaker, on="unit", how="left")

    phone_weight_sum = max(float(phone_weight_max) + float(phone_weight_mean), 1e-12)
    speaker_weight_sum = max(float(speaker_weight_max) + float(speaker_weight_mean), 1e-12)
    phone_w_max = float(phone_weight_max) / phone_weight_sum
    phone_w_mean = float(phone_weight_mean) / phone_weight_sum
    speaker_w_max = float(speaker_weight_max) / speaker_weight_sum
    speaker_w_mean = float(speaker_weight_mean) / speaker_weight_sum
    units["PhoneScore"] = (
        phone_w_max * units["phone_active_auroc_positive_max"].astype(float)
        + phone_w_mean * units["phone_active_auroc_positive_mean"].astype(float)
    )
    units["SpeakerScore"] = (
        speaker_w_max * units["speaker_amplitude_r_positive_max"].astype(float)
        + speaker_w_mean * units["speaker_amplitude_r_positive_mean"].astype(float)
    )
    units["D"] = (units["PhoneScore"] - units["SpeakerScore"]) / (
        units["PhoneScore"] + units["SpeakerScore"] + 1e-12
    )
    units["M"] = units["PhoneScore"] + units["SpeakerScore"]

    threshold_percentile = float(threshold_percentile)
    if not 0 < threshold_percentile < 1:
        raise ValueError("threshold_percentile must be between 0 and 1.")
    # Estimate thresholds over units with actual evidence for that factor.  In
    # quick/local profiles many units can be completely unobserved, so including
    # the zero mass in the empirical quantile can make the threshold exactly 0
    # and falsely label dead/unobserved units as phone or speaker units.
    phone_positive = units.loc[units["PhoneScore"] > 0, "PhoneScore"]
    speaker_positive = units.loc[units["SpeakerScore"] > 0, "SpeakerScore"]
    phone_threshold = float(phone_positive.quantile(threshold_percentile)) if len(phone_positive) else float("inf")
    speaker_threshold = (
        float(speaker_positive.quantile(threshold_percentile)) if len(speaker_positive) else float("inf")
    )
    high_phone = (units["PhoneScore"] > 0) & (units["PhoneScore"] >= phone_threshold)
    high_speaker = (units["SpeakerScore"] > 0) & (units["SpeakerScore"] >= speaker_threshold)
    units["category"] = "other"
    units.loc[high_phone & ~high_speaker, "category"] = "phone"
    units.loc[~high_phone & high_speaker, "category"] = "speaker"
    units.loc[high_phone & high_speaker, "category"] = "entangled"

    tables = output / "tables"
    units.to_csv(tables / "unit_phone_speaker_scores.csv", index=False)
    units.sort_values(["PhoneScore", "M"], ascending=False).to_csv(tables / "phone_units_ranked.csv", index=False)
    units.sort_values(["SpeakerScore", "M"], ascending=False).to_csv(tables / "speaker_units_ranked.csv", index=False)
    units[units.category == "entangled"].sort_values("M", ascending=False).to_csv(
        tables / "entangled_units_ranked.csv", index=False)
    try:
        units.to_parquet(tables / "unit_phone_speaker_scores.parquet", index=False)
    except Exception:
        pass
    summary = {
        "phone_metric": "frame_binary_activation_positive_auroc",
        "speaker_metric": "utterance_mean_activation_positive_point_biserial_r",
        "phone_score_formula": f"{phone_w_max:.6g}*phone_active_auroc_positive_max + {phone_w_mean:.6g}*phone_active_auroc_positive_mean",
        "speaker_score_formula": f"{speaker_w_max:.6g}*speaker_amplitude_r_positive_max + {speaker_w_mean:.6g}*speaker_amplitude_r_positive_mean",
        "threshold_percentile": threshold_percentile,
        "phone_threshold": phone_threshold,
        "speaker_threshold": speaker_threshold,
        "phone_positive_units": int(len(phone_positive)),
        "speaker_positive_units": int(len(speaker_positive)),
        "categories": units["category"].value_counts().to_dict(),
    }
    write_json(output / "phone_speaker_scores.json", summary)
    return units, summary


def phone_unit_confusion(
    cache: FeatureCache,
    bundle: AnalysisBundle,
    output: Path,
    *,
    min_phone_frames: int = 20,
    selection_splits: str = "train,validation",
    evaluation_splits: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select unique positively specific units and evaluate them held out.

    Rows are selected phone-associated units; columns are actual phone labels.
    Cell values are P(unit is in the Top-K set | frame has this phone).  This is
    a descriptive coverage view, not a classifier. Units are assigned on the
    selection splits by diagonal margin and visualized on the evaluation split.
    """
    tables = output / "tables"
    phones = np.asarray(cache.phones).astype("U32")
    aligned = np.char.upper(phones) != "<UNALIGNED>"
    _, selection_frame_mask, selection_summary = _score_masks(cache, bundle, selection_splits)
    _, evaluation_frame_mask, evaluation_summary = _score_masks(cache, bundle, evaluation_splits)
    selection_mask = aligned & selection_frame_mask
    evaluation_mask = aligned & evaluation_frame_mask
    selection_values, selection_counts = np.unique(phones[selection_mask], return_counts=True)
    evaluation_values, evaluation_counts = np.unique(phones[evaluation_mask], return_counts=True)
    selection_count_map = {str(k): int(v) for k, v in zip(selection_values, selection_counts)}
    evaluation_count_map = {str(k): int(v) for k, v in zip(evaluation_values, evaluation_counts)}
    phone_order = [
        str(p) for p in sorted(set(selection_values.tolist()) & set(evaluation_values.tolist()))
        if selection_count_map[str(p)] >= min_phone_frames
        and evaluation_count_map[str(p)] >= max(1, min_phone_frames // 2)
    ]

    if not phone_order:
        empty = pd.DataFrame()
        empty.to_csv(tables / "phone_selected_unit_confusion.csv", index=False)
        empty.to_csv(tables / "phone_selected_units.csv", index=False)
        summary = {
            "phones": 0, "selected_units": 0,
            "metric": "heldout_p_unit_active_given_phone",
            "selection_splits": selection_summary,
            "evaluation_splits": evaluation_summary,
        }
        write_json(output / "phone_unit_confusion.json", summary)
        return empty, empty, summary

    def probabilities(mask: np.ndarray) -> np.ndarray:
        out = np.zeros((len(phone_order), cache.K), dtype=np.float32)
        for pi, phone in enumerate(phone_order):
            phone_mask = mask & (phones == phone)
            n = int(phone_mask.sum())
            if n:
                counts = np.bincount(
                    cache.indices[phone_mask].reshape(-1), minlength=cache.K,
                ).astype(np.float32)
                out[pi] = counts / n
        return out

    selection_prob = probabilities(selection_mask)
    evaluation_prob = probabilities(evaluation_mask)
    if len(phone_order) > 1:
        order = np.argsort(selection_prob, axis=0)
        max_idx = order[-1]
        max_value = np.take_along_axis(selection_prob, order[-1:].reshape(1, -1), axis=0)[0]
        second_value = np.take_along_axis(selection_prob, order[-2:-1].reshape(1, -1), axis=0)[0]
        other_max = np.broadcast_to(max_value, selection_prob.shape).copy()
        for pi in range(len(phone_order)):
            other_max[pi, max_idx == pi] = second_value[max_idx == pi]
    else:
        other_max = np.zeros_like(selection_prob)
    margin = selection_prob - other_max
    observed = selection_prob.max(axis=0) > 0
    candidate_units = np.flatnonzero(observed & (cache.route >= 0))
    try:
        from scipy.optimize import linear_sum_assignment
        candidate_margin = margin[:, candidate_units]
        # Zero-valued dummy columns allow a phone to remain unsupported instead
        # of forcing a negatively selective unit into the matrix.
        augmented = np.concatenate(
            [candidate_margin, np.zeros((len(phone_order), len(phone_order)), dtype=np.float32)],
            axis=1,
        )
        phone_idx, local_unit_idx = linear_sum_assignment(-augmented)
        assigned_pairs = [
            (int(pi), int(candidate_units[ui]))
            for pi, ui in zip(phone_idx.tolist(), local_unit_idx.tolist())
            if ui < len(candidate_units)
        ]
    except Exception:
        assigned_pairs = []
        used: set[int] = set()
        for pi in np.argsort(margin.max(axis=1))[::-1]:
            for unit in candidate_units[np.argsort(margin[pi, candidate_units])[::-1]]:
                if int(unit) not in used:
                    assigned_pairs.append((int(pi), int(unit)))
                    used.add(int(unit))
                    break

    selected_rows = []
    for pi, unit in sorted(assigned_pairs):
        selection_margin = float(margin[pi, unit])
        if selection_margin <= 0:
            continue
        eval_values = evaluation_prob[pi, unit]
        eval_col = evaluation_prob[:, unit]
        eval_other = np.delete(eval_col, pi)
        eval_max_other = float(eval_other.max()) if len(eval_other) else 0.0
        selected_rows.append({
            "phone": phone_order[pi],
            "phone_family": _phone_property(phone_order[pi], "manner"),
            "unit": int(unit),
            "route_id": int(cache.route[unit]),
            "route": ROUTE_NAMES.get(int(cache.route[unit]), str(int(cache.route[unit]))),
            "selection_target_probability": float(selection_prob[pi, unit]),
            "selection_max_other_probability": float(other_max[pi, unit]),
            "selection_margin": selection_margin,
            "evaluation_target_probability": float(eval_values),
            "evaluation_max_other_probability": eval_max_other,
            "evaluation_margin": float(eval_values - eval_max_other),
            "evaluation_max_probability": float(eval_col.max()),
            "evaluation_brightest_phone": phone_order[int(eval_col.argmax())],
            "evaluation_diagonal_is_max": bool(eval_values >= eval_col.max() - 1e-12),
            "selection_phone_frames": int(selection_count_map[phone_order[pi]]),
            "evaluation_phone_frames": int(evaluation_count_map[phone_order[pi]]),
        })
    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        selected.to_csv(tables / "phone_selected_units.csv", index=False)
        selected.to_csv(tables / "phone_selected_unit_confusion.csv", index=False)
        summary = {
            "phones": 0, "selected_units": 0,
            "metric": "heldout_p_unit_active_given_phone",
            "selection_splits": selection_summary,
            "evaluation_splits": evaluation_summary,
        }
        write_json(output / "phone_unit_confusion.json", summary)
        return selected, selected, summary

    selected_units_np = selected["unit"].to_numpy(dtype=int)
    total_frames = int(evaluation_mask.sum())
    total_active = np.bincount(cache.indices[evaluation_mask].reshape(-1), minlength=cache.K).astype(float)
    baseline = total_active[selected_units_np] / max(total_frames, 1)
    matrix = evaluation_prob[:, selected_units_np].T
    mat = pd.DataFrame(matrix, columns=phone_order)
    mat.insert(0, "selected_unit", selected_units_np)
    mat.insert(0, "selected_phone", selected["phone"].tolist())
    mat.insert(2, "route", selected["route"].tolist())
    mat.insert(3, "unit_baseline_active_probability", baseline.astype(float))

    long_rows = []
    for _, row in mat.iterrows():
        for phone in phone_order:
            base = float(row["unit_baseline_active_probability"])
            p_active = float(row[phone])
            long_rows.append({
                "selected_phone": row["selected_phone"],
                "selected_unit": int(row["selected_unit"]),
                "route": row["route"],
                "actual_phone": phone,
                "p_unit_active_given_phone": p_active,
                "lift_over_unit_baseline": p_active / max(base, 1e-12),
            })
    long = pd.DataFrame(long_rows)

    selected.to_csv(tables / "phone_selected_units.csv", index=False)
    mat.to_csv(tables / "phone_selected_unit_confusion.csv", index=False)
    long.to_csv(tables / "phone_selected_unit_confusion_long.csv", index=False)
    summary = {
        "phones": int(len(selected)),
        "selected_units": int(selected["unit"].nunique()),
        "candidate_phones": int(len(phone_order)),
        "metric": "heldout_p_unit_active_given_phone",
        "min_phone_frames": int(min_phone_frames),
        "unique_unit_rows": bool(selected["unit"].nunique() == len(selected)),
        "positive_selection_margin_rows": int((selected["selection_margin"] > 0).sum()),
        "positive_evaluation_margin_rows": int((selected["evaluation_margin"] > 0).sum()),
        "median_evaluation_margin": float(selected["evaluation_margin"].median()),
        "evaluation_diagonal_max_rows": int(selected["evaluation_diagonal_is_max"].sum()),
        "evaluation_diagonal_max_fraction": float(selected["evaluation_diagonal_is_max"].mean()),
        "selection_splits": selection_summary,
        "evaluation_splits": evaluation_summary,
    }
    write_json(output / "phone_unit_confusion.json", summary)
    return mat, selected, summary


def _route_sparse_matrix(cache: FeatureCache, row_ids: np.ndarray, route_id: int):
    from scipy import sparse

    row_ids = np.asarray(row_ids, dtype=int)
    route_units = np.flatnonzero(cache.route == route_id)
    local = np.full(cache.K, -1, dtype=np.int32)
    local[route_units] = np.arange(len(route_units), dtype=np.int32)
    idx = cache.indices[row_ids]
    val = cache.values[row_ids].astype(np.float32)
    cols = local[idx.reshape(-1)]
    keep = cols >= 0
    rows = np.repeat(np.arange(len(row_ids), dtype=np.int32), idx.shape[1])[keep]
    return sparse.csr_matrix((val.reshape(-1)[keep], (rows, cols[keep])),
                             shape=(len(row_ids), len(route_units)), dtype=np.float32)


def _full_sparse_matrix(cache: FeatureCache, row_ids: np.ndarray):
    from scipy import sparse

    row_ids = np.asarray(row_ids, dtype=int)
    idx = cache.indices[row_ids].astype(np.int32)
    val = cache.values[row_ids].astype(np.float32)
    rows = np.repeat(np.arange(len(row_ids), dtype=np.int32), idx.shape[1])
    return sparse.csr_matrix(
        (val.reshape(-1), (rows, idx.reshape(-1))),
        shape=(len(row_ids), cache.K), dtype=np.float32,
    )


def _embed_matrix(x, seed: int = 42) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if x.shape[0] < 3 or x.shape[1] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    arr = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    arr = np.asarray(arr, dtype=np.float32)
    try:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=seed).fit_transform(arr)
    except Exception:
        arr = arr - arr.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(arr, full_matrices=False)
        emb = arr @ vh[:2].T
    emb = np.asarray(emb, dtype=np.float32)
    if emb.shape[1] == 1:
        emb = np.c_[emb[:, 0], np.zeros(len(emb), dtype=np.float32)]
    return emb[:, :2]


def _load_umap():
    """Load UMAP once, before long SciPy-heavy analysis on Python 3.14."""
    global _UMAP_MODULE
    if _UMAP_MODULE is None:
        try:
            import umap
        except ImportError as exc:
            raise AnalysisError(
                "UMAP figures require umap-learn; install "
                "SAEUnitAnalysis/requirements.txt."
            ) from exc
        _UMAP_MODULE = umap
    return _UMAP_MODULE


def _embed_umap(
    x,
    seed: int = 42,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> np.ndarray:
    """Deterministic supplementary UMAP on the same observations as PCA."""
    if x.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if x.shape[0] < 3 or x.shape[1] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    umap = _load_umap()
    neighbors = min(int(n_neighbors), max(2, int(x.shape[0]) - 1))
    model = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(seed),
        transform_seed=int(seed),
        n_jobs=1,
        low_memory=True,
    )
    embedding = model.fit_transform(x)
    return np.asarray(embedding, dtype=np.float32)[:, :2]


def _highdim_label_margins(x, labels: np.ndarray) -> pd.DataFrame:
    """Route-neutral label clarity in the original representation space."""
    labels = np.asarray(labels).astype(str)
    unique = sorted(np.unique(labels).tolist())
    if len(unique) < 2:
        return pd.DataFrame(columns=["label", "n", "full_space_margin", "full_space_centroid_accuracy"])
    xn = _row_l2_normalize(x)
    centroids = []
    for label in unique:
        centroid = np.asarray(xn[labels == label].mean(axis=0)).reshape(-1)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid)
    centroids = np.stack(centroids, axis=0)
    scores = np.asarray(xn @ centroids.T)
    rows = []
    for li, label in enumerate(unique):
        mask = labels == label
        own = scores[mask, li]
        other = np.max(np.delete(scores[mask], li, axis=1), axis=1)
        pred = scores[mask].argmax(axis=1)
        rows.append({
            "label": label,
            "n": int(mask.sum()),
            "full_space_margin": float(np.mean(own - other)),
            "full_space_centroid_accuracy": float(np.mean(pred == li)),
        })
    return pd.DataFrame(rows).sort_values(
        ["full_space_margin", "full_space_centroid_accuracy", "n"], ascending=False)


def _row_l2_normalize(x):
    if hasattr(x, "multiply"):
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).reshape(-1))
        inv = np.divide(1.0, norms, out=np.zeros_like(norms, dtype=np.float32), where=norms > 1e-12)
        return x.multiply(inv[:, None]).tocsr()
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.divide(arr, norms, out=np.zeros_like(arr), where=norms > 1e-12)


def _stratified_partitions(labels: np.ndarray, *, seed: int = 42) -> np.ndarray:
    labels = np.asarray(labels).astype(str)
    partitions = np.full(len(labels), "unused", dtype="U10")
    rng = np.random.default_rng(seed)
    for label in sorted(np.unique(labels).tolist()):
        ids = np.flatnonzero(labels == label)
        ids = ids[rng.permutation(len(ids))]
        if len(ids) < 2:
            continue
        n_train = min(len(ids) - 1, max(1, int(round(0.7 * len(ids)))))
        partitions[ids[:n_train]] = "train"
        partitions[ids[n_train:]] = "evaluation"
    return partitions


def _holdout_metrics_and_confusion(
    x,
    labels: np.ndarray,
    *,
    seed: int = 42,
    partitions: np.ndarray | None = None,
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    """Stratified route metrics plus held-out frozen-probe confusion rows."""
    confusion_columns = [
        "true_label", "predicted_label", "count", "true_label_count",
        "row_fraction",
    ]
    labels = np.asarray(labels).astype(str)
    unique = sorted(np.unique(labels).tolist())
    if len(unique) < 2:
        return {
            "labels": len(unique), "points": len(labels),
            "balanced_accuracy": None, "linear_probe_balanced_accuracy": None,
            "mean_cosine_margin": None,
        }, pd.DataFrame(columns=confusion_columns)
    if partitions is None:
        partitions = _stratified_partitions(labels, seed=seed)
    partitions = np.asarray(partitions).astype(str)
    train_ids = np.flatnonzero(partitions == "train").tolist()
    eval_ids = np.flatnonzero(partitions == "evaluation").tolist()
    if not train_ids or not eval_ids:
        return {
            "labels": len(unique), "points": len(labels),
            "balanced_accuracy": None, "linear_probe_balanced_accuracy": None,
            "mean_cosine_margin": None,
        }, pd.DataFrame(columns=confusion_columns)
    xn = _row_l2_normalize(x)
    centroids = []
    for label in unique:
        ids = [i for i in train_ids if labels[i] == label]
        if not ids:
            centroids.append(np.zeros(x.shape[1], dtype=np.float32))
            continue
        centroid = np.asarray(xn[ids].mean(axis=0)).reshape(-1)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid)
    centroids = np.stack(centroids, axis=0)
    scores = np.asarray(xn[eval_ids] @ centroids.T)
    truth = np.asarray([unique.index(labels[i]) for i in eval_ids], dtype=int)
    pred = scores.argmax(axis=1)
    recalls = [float(np.mean(pred[truth == li] == li)) for li in range(len(unique)) if np.any(truth == li)]
    own = scores[np.arange(len(truth)), truth]
    masked = scores.copy(); masked[np.arange(len(truth)), truth] = -np.inf
    margin = own - masked.max(axis=1)
    linear_accuracy = None
    confusion_rows: list[dict[str, Any]] = []
    try:
        train_x = xn[train_ids]
        eval_x = xn[eval_ids]
        train_kernel = train_x @ train_x.T
        eval_kernel = eval_x @ train_x.T
        if hasattr(train_kernel, "toarray"):
            train_kernel = train_kernel.toarray()
        if hasattr(eval_kernel, "toarray"):
            eval_kernel = eval_kernel.toarray()
        train_kernel = np.asarray(train_kernel, dtype=np.float64)
        eval_kernel = np.asarray(eval_kernel, dtype=np.float64)
        targets = np.zeros((len(train_ids), len(unique)), dtype=np.float64)
        for row_id, original_id in enumerate(train_ids):
            targets[row_id, unique.index(labels[original_id])] = 1.0
        dual = np.linalg.solve(
            train_kernel + np.eye(len(train_ids), dtype=np.float64), targets,
        )
        linear_pred = np.argmax(eval_kernel @ dual, axis=1)
        linear_recalls = [
            float(np.mean(linear_pred[truth == li] == li))
            for li in range(len(unique)) if np.any(truth == li)
        ]
        linear_accuracy = float(np.mean(linear_recalls)) if linear_recalls else None
        confusion_counts = np.zeros((len(unique), len(unique)), dtype=int)
        np.add.at(confusion_counts, (truth, linear_pred), 1)
        for true_id, true_label in enumerate(unique):
            true_count = int(confusion_counts[true_id].sum())
            for predicted_id, predicted_label in enumerate(unique):
                count = int(confusion_counts[true_id, predicted_id])
                confusion_rows.append({
                    "true_label": str(true_label),
                    "predicted_label": str(predicted_label),
                    "count": count,
                    "true_label_count": true_count,
                    "row_fraction": float(count / max(true_count, 1)),
                })
    except (ValueError, np.linalg.LinAlgError):
        linear_accuracy = None
    return {
        "labels": int(len(unique)),
        "points": int(len(labels)),
        "train_points": int(len(train_ids)),
        "evaluation_points": int(len(eval_ids)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "linear_probe_balanced_accuracy": linear_accuracy,
        "mean_cosine_margin": float(np.mean(margin)),
    }, pd.DataFrame(confusion_rows, columns=confusion_columns)


def _centroid_holdout_metrics(
    x,
    labels: np.ndarray,
    *,
    seed: int = 42,
    partitions: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Backwards-compatible metrics-only route holdout evaluation."""
    metrics, _ = _holdout_metrics_and_confusion(
        x, labels, seed=seed, partitions=partitions,
    )
    return metrics


def route_representation_embeddings(
    cache: FeatureCache,
    bundle: AnalysisBundle,
    output: Path,
    *,
    max_labels: int = 8,
    max_frames_per_phone: int = 160,
    max_utts_per_speaker: int = 40,
    min_utts_per_speaker: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Embed route-restricted L/P vectors for phones and speakers.

    Phone plots use held-out test frame vectors. Speaker plots use held-out test
    utterance vectors. A route-neutral full-SAE margin on the probe-training
    partition selects one common label set, then L and P are compared on exactly
    the same observations and evaluation partition.
    """
    rng = np.random.default_rng(seed)
    metadata = cache_metadata(bundle, cache)
    tables = output / "tables"

    phone_rows = []
    separation_rows: list[dict[str, Any]] = []
    probe_confusion_rows: list[pd.DataFrame] = []
    phones = np.asarray(cache.phones).astype("U32")
    _, test_frame_mask, test_split_summary = _score_masks(cache, bundle, "test")
    aligned = (np.char.upper(phones) != "<UNALIGNED>") & test_frame_mask
    phone_counts = pd.Series(phones[aligned]).value_counts()
    candidate_phones = [str(p) for p, n in phone_counts.items() if int(n) >= max(20, max_frames_per_phone // 4)]
    candidate_phones = sorted(candidate_phones)
    sampled_frames = []
    sampled_labels = []
    for phone in candidate_phones:
        ids = np.flatnonzero(aligned & (phones == phone))
        if len(ids) > max_frames_per_phone:
            ids = np.sort(rng.choice(ids, size=max_frames_per_phone, replace=False))
        sampled_frames.extend(ids.tolist())
        sampled_labels.extend([phone] * len(ids))
    sampled_frames = np.asarray(sampled_frames, dtype=int)
    sampled_labels = np.asarray(sampled_labels, dtype="U32")
    selected_phone_labels: list[str] = []
    if len(sampled_frames):
        neutral_x = _full_sparse_matrix(cache, sampled_frames)
        sampled_partitions = _stratified_partitions(sampled_labels, seed=seed)
        selection_mask = sampled_partitions == "train"
        neutral_margins = _highdim_label_margins(
            neutral_x[selection_mask], sampled_labels[selection_mask]
        ).rename(columns={"label": "phone"})
        selected_phone_labels = neutral_margins.head(max_labels)["phone"].astype(str).tolist()
        keep_labels = np.isin(sampled_labels, selected_phone_labels)
        selected_frames = sampled_frames[keep_labels]
        selected_labels = sampled_labels[keep_labels]
        selected_partitions = sampled_partitions[keep_labels]
        for route_id in [r for r in (0, 1) if np.any(cache.route == r)]:
            route = ROUTE_NAMES.get(route_id, str(route_id))
            x = _route_sparse_matrix(cache, selected_frames, route_id)
            emb = _embed_matrix(x, seed=seed)
            umap_emb = _embed_umap(x, seed=seed)
            tmp = pd.DataFrame({
                "route": route,
                "route_id": route_id,
                "phone": selected_labels.astype(str),
                "frame": selected_frames.astype(int),
                "probe_partition": selected_partitions.astype(str),
                "x": emb[:, 0],
                "y": emb[:, 1],
                "umap_x": umap_emb[:, 0],
                "umap_y": umap_emb[:, 1],
            })
            tmp = tmp.merge(neutral_margins, on="phone", how="left")
            tmp["selected_for_plot"] = True
            phone_rows.append(tmp)
            route_metrics, route_confusion = _holdout_metrics_and_confusion(
                x, selected_labels, seed=seed, partitions=selected_partitions,
            )
            separation_rows.append({
                "target": "phone", "route": route, **route_metrics,
            })
            if len(route_confusion):
                route_confusion.insert(0, "route", route)
                route_confusion.insert(0, "target", "phone")
                probe_confusion_rows.append(route_confusion)
    phone_embedding = pd.concat(phone_rows, ignore_index=True) if phone_rows else pd.DataFrame()

    speaker_rows = []
    selected_speaker_labels: list[str] = []
    if "speaker_id" in metadata.columns and len(metadata):
        test_name = str(bundle.spec.split_map.get("test", "test"))
        test_utterances = metadata["split"].astype(str).to_numpy() == test_name
        speaker_labels_all = metadata["speaker_id"].fillna("<missing>").astype(str).to_numpy()
        speaker_labels = speaker_labels_all[test_utterances]
        test_utt_ids = np.flatnonzero(test_utterances)
        counts = pd.Series(speaker_labels).value_counts()
        candidate_speakers = [
            str(s) for s, n in counts.items() if int(n) >= int(min_utts_per_speaker)
        ]
        sampled_utts = []
        sampled_speakers = []
        for speaker in candidate_speakers:
            ids = np.flatnonzero(speaker_labels == speaker)
            if len(ids) > max_utts_per_speaker:
                ids = np.sort(rng.choice(ids, size=max_utts_per_speaker, replace=False))
            sampled_utts.extend(test_utt_ids[ids].tolist())
            sampled_speakers.extend([speaker] * len(ids))
        sampled_utts = np.asarray(sampled_utts, dtype=int)
        sampled_speakers = np.asarray(sampled_speakers, dtype="U64")
        if len(sampled_utts):
            neutral_x = cache.pooled_z[sampled_utts].astype(np.float32)
            sampled_partitions = _stratified_partitions(sampled_speakers, seed=seed)
            selection_mask = sampled_partitions == "train"
            neutral_margins = _highdim_label_margins(
                neutral_x[selection_mask], sampled_speakers[selection_mask]
            ).rename(columns={"label": "speaker_id"})
            selected_speaker_labels = neutral_margins.head(max_labels)["speaker_id"].astype(str).tolist()
            keep_labels = np.isin(sampled_speakers, selected_speaker_labels)
            selected_utts = sampled_utts[keep_labels]
            selected_speakers = sampled_speakers[keep_labels]
            selected_partitions = sampled_partitions[keep_labels]
            for route_id in [r for r in (0, 1) if np.any(cache.route == r)]:
                route = ROUTE_NAMES.get(route_id, str(route_id))
                route_units = np.flatnonzero(cache.route == route_id)
                x = cache.pooled_z[selected_utts][:, route_units].astype(np.float32)
                emb = _embed_matrix(x, seed=seed)
                umap_emb = _embed_umap(x, seed=seed)
                tmp = pd.DataFrame({
                    "route": route,
                    "route_id": route_id,
                    "speaker_id": selected_speakers.astype(str),
                    "utterance_index": selected_utts.astype(int),
                    "utterance_id": cache.utterance_ids[selected_utts].astype(str),
                    "probe_partition": selected_partitions.astype(str),
                    "x": emb[:, 0],
                    "y": emb[:, 1],
                    "umap_x": umap_emb[:, 0],
                    "umap_y": umap_emb[:, 1],
                })
                tmp = tmp.merge(neutral_margins, on="speaker_id", how="left")
                tmp["selected_for_plot"] = True
                speaker_rows.append(tmp)
                route_metrics, route_confusion = _holdout_metrics_and_confusion(
                    x, selected_speakers, seed=seed, partitions=selected_partitions,
                )
                separation_rows.append({
                    "target": "speaker_id", "route": route, **route_metrics,
                })
                if len(route_confusion):
                    route_confusion.insert(0, "route", route)
                    route_confusion.insert(0, "target", "speaker_id")
                    probe_confusion_rows.append(route_confusion)
    speaker_embedding = pd.concat(speaker_rows, ignore_index=True) if speaker_rows else pd.DataFrame()
    separation = pd.DataFrame(separation_rows)
    probe_confusion = (
        pd.concat(probe_confusion_rows, ignore_index=True)
        if probe_confusion_rows else
        pd.DataFrame(columns=[
            "target", "route", "true_label", "predicted_label", "count",
            "true_label_count", "row_fraction",
        ])
    )

    phone_embedding.to_csv(tables / "route_phone_representation_embedding.csv", index=False)
    speaker_embedding.to_csv(tables / "route_speaker_representation_embedding.csv", index=False)
    separation.to_csv(tables / "route_representation_separation.csv", index=False)
    probe_confusion.to_csv(tables / "route_probe_confusion.csv", index=False)
    summary = {
        "method": "route_restricted_centered_pca_2d",
        "supplementary_method": "route_restricted_umap_2d",
        "umap_parameters": {
            "n_neighbors": 30, "min_dist": 0.1, "metric": "cosine",
            "random_state": int(seed), "n_jobs": 1,
        },
        "label_selection": "probe_train_only_common_route_neutral_full_sae_cosine_margin",
        "evaluation_split": test_split_summary,
        "phone_points": int(len(phone_embedding)),
        "speaker_points": int(len(speaker_embedding)),
        "selected_phones": selected_phone_labels,
        "selected_speakers": selected_speaker_labels,
        "separation": separation.to_dict(orient="records"),
        "probe_confusion_metric": "heldout_row_normalized_frozen_linear_probe_predictions",
        "max_labels": int(max_labels),
        "max_frames_per_phone": int(max_frames_per_phone),
        "max_utts_per_speaker": int(max_utts_per_speaker),
        "min_utts_per_speaker": int(min_utts_per_speaker),
    }
    write_json(output / "route_representation_embeddings.json", summary)
    return phone_embedding, speaker_embedding, separation, probe_confusion, summary


def _controlled_pair_indices(
    labels: np.ndarray,
    nuisance: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match each anchor to same/different labels while changing the nuisance."""
    labels = np.asarray(labels).astype(str)
    nuisance = np.asarray(nuisance).astype(str)
    clusters = np.asarray(clusters).astype(str)
    rng = np.random.default_rng(seed)
    anchors, same_partners, different_partners = [], [], []
    all_ids = np.arange(len(labels), dtype=int)
    for anchor in all_ids:
        independent = (clusters != clusters[anchor]) & (nuisance != nuisance[anchor])
        same = all_ids[independent & (labels == labels[anchor])]
        different = all_ids[independent & (labels != labels[anchor])]
        if not len(same) or not len(different):
            continue
        anchors.append(int(anchor))
        same_partners.append(int(rng.choice(same)))
        different_partners.append(int(rng.choice(different)))
    return (
        np.asarray(anchors, dtype=int),
        np.asarray(same_partners, dtype=int),
        np.asarray(different_partners, dtype=int),
    )


def _paired_cosines(x, anchors: np.ndarray, same: np.ndarray, different: np.ndarray):
    normalized = _row_l2_normalize(x)
    if hasattr(normalized, "multiply"):
        same_similarity = np.asarray(
            normalized[anchors].multiply(normalized[same]).sum(axis=1)
        ).reshape(-1)
        different_similarity = np.asarray(
            normalized[anchors].multiply(normalized[different]).sum(axis=1)
        ).reshape(-1)
    else:
        same_similarity = np.sum(normalized[anchors] * normalized[same], axis=1)
        different_similarity = np.sum(normalized[anchors] * normalized[different], axis=1)
    return same_similarity.astype(float), different_similarity.astype(float)


def _cluster_bootstrap_interval(
    delta: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int,
    repetitions: int = 1000,
) -> tuple[float | None, float | None, int]:
    frame = pd.DataFrame({"delta": np.asarray(delta, dtype=float), "cluster": np.asarray(clusters).astype(str)})
    cluster_means = frame.groupby("cluster", sort=True)["delta"].mean().to_numpy(dtype=float)
    if len(cluster_means) < 2:
        return None, None, int(len(cluster_means))
    rng = np.random.default_rng(seed)
    draws = rng.choice(cluster_means, size=(int(repetitions), len(cluster_means)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high), int(len(cluster_means))


def _rank_auc(same_similarity: np.ndarray, different_similarity: np.ndarray) -> float | None:
    if not len(same_similarity) or not len(different_similarity):
        return None
    from scipy.stats import rankdata
    combined = np.concatenate([same_similarity, different_similarity])
    ranks = rankdata(combined, method="average")[:len(same_similarity)]
    n_same, n_different = len(same_similarity), len(different_similarity)
    u = float(ranks.sum() - n_same * (n_same + 1) / 2)
    return float(u / max(n_same * n_different, 1))


def classifier_free_route_geometry(
    cache: FeatureCache,
    bundle: AnalysisBundle,
    output: Path,
    *,
    seed: int = 42,
    max_frames_per_phone: int = 160,
    bootstrap_repetitions: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Controlled full-space cosine geometry without fitting a classifier.

    Phone pairs always cross speakers and utterances. Speaker pairs always cross
    transcript/content and utterances. Identical anchor/partner pairs are used
    for L and P so route differences are paired rather than sampling artifacts.
    """
    metadata = cache_metadata(bundle, cache)
    rng = np.random.default_rng(seed)
    pair_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    observation_sets: list[dict[str, Any]] = []
    phones = np.asarray(cache.phones).astype("U32")
    _, test_frame_mask, test_summary = _score_masks(cache, bundle, "test")
    aligned = test_frame_mask & (np.char.upper(phones) != "<UNALIGNED>")
    counts = pd.Series(phones[aligned]).value_counts()
    supported_phones = sorted([str(phone) for phone, count in counts.items() if int(count) >= 40])
    phone_frames, phone_labels = [], []
    for phone in supported_phones:
        ids = np.flatnonzero(aligned & (phones == phone))
        if len(ids) > max_frames_per_phone:
            ids = np.sort(rng.choice(ids, size=max_frames_per_phone, replace=False))
        phone_frames.extend(ids.tolist())
        phone_labels.extend([phone] * len(ids))
    phone_frames = np.asarray(phone_frames, dtype=int)
    phone_labels = np.asarray(phone_labels, dtype="U32")
    if len(phone_frames):
        phone_utterance_rows = frame_to_utterance(cache, phone_frames)
        speaker_nuisance = metadata.iloc[phone_utterance_rows]["speaker_id"].fillna("<missing>").astype(str).to_numpy()
        phone_clusters = cache.utterance_ids[phone_utterance_rows].astype(str)
        anchors, same, different = _controlled_pair_indices(
            phone_labels, speaker_nuisance, phone_clusters, seed=seed + 101,
        )
        observation_sets.append({
            "target": "phone", "row_ids": phone_frames, "labels": phone_labels,
            "nuisance": speaker_nuisance, "clusters": phone_clusters,
            "anchors": anchors, "same": same, "different": different,
            "observation_ids": phone_frames.astype(str),
        })

    test_name = str(bundle.spec.split_map.get("test", "test"))
    split = metadata["split"].astype(str).to_numpy()
    speaker_labels_all = metadata["speaker_id"].fillna("<missing>").astype(str).to_numpy()
    test_utterances = np.flatnonzero(split == test_name)
    speaker_counts = pd.Series(speaker_labels_all[test_utterances]).value_counts()
    supported_speakers = set(speaker_counts[speaker_counts >= 2].index.astype(str).tolist())
    speaker_utterances = np.asarray(
        [ui for ui in test_utterances if speaker_labels_all[ui] in supported_speakers], dtype=int,
    )
    if len(speaker_utterances):
        speaker_labels = speaker_labels_all[speaker_utterances]
        if "transcript" in metadata.columns:
            content_nuisance = metadata.iloc[speaker_utterances]["transcript"].fillna("").astype(str).to_numpy()
        else:
            content_nuisance = cache.utterance_ids[speaker_utterances].astype(str)
        empty_content = np.char.str_len(content_nuisance.astype(str)) == 0
        content_nuisance[empty_content] = cache.utterance_ids[speaker_utterances][empty_content].astype(str)
        speaker_clusters = cache.utterance_ids[speaker_utterances].astype(str)
        anchors, same, different = _controlled_pair_indices(
            speaker_labels, content_nuisance, speaker_clusters, seed=seed + 202,
        )
        observation_sets.append({
            "target": "speaker_id", "row_ids": speaker_utterances,
            "labels": speaker_labels, "nuisance": content_nuisance,
            "clusters": speaker_clusters, "anchors": anchors,
            "same": same, "different": different,
            "observation_ids": cache.utterance_ids[speaker_utterances].astype(str),
        })

    for observation in observation_sets:
        target = str(observation["target"])
        anchors = observation["anchors"]
        same = observation["same"]
        different = observation["different"]
        if not len(anchors):
            continue
        for route_id in [route_id for route_id in (0, 1) if np.any(cache.route == route_id)]:
            route = ROUTE_NAMES.get(route_id, str(route_id))
            if target == "phone":
                x = _route_sparse_matrix(cache, observation["row_ids"], route_id)
            else:
                route_units = np.flatnonzero(cache.route == route_id)
                x = cache.pooled_z[observation["row_ids"]][:, route_units].astype(np.float32)
            same_similarity, different_similarity = _paired_cosines(x, anchors, same, different)
            delta = same_similarity - different_similarity
            cluster_values = observation["clusters"][anchors]
            ci_low, ci_high, cluster_count = _cluster_bootstrap_interval(
                delta, cluster_values, seed=seed + 1000 * route_id + (1 if target == "phone" else 2),
                repetitions=bootstrap_repetitions,
            )
            pair_rows.append(pd.DataFrame({
                "target": target,
                "route": route,
                "anchor_observation": observation["observation_ids"][anchors],
                "anchor_cluster": cluster_values,
                "anchor_label": observation["labels"][anchors].astype(str),
                "same_partner_observation": observation["observation_ids"][same],
                "different_partner_observation": observation["observation_ids"][different],
                "different_partner_label": observation["labels"][different].astype(str),
                "same_cosine": same_similarity,
                "different_cosine": different_similarity,
                "paired_difference": delta,
            }))
            summary_rows.append({
                "target": target,
                "route": route,
                "labels": int(len(np.unique(observation["labels"][anchors]))),
                "controlled_pairs": int(len(anchors)),
                "bootstrap_clusters": int(cluster_count),
                "mean_same_cosine": float(np.mean(same_similarity)),
                "mean_different_cosine": float(np.mean(different_similarity)),
                "paired_cosine_difference": float(np.mean(delta)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "paired_same_greater_fraction": float(np.mean(same_similarity > different_similarity)),
                "rank_auc": _rank_auc(same_similarity, different_similarity),
            })

    pairs = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    summary_table = pd.DataFrame(summary_rows)
    tables = output / "tables"
    pairs.to_csv(tables / "route_classifier_free_geometry_pairs.csv", index=False)
    summary_table.to_csv(tables / "route_classifier_free_geometry_summary.csv", index=False)
    summary = {
        "metric": "controlled_same_minus_different_cosine",
        "classifier_trained": False,
        "phone_control": "same/different phone pairs cross both speaker and utterance",
        "speaker_control": "same/different speaker pairs cross transcript/content and utterance",
        "bootstrap": "anchor-cluster bootstrap of paired cosine differences",
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "evaluation_split": test_summary,
        "rows": summary_table.to_dict(orient="records"),
    }
    write_json(output / "classifier_free_geometry.json", summary)
    return pairs, summary_table, summary


_PHONE_IDENTITY_FACTORS = ("phone", "manner", "place", "phonetic_voicing")
_PHONE_TIMING_FACTORS = ("phone_boundary", "phone_transition")
_PHONE_LIKE_FACTORS = _PHONE_IDENTITY_FACTORS
_PHONE_SELECTIVITY_MIN_AUPRC_GAIN = 0.02
_PROSODY_LIKE_FACTORS = ("f0", "energy", "voicing")
_METADATA_LIKE_FACTORS = ("sex", "gender", "dialect_region", "age")


def _max_existing(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    present = [c for c in columns if c in frame.columns]
    if not present:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return frame[present].max(axis=1).astype(float)


def _significant_units(
    scores: pd.DataFrame, factors: tuple[str, ...], *, q_threshold: float,
    score_threshold: float, min_auprc_gain: float = 0.0,
    min_positive_auroc: float = 0.0,
) -> set[int]:
    if scores is None or scores.empty:
        return set()
    sub = scores[scores["factor"].isin(factors)].copy()
    if sub.empty:
        return set()
    # Categorical frame labels contain many correlated observations from the
    # same utterance. Their frame-wise z/FDR columns are useful diagnostics,
    # but are not valid independent-sample evidence and can make negligible
    # effects look decisive. Headline categorical selections therefore use a
    # directional practical effect: frame-activity AUROC for phones and
    # mean-amplitude correlation for utterance factors. Continuous factors
    # retain the q/score rule.
    utterance_association = (
        "amplitude_r_positive" in sub.columns
        and sub.get("metric", pd.Series("", index=sub.index)).eq(
            "utterance_mean_activation"
        ).any()
        and sub["amplitude_r_positive"].notna().any()
    )
    effect_column = "amplitude_r_positive" if utterance_association else "active_auroc_positive"
    categorical = effect_column in sub.columns and sub[effect_column].notna().any()
    if categorical:
        practical_threshold = max(float(min_positive_auroc), 0.05)
        sub = sub[sub[effect_column] >= practical_threshold]
    elif "q" in sub:
        sub = sub[(sub["q"] <= q_threshold) & (sub["score"].abs() >= 3.0)]
    else:
        sub = sub[sub["score"].abs() >= score_threshold]
    if min_auprc_gain > 0 and {"auprc", "prevalence"}.issubset(sub.columns):
        # On millions of frames, z/FDR alone can make tiny practical effects
        # look important.  For phone identity, require that a unit's active
        # frames predict the label better than the base-rate classifier.
        sub = sub[(sub["auprc"] - sub["prevalence"]) >= min_auprc_gain]
    if not categorical and min_positive_auroc > 0 and "active_auroc_positive" in sub.columns:
        sub = sub[sub["active_auroc_positive"] >= min_positive_auroc]
    return set(sub["unit"].astype(int).tolist())


def disentanglement_tables(
    health: pd.DataFrame | None,
    profiles: pd.DataFrame,
    scores: pd.DataFrame,
    output: Path,
    *,
    score_threshold: float = 5.0,
    q_threshold: float = 0.05,
    focus: str = "speaker_content",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Create thesis-facing unit tables for routed disentanglement.

    The raw selectivity table is intentionally broad.  This helper condenses it
    into the views we actually need for the dissertation claim: phone-like
    units, speaker-like units, prosody-like units, mixed units, and route
    violations such as speaker-selective L units or phone-selective P units.
    """
    units = profiles.copy()
    units["unit"] = units["unit"].astype(int)
    if health is not None and len(health):
        keep = [
            "unit", "frame_frequency", "utterance_frequency", "decoder_norm",
            "mean_abs_contribution", "observed_active", "unobserved",
            "train_like_dead", "dead", "rare", "ubiquitous",
            "final_inactive_batches", "max_inactive_batches", "fired_batches",
        ]
        keep = [c for c in keep if c in health.columns]
        units = units.drop(columns=[c for c in keep if c in units.columns and c != "unit"], errors="ignore")
        units = units.merge(health[keep], on="unit", how="left")

    for col in ("linguistic_score", "paralinguistic_score", "delta_L_minus_P"):
        if col not in units:
            units[col] = 0.0

    factor_score_cols = {c[:-7]: c for c in units.columns if c.endswith("__score")}
    phone_cols = tuple(f"{f}__score" for f in _PHONE_LIKE_FACTORS)
    prosody_cols = tuple(f"{f}__score" for f in _PROSODY_LIKE_FACTORS)
    units["phone_like_score"] = (
        units["PhoneScore"].astype(float) if "PhoneScore" in units
        else _max_existing(units, phone_cols)
    )
    units["speaker_score"] = units.get("SpeakerScore", units.get("speaker_id__score", 0.0))
    units["speaker_score"] = units["speaker_score"].astype(float)
    units["prosody_score"] = _max_existing(units, prosody_cols)
    metadata_cols = tuple(f"{f}__score" for f in _METADATA_LIKE_FACTORS)
    units["metadata_paralinguistic_score"] = _max_existing(units, metadata_cols)
    units["emotion_score"] = units.get("emotion__score", 0.0)
    units["emotion_score"] = units["emotion_score"].astype(float)
    units["max_factor_score"] = _max_existing(units, tuple(factor_score_cols.values()))
    units["dominant_family"] = np.where(
        units["linguistic_score"] > units["paralinguistic_score"],
        "linguistic",
        np.where(units["paralinguistic_score"] > units["linguistic_score"], "paralinguistic", "tie"),
    )

    sig_phone = _significant_units(
        scores, _PHONE_LIKE_FACTORS, q_threshold=q_threshold,
        score_threshold=score_threshold,
        min_auprc_gain=_PHONE_SELECTIVITY_MIN_AUPRC_GAIN,
        min_positive_auroc=0.05,
    )
    sig_speaker = _significant_units(scores, ("speaker_id",), q_threshold=q_threshold,
                                     score_threshold=score_threshold,
                                     min_positive_auroc=0.10)
    sig_prosody = _significant_units(scores, _PROSODY_LIKE_FACTORS, q_threshold=q_threshold,
                                     score_threshold=score_threshold)
    sig_emotion = _significant_units(scores, ("emotion",), q_threshold=q_threshold,
                                     score_threshold=score_threshold)
    sig_metadata = _significant_units(scores, _METADATA_LIKE_FACTORS, q_threshold=q_threshold,
                                      score_threshold=score_threshold)

    # The focused phone/speaker table already defines "highly associated" as
    # the upper empirical tail among units with positive evidence. Reuse that
    # ranking here instead of declaring a unit selective when *any* one-vs-rest
    # speaker comparison clears a small fixed effect threshold. The latter is
    # badly inflated when taking the maximum over many speakers, especially in
    # quick mode. Raw per-level scores remain available in unit_factor_scores.
    ranked_categories = "category" in units.columns and units["category"].notna().any()
    if ranked_categories:
        categorical_selection = "top_decile_positive_phone_or_speaker_effect"
        units["phone_selective"] = units["category"].isin(["phone", "entangled"])
        units["speaker_selective"] = units["category"].isin(["speaker", "entangled"])
    elif scores is not None and not scores.empty:
        categorical_selection = "positive_phone_auroc_or_utterance_amplitude_r"
        units["phone_selective"] = units["unit"].isin(sig_phone)
        units["speaker_selective"] = units["unit"].isin(sig_speaker)
    else:
        categorical_selection = "profile_score_threshold"
        units["phone_selective"] = units["phone_like_score"].ge(score_threshold)
        units["speaker_selective"] = units["speaker_score"].ge(score_threshold)
    units["prosody_selective"] = (
        units["prosody_score"].ge(score_threshold) |
        units["unit"].isin(sig_prosody)
    )
    units["emotion_selective"] = (
        units["emotion_score"].ge(score_threshold) |
        units["unit"].isin(sig_emotion)
    )
    units["metadata_paralinguistic_selective"] = (
        units["metadata_paralinguistic_score"].ge(score_threshold) |
        units["unit"].isin(sig_metadata)
    )
    units["paralinguistic_selective"] = (
        units["speaker_selective"] | units["prosody_selective"] |
        units["emotion_selective"] | units["metadata_paralinguistic_selective"] |
        units["paralinguistic_score"].ge(score_threshold)
    )
    units["linguistic_selective"] = (
        units["phone_selective"] | units["linguistic_score"].ge(score_threshold)
    )
    units["mixed_phone_speaker"] = units["phone_selective"] & units["speaker_selective"]
    units["mixed_linguistic_paralinguistic"] = (
        units["linguistic_selective"] & units["paralinguistic_selective"]
    )

    focus = str(focus or "speaker_content").lower().replace("-", "_")
    if focus not in {"speaker_content", "broad"}:
        raise ValueError(f"Unknown disentanglement focus: {focus}")

    units["route_violation"] = False
    if focus == "speaker_content":
        # Current Libri adversarial runs are about content vs speaker.  Energy,
        # voicing and other acoustic correlates remain diagnostic columns, but
        # they are not counted as headline route failures here.
        units.loc[(units["route"] == "L") & units["speaker_selective"], "route_violation"] = True
        units.loc[(units["route"] == "P") & units["phone_selective"], "route_violation"] = True
        units.loc[(units["route"] == "U") & (units["speaker_selective"] | units["phone_selective"]),
                  "route_violation"] = True
    else:
        # Broad nuisance validation mode: useful for TIMIT-style analyses where
        # sex, dialect-region, prosody, emotion, etc. are explicit nuisance
        # factors.  This is stricter than the Libri speaker/content headline.
        units.loc[(units["route"] == "L") & units["paralinguistic_selective"], "route_violation"] = True
        units.loc[(units["route"] == "P") & units["phone_selective"], "route_violation"] = True
        units.loc[(units["route"] == "U") & (units["linguistic_selective"] | units["paralinguistic_selective"]),
                  "route_violation"] = True

    def tags(row: pd.Series) -> str:
        out: list[str] = []
        if row.get("mixed_phone_speaker", False): out.append("mixed_phone_speaker")
        if row.get("mixed_linguistic_paralinguistic", False): out.append("mixed_linguistic_paralinguistic")
        if row.get("route") == "L" and row.get("speaker_selective", False): out.append("speaker_in_L")
        if focus == "broad":
            if row.get("route") == "L" and row.get("prosody_selective", False): out.append("prosody_in_L")
            if row.get("route") == "L" and row.get("emotion_selective", False): out.append("emotion_in_L")
            if row.get("route") == "L" and row.get("metadata_paralinguistic_selective", False): out.append("metadata_in_L")
            if row.get("route") == "L" and row.get("paralinguistic_selective", False): out.append("paralinguistic_in_L")
        if row.get("route") == "P" and row.get("phone_selective", False): out.append("phone_in_P")
        if row.get("route") == "U" and row.get("linguistic_selective", False): out.append("linguistic_in_U")
        if focus == "broad" and row.get("route") == "U" and row.get("paralinguistic_selective", False):
            out.append("paralinguistic_in_U")
        # Preserve order while removing duplicates such as speaker_in_L and the
        # broader paralinguistic_in_L both firing.
        return ";".join(dict.fromkeys(out))

    units["issue_tags"] = units.apply(tags, axis=1)
    units["leakage_score"] = 0.0
    if focus == "speaker_content":
        units.loc[units.route == "L", "leakage_score"] = units.loc[units.route == "L", "speaker_score"]
    else:
        units.loc[units.route == "L", "leakage_score"] = units.loc[units.route == "L", "paralinguistic_score"]
    units.loc[units.route == "P", "leakage_score"] = units.loc[units.route == "P", "phone_like_score"]
    units.loc[units.route == "U", "leakage_score"] = units.loc[
        units.route == "U", ["linguistic_score", "paralinguistic_score"]
    ].max(axis=1)

    leaky = units[(units["issue_tags"] != "") | units["route_violation"]].copy()
    leaky = leaky.sort_values(["leakage_score", "max_factor_score"], ascending=False)

    route_rows: list[dict[str, Any]] = []
    for route, group in units.groupby("route", dropna=False):
        if "observed_active" in group:
            active = group[group["observed_active"].fillna(False)]
        elif "dead" in group:
            active = group[~group.get("dead", False).fillna(False)]
        else:
            active = group
        active_denom = len(active)
        def active_fraction(column: str) -> float:
            return float(active[column].mean()) if active_denom else 0.0
        route_rows.append({
            "route": route,
            "units": int(len(group)),
            "active_units": int(len(active)),
            "unobserved_fraction": float(group.get("unobserved", False).fillna(False).mean()) if "unobserved" in group else 0.0,
            "train_like_dead_fraction": float(group.get("train_like_dead", False).fillna(False).mean()) if "train_like_dead" in group else 0.0,
            "dead_fraction": float(group.get("train_like_dead", False).fillna(False).mean()) if "train_like_dead" in group else (
                float(group.get("dead", False).fillna(False).mean()) if "dead" in group else 0.0
            ),
            # Headline fractions use observed-active units. Assigned-unit
            # fractions remain available explicitly for capacity accounting.
            "phone_selective_fraction": active_fraction("phone_selective"),
            "speaker_selective_fraction": active_fraction("speaker_selective"),
            "prosody_selective_fraction": active_fraction("prosody_selective"),
            "metadata_paralinguistic_selective_fraction": active_fraction("metadata_paralinguistic_selective"),
            "mixed_phone_speaker_fraction": active_fraction("mixed_phone_speaker"),
            "mixed_any_fraction": active_fraction("mixed_linguistic_paralinguistic"),
            "route_violation_fraction": active_fraction("route_violation"),
            "active_phone_selective_fraction": active_fraction("phone_selective"),
            "active_speaker_selective_fraction": active_fraction("speaker_selective"),
            "assigned_phone_selective_fraction": float(group["phone_selective"].mean()),
            "assigned_speaker_selective_fraction": float(group["speaker_selective"].mean()),
            "assigned_route_violation_fraction": float(group["route_violation"].mean()),
            "mean_phone_like_score": float(group["phone_like_score"].mean()),
            "mean_speaker_score": float(group["speaker_score"].mean()),
            "mean_linguistic_score": float(group["linguistic_score"].mean()),
            "mean_paralinguistic_score": float(group["paralinguistic_score"].mean()),
            "median_frame_frequency": float(group.get("frame_frequency", pd.Series([0.0])).median()),
        })
    route_summary = pd.DataFrame(route_rows).sort_values("route")

    def route_value(route: str, column: str) -> float | None:
        row = route_summary[route_summary.route == route]
        if row.empty or column not in row:
            return None
        return float(row.iloc[0][column])

    if "observed_active" in units:
        active_units_all = units[units["observed_active"].fillna(False)]
    else:
        active_units_all = units
    thesis = pd.DataFrame([{
        "focus": focus,
        "score_threshold": score_threshold,
        "q_threshold": q_threshold,
        "categorical_selection": categorical_selection,
        "L_phone_selective_fraction": route_value("L", "phone_selective_fraction"),
        "L_speaker_leak_fraction": route_value("L", "speaker_selective_fraction"),
        "L_paralinguistic_leak_fraction": route_value("L", "route_violation_fraction"),
        "L_speaker_content_leak_fraction": route_value("L", "route_violation_fraction"),
        "P_speaker_selective_fraction": route_value("P", "speaker_selective_fraction"),
        "P_phone_leak_fraction": route_value("P", "phone_selective_fraction"),
        "U_info_fraction": route_value("U", "route_violation_fraction"),
        "denominator": "observed_active_units",
        "mixed_phone_speaker_fraction_all": float(active_units_all["mixed_phone_speaker"].mean()) if len(active_units_all) else 0.0,
        "mixed_any_fraction_all": float(active_units_all["mixed_linguistic_paralinguistic"].mean()) if len(active_units_all) else 0.0,
        "route_violation_fraction_all": float(active_units_all["route_violation"].mean()) if len(active_units_all) else 0.0,
        "leaky_units": int(len(leaky)),
    }])

    units.to_csv(output / "tables" / "unit_disentanglement.csv", index=False)
    leaky.to_csv(output / "tables" / "leaky_units.csv", index=False)
    route_summary.to_csv(output / "tables" / "route_disentanglement_summary.csv", index=False)
    thesis.to_csv(output / "tables" / "thesis_disentanglement_summary.csv", index=False)
    try:
        units.to_parquet(output / "tables" / "unit_disentanglement.parquet", index=False)
        leaky.to_parquet(output / "tables" / "leaky_units.parquet", index=False)
    except Exception:
        pass
    leaky_preview_cols = [
        c for c in (
            "unit", "route", "issue_tags", "phone_like_score",
            "speaker_score", "prosody_score", "metadata_paralinguistic_score",
            "paralinguistic_score", "leakage_score",
            "frame_frequency",
        ) if c in leaky.columns
    ]
    summary = {
        "score_threshold": score_threshold,
        "q_threshold": q_threshold,
        "categorical_selection": categorical_selection,
        "categorical_frame_q_values": "diagnostic_only",
        "phone_selectivity_min_auprc_gain": _PHONE_SELECTIVITY_MIN_AUPRC_GAIN,
        "focus": focus,
        "route_summary": route_summary.to_dict(orient="records"),
        "thesis_summary": thesis.iloc[0].to_dict(),
        "top_leaky_units": leaky.head(25)[leaky_preview_cols].to_dict(orient="records") if len(leaky) else [],
    }
    write_json(output / "disentanglement.json", summary)
    return units, leaky, route_summary, summary


def _kmeans(x: np.ndarray, k: int, seed: int = 42, steps: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(x) < k:
        k = max(1, len(x))
    centers = x[rng.choice(len(x), k, replace=False)].copy()
    labels = np.zeros(len(x), dtype=np.int32)
    for _ in range(steps):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = dist.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if np.any(labels == j):
                centers[j] = x[labels == j].mean(0)
    return labels, centers


def _entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return max(0.0, float(-(p * np.log(np.maximum(p, 1e-12))).sum()))


def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    ha, hb = _entropy(a), _entropy(b)
    mi = 0.0
    for av in np.unique(a):
        for bv in np.unique(b):
            pab = np.mean((a == av) & (b == bv))
            if pab:
                mi += pab * math.log(pab / (np.mean(a == av) * np.mean(b == bv)))
    return float(mi / max(math.sqrt(max(ha * hb, 0.0)), 1e-12))


def clustering_analysis(
    cache: FeatureCache, profiles: pd.DataFrame, output: Path, seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    if {"PhoneScore", "SpeakerScore"}.issubset(profiles.columns):
        numeric = ["PhoneScore", "SpeakerScore"]
    else:
        numeric = [c for c in profiles.columns if c.endswith("__score")]
        if not numeric:
            numeric = [c for c in ("linguistic_score", "paralinguistic_score") if c in profiles]
    raw = profiles[numeric].to_numpy(dtype=np.float32)
    active = np.linalg.norm(raw, axis=1) > 0
    x = raw.copy()
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    xa = x[active]
    best = None
    if len(xa) >= 2:
        max_k = min(len(xa), 12, max(2, int(math.sqrt(len(xa))) + 1))
        for k in range(2, max_k + 1):
            labels, centers = _kmeans(xa, k, seed)
            inertia = float(((xa - centers[labels]) ** 2).mean())
            penalty = k * xa.shape[1] / max(len(xa), 1)
            score = inertia + penalty
            if best is None or score < best[0]:
                best = (score, labels, centers)
    clusters = np.full(cache.K, -1, dtype=np.int32)
    if best is not None:
        clusters[active] = best[1]
    out = profiles.copy()
    out["cluster"] = clusters
    nmi = _nmi(clusters[active], cache.route[active]) if active.any() else 0.0
    summary = {"clusters": int(len(set(clusters[active]))) if active.any() else 0, "route_nmi": nmi}
    out.to_csv(output / "tables" / "unit_clusters.csv", index=False)
    try: out.to_parquet(output / "tables" / "unit_clusters.parquet", index=False)
    except Exception: pass
    write_json(output / "clustering.json", summary)
    return out, summary


def _cos_rows(x, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xa, xb = x[a], x[b]
    if hasattr(xa, "multiply"):
        numerator = np.asarray(xa.multiply(xb).sum(axis=1)).reshape(-1)
        na = np.sqrt(np.asarray(xa.multiply(xa).sum(axis=1)).reshape(-1))
        nb = np.sqrt(np.asarray(xb.multiply(xb).sum(axis=1)).reshape(-1))
        return numerator / np.maximum(na * nb, 1e-12)
    return (xa * xb).sum(1) / np.maximum(np.linalg.norm(xa, axis=1) * np.linalg.norm(xb, axis=1), 1e-12)


def similarity_analysis(cache: FeatureCache, bundle: AnalysisBundle, output: Path, seed: int = 42) -> dict:
    metadata = cache_metadata(bundle, cache)
    x = cache.pooled_z.astype(np.float32)
    rng = np.random.default_rng(seed)
    results = {}
    for rid, name in ((0, "L"), (1, "P")):
        mask = cache.route == rid
        if not mask.any():
            continue
        xr = x[:, mask]
        if "speaker_id" in metadata:
            speakers = metadata["speaker_id"].astype(str).to_numpy()
            same_a, same_b, diff_a, diff_b = [], [], [], []
            groups = defaultdict(list)
            for i, s in enumerate(speakers): groups[s].append(i)
            for ids in groups.values():
                if len(ids) >= 2:
                    same_a.append(ids[0]); same_b.append(ids[1])
            for i in range(min(len(xr), 2000)):
                j = int(rng.integers(len(xr)))
                if speakers[i] != speakers[j]: diff_a.append(i); diff_b.append(j)
            same = _cos_rows(xr, np.asarray(same_a), np.asarray(same_b)) if same_a else np.zeros(0)
            diff = _cos_rows(xr, np.asarray(diff_a), np.asarray(diff_b)) if diff_a else np.zeros(0)
            results[f"{name}_same_speaker_cos"] = float(same.mean()) if len(same) else None
            results[f"{name}_different_speaker_cos"] = float(diff.mean()) if len(diff) else None
            results[f"{name}_speaker_similarity_gap"] = float(same.mean()-diff.mean()) if len(same) and len(diff) else None
        if "emotion" in metadata:
            emotion = metadata["emotion"].astype(str).to_numpy()
            same_a, same_b, diff_a, diff_b = [], [], [], []
            groups = defaultdict(list)
            for i, e in enumerate(emotion): groups[e].append(i)
            for ids in groups.values():
                if len(ids) >= 2: same_a.append(ids[0]); same_b.append(ids[1])
            for i in range(min(len(xr), 2000)):
                j = int(rng.integers(len(xr)))
                if emotion[i] != emotion[j]: diff_a.append(i); diff_b.append(j)
            same = _cos_rows(xr, np.asarray(same_a), np.asarray(same_b)) if same_a else np.zeros(0)
            diff = _cos_rows(xr, np.asarray(diff_a), np.asarray(diff_b)) if diff_a else np.zeros(0)
            results[f"{name}_emotion_similarity_gap"] = float(same.mean()-diff.mean()) if len(same) and len(diff) else None

    # Frame-matched linguistic test on the bounded, deterministic SPEAR sample.
    sample_frames = cache.h_sample_frames
    if len(sample_frames):
        phone = cache.phones[sample_frames].astype(str)
        valid_phone = np.char.upper(phone.astype("U32")) != "<UNALIGNED>"
        ui = frame_to_utterance(cache, sample_frames)
        speakers = metadata.iloc[ui]["speaker_id"].astype(str).to_numpy() if "speaker_id" in metadata else np.full(len(ui), "")
        for rid, name in ((0, "L"), (1, "P")):
            zf = _route_sparse_matrix(cache, sample_frames, rid)
            same_a, same_b, contrast_a, contrast_b = [], [], [], []
            for i in range(zf.shape[0]):
                if phone[i].upper() == "<UNALIGNED>":
                    continue
                same = np.flatnonzero(valid_phone & (phone == phone[i]) & (speakers != speakers[i]))
                contrast = np.flatnonzero(valid_phone & (phone != phone[i]) & (speakers == speakers[i]))
                if len(same): same_a.append(i); same_b.append(int(rng.choice(same)))
                if len(contrast): contrast_a.append(i); contrast_b.append(int(rng.choice(contrast)))
            a = _cos_rows(zf, np.asarray(same_a), np.asarray(same_b)) if same_a else np.zeros(0)
            b = _cos_rows(zf, np.asarray(contrast_a), np.asarray(contrast_b)) if contrast_a else np.zeros(0)
            results[f"{name}_same_phone_different_speaker_cos"] = float(a.mean()) if len(a) else None
            results[f"{name}_same_speaker_different_phone_cos"] = float(b.mean()) if len(b) else None
            results[f"{name}_phone_similarity_gap"] = float(a.mean()-b.mean()) if len(a) and len(b) else None
    write_json(output / "similarity.json", results)
    return results


def geometry_analysis(cache: FeatureCache, resolved: ResolvedModel, health: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, dict]:
    decoder = resolved.state["sae.dec_weight"].detach().float().cpu().numpy().T
    decoder /= np.maximum(np.linalg.norm(decoder, axis=1, keepdims=True), 1e-12)
    rows = []
    # Chunked exact nearest decoder atoms; avoids materializing K x K.
    n_neighbors = min(5, max(cache.K - 1, 0))
    for start in range(0, cache.K, 256):
        sims = decoder[start:start+256] @ decoder.T
        for local in range(len(sims)):
            unit = start + local
            sims[local, unit] = -np.inf
            if n_neighbors <= 0:
                continue
            near = np.argpartition(sims[local], -n_neighbors)[-n_neighbors:]
            near = near[np.argsort(sims[local, near])[::-1]]
            for rank, other in enumerate(near, 1):
                rows.append({
                    "unit": unit, "neighbor": int(other), "rank": rank,
                    "decoder_cosine": float(sims[local, other]),
                    "unit_route": ROUTE_NAMES.get(int(cache.route[unit]), str(cache.route[unit])),
                    "neighbor_route": ROUTE_NAMES.get(int(cache.route[other]), str(cache.route[other])),
                })
    table = pd.DataFrame(rows)
    cross = table.unit_route != table.neighbor_route if len(table) else np.asarray([], dtype=bool)
    # Empirical coactivation is complementary to decoder geometry. Restrict to
    # frequently active units and a deterministic frame grid to bound memory.
    if health is None or not len(health):
        counts = np.bincount(cache.indices.reshape(-1), minlength=cache.K)
        order = np.argsort(counts)[::-1]
        top = order[counts[order] > 0][:min(1000, cache.K)].astype(int)
    else:
        top = health.sort_values("active_frames", ascending=False).head(min(1000, cache.K)).unit.to_numpy(dtype=int)
    frame_rows = np.linspace(0, max(cache.n_frames-1, 0), min(cache.n_frames, 20000), dtype=int)
    co_rows = []
    if len(top) and len(frame_rows):
        lookup = np.full(cache.K, -1, dtype=int); lookup[top] = np.arange(len(top))
        active = np.zeros((len(frame_rows), len(top)), dtype=np.float32)
        mapped = lookup[cache.indices[frame_rows]]
        rr = np.repeat(np.arange(len(frame_rows)), mapped.shape[1]); cc = mapped.reshape(-1)
        valid = cc >= 0; active[rr[valid], cc[valid]] = 1.0
        intersection = active.T @ active
        freq = np.diag(intersection)
        union = freq[:, None] + freq[None, :] - intersection
        jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        np.fill_diagonal(jaccard, -1)
        for i, unit in enumerate(top):
            other_i = int(np.argmax(jaccard[i])); other = int(top[other_i])
            co_rows.append({"unit": int(unit), "neighbor": other,
                            "coactivation_jaccard": float(jaccard[i, other_i]),
                            "same_route": bool(cache.route[unit] == cache.route[other])})
    co_table = pd.DataFrame(co_rows)
    summary = {
        "mean_nearest_cosine": float(table[table["rank"] == 1].decoder_cosine.mean()) if len(table) else None,
        "cross_route_neighbor_fraction": float(cross.mean()) if len(table) else None,
        "mean_top_coactivation_jaccard": float(co_table.coactivation_jaccard.mean()) if len(co_table) else None,
    }
    table.to_csv(output / "tables" / "decoder_neighbors.csv", index=False)
    co_table.to_csv(output / "tables" / "coactivation_neighbors.csv", index=False)
    try:
        table.to_parquet(output / "tables" / "decoder_neighbors.parquet", index=False)
        co_table.to_parquet(output / "tables" / "coactivation_neighbors.parquet", index=False)
    except Exception:
        pass
    write_json(output / "geometry.json", summary)
    return table, summary
