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
from .utils import bh_fdr, bootstrap_ci, write_json, write_rows


ROUTE_NAMES = {-1: "unassigned", 0: "L", 1: "P", 2: "U"}

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
    bursts = np.zeros(K, dtype=np.int64)
    utterance_count = np.zeros(K, dtype=np.int64)
    for i in range(len(cache.utterance_ids)):
        sl = cache.utterance_slice(i)
        prev: set[int] = set()
        seen: set[int] = set()
        for active in cache.indices[sl]:
            current = set(int(x) for x in active)
            for unit in current - prev:
                bursts[unit] += 1
            prev = current
            seen.update(current)
        if seen:
            utterance_count[np.fromiter(seen, dtype=np.int64)] += 1
    decoder = resolved.state["sae.dec_weight"].detach().float().cpu().numpy()
    decoder_norm = np.linalg.norm(decoder, axis=0)
    mean = np.divide(total, count, out=np.zeros(K), where=count > 0)
    var = np.divide(sq, count, out=np.zeros(K), where=count > 0) - mean ** 2
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
            "decoder_norm": float(decoder_norm[j]), "bursts": int(bursts[j]),
            "mean_burst_frames": float(count[j] / bursts[j]) if bursts[j] else 0.0,
            "dead": bool(count[j] == 0), "rare": bool(0 < count[j] < max(5, cache.n_frames * 1e-5)),
            "ubiquitous": bool(count[j] > 0.5 * cache.n_frames),
        })
    frame = pd.DataFrame(rows)
    route_summary = []
    for rid in sorted(set(cache.route.tolist())):
        subset = frame[frame.route_id == rid]
        route_summary.append({
            "route": ROUTE_NAMES.get(int(rid), str(rid)), "assigned_units": int(len(subset)),
            "active_units": int((~subset.dead).sum()), "dead_fraction": float(subset.dead.mean()),
            "median_frame_frequency": float(subset.frame_frequency.median()),
            "active_slots_per_frame": float(subset.active_frames.sum() / max(cache.n_frames, 1)),
        })
    summary = {
        "frames": cache.n_frames, "utterances": len(cache.utterance_ids), "K": K,
        "active_units": int((count > 0).sum()), "dead_units": int((count == 0).sum()),
        "route_summary": route_summary,
    }
    frame.to_csv(output / "tables" / "units.csv", index=False)
    try:
        frame.to_parquet(output / "tables" / "units.parquet", index=False)
    except Exception:
        pass
    write_json(output / "health.json", summary)
    return frame, summary


def top_examples(cache: FeatureCache, bundle: AnalysisBundle, n: int = 10) -> pd.DataFrame:
    heaps: list[list[tuple[float, int, float]]] = [[] for _ in range(cache.K)]
    reservoirs: list[list[tuple[float, int, float]]] = [[] for _ in range(cache.K)]
    seen = np.zeros(cache.K, dtype=np.int64)
    rng = np.random.default_rng(42)
    import heapq
    for frame in range(cache.n_frames):
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
                "effect": float(effect[unit]), "z": float(z[unit]), "p": float(p[unit]),
                "q": float(q[unit]), "score": float(abs(z[unit])),
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata = cache_metadata(bundle, cache)
    rows: list[dict[str, Any]] = []
    factors = {f.name: f for f in bundle.spec.factors}
    if "phone" in factors:
        rows += _categorical_scores(cache.indices, cache.values.astype(np.float32), cache.phones,
                                    cache.K, "phone", "linguistic", min_count=20)
        for prop in ("manner", "place", "phonetic_voicing"):
            labels = np.asarray([_phone_property(p, prop) for p in cache.phones], dtype="U32")
            rows += _categorical_scores(cache.indices, cache.values.astype(np.float32), labels,
                                        cache.K, prop, "linguistic", min_count=20)
        previous = np.roll(cache.phones, 1)
        previous[cache.offsets] = cache.phones[cache.offsets]
        boundary = np.where(cache.phones != previous, "boundary", "inside")
        transition = np.char.add(np.char.add(previous.astype("U32"), ">"), cache.phones.astype("U32"))
        transition[boundary == "inside"] = "<none>"
        rows += _categorical_scores(cache.indices, cache.values.astype(np.float32), boundary,
                                    cache.K, "phone_boundary", "linguistic", min_count=20)
        rows += _categorical_scores(cache.indices, cache.values.astype(np.float32), transition,
                                    cache.K, "phone_transition", "linguistic", min_count=20)
    for name in ("f0", "energy", "voicing"):
        if name in factors:
            rows += _continuous_scores(cache.indices, cache.values.astype(np.float32), getattr(cache, name),
                                       cache.K, name, "paralinguistic")
    # Utterance factors use the time-pooled SAE activation.
    pooled = cache.pooled_z.astype(np.float32)
    pooled_idx = np.argsort(np.abs(pooled), axis=1)[:, -min(256, cache.K):]
    pooled_val = np.take_along_axis(pooled, pooled_idx, axis=1)
    for factor in bundle.spec.factors:
        if factor.level != "utterance" or factor.source.startswith("computed:"):
            continue
        if factor.source not in metadata:
            continue
        y = metadata[factor.source].fillna("<missing>").to_numpy()
        if factor.kind == "categorical":
            rows += _categorical_scores(pooled_idx, pooled_val, y, cache.K, factor.name, factor.family, min_count=3)
        else:
            rows += _continuous_scores(pooled_idx, pooled_val, pd.to_numeric(metadata[factor.source], errors="coerce"),
                                       cache.K, factor.name, factor.family)
    scores = pd.DataFrame(rows)
    if scores.empty:
        profiles = pd.DataFrame({"unit": np.arange(cache.K), "linguistic_score": 0., "paralinguistic_score": 0.})
    else:
        by_factor = scores.groupby(["unit", "family", "factor"], as_index=False)["score"].max()
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


_PHONE_LIKE_FACTORS = (
    "phone", "manner", "place", "phonetic_voicing",
    "phone_boundary", "phone_transition",
)
_PROSODY_LIKE_FACTORS = ("f0", "energy", "voicing")
_METADATA_LIKE_FACTORS = ("sex", "gender", "dialect_region", "age")


def _max_existing(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    present = [c for c in columns if c in frame.columns]
    if not present:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return frame[present].max(axis=1).astype(float)


def _significant_units(
    scores: pd.DataFrame, factors: tuple[str, ...], *, q_threshold: float,
    score_threshold: float,
) -> set[int]:
    if scores is None or scores.empty:
        return set()
    sub = scores[scores["factor"].isin(factors)].copy()
    if sub.empty:
        return set()
    if "q" in sub:
        sub = sub[(sub["q"] <= q_threshold) & (sub["score"].abs() >= 3.0)]
    else:
        sub = sub[sub["score"].abs() >= score_threshold]
    return set(sub["unit"].astype(int).tolist())


def disentanglement_tables(
    health: pd.DataFrame | None,
    profiles: pd.DataFrame,
    scores: pd.DataFrame,
    output: Path,
    *,
    score_threshold: float = 5.0,
    q_threshold: float = 0.05,
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
            "mean_abs_contribution", "dead", "rare", "ubiquitous",
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
    units["phone_like_score"] = _max_existing(units, phone_cols)
    units["speaker_score"] = units.get("speaker_id__score", 0.0)
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

    sig_phone = _significant_units(scores, _PHONE_LIKE_FACTORS, q_threshold=q_threshold,
                                   score_threshold=score_threshold)
    sig_speaker = _significant_units(scores, ("speaker_id",), q_threshold=q_threshold,
                                     score_threshold=score_threshold)
    sig_prosody = _significant_units(scores, _PROSODY_LIKE_FACTORS, q_threshold=q_threshold,
                                     score_threshold=score_threshold)
    sig_emotion = _significant_units(scores, ("emotion",), q_threshold=q_threshold,
                                     score_threshold=score_threshold)
    sig_metadata = _significant_units(scores, _METADATA_LIKE_FACTORS, q_threshold=q_threshold,
                                      score_threshold=score_threshold)

    units["phone_selective"] = (
        units["phone_like_score"].ge(score_threshold) |
        units["unit"].isin(sig_phone)
    )
    units["speaker_selective"] = (
        units["speaker_score"].ge(score_threshold) |
        units["unit"].isin(sig_speaker)
    )
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

    units["route_violation"] = False
    # L is the linguistic route: any speaker/prosody/emotion/metadata
    # paralinguistic selectivity is leakage, not only closed-set speaker ID.
    # This matters for TIMIT-style phonetic validation bundles where sex and
    # dialect-region labels are useful nuisance factors even when each speaker
    # has few utterances.
    units.loc[(units["route"] == "L") & units["paralinguistic_selective"], "route_violation"] = True
    # P is the paralinguistic route: phone-like selectivity is content leakage.
    units.loc[(units["route"] == "P") & units["phone_selective"], "route_violation"] = True
    units.loc[(units["route"] == "U") & (units["linguistic_selective"] | units["paralinguistic_selective"]),
              "route_violation"] = True

    def tags(row: pd.Series) -> str:
        out: list[str] = []
        if row.get("mixed_phone_speaker", False): out.append("mixed_phone_speaker")
        if row.get("mixed_linguistic_paralinguistic", False): out.append("mixed_linguistic_paralinguistic")
        if row.get("route") == "L" and row.get("speaker_selective", False): out.append("speaker_in_L")
        if row.get("route") == "L" and row.get("prosody_selective", False): out.append("prosody_in_L")
        if row.get("route") == "L" and row.get("emotion_selective", False): out.append("emotion_in_L")
        if row.get("route") == "L" and row.get("metadata_paralinguistic_selective", False): out.append("metadata_in_L")
        if row.get("route") == "L" and row.get("paralinguistic_selective", False): out.append("paralinguistic_in_L")
        if row.get("route") == "P" and row.get("phone_selective", False): out.append("phone_in_P")
        if row.get("route") == "U" and row.get("linguistic_selective", False): out.append("linguistic_in_U")
        if row.get("route") == "U" and row.get("paralinguistic_selective", False): out.append("paralinguistic_in_U")
        # Preserve order while removing duplicates such as speaker_in_L and the
        # broader paralinguistic_in_L both firing.
        return ";".join(dict.fromkeys(out))

    units["issue_tags"] = units.apply(tags, axis=1)
    units["leakage_score"] = 0.0
    units.loc[units.route == "L", "leakage_score"] = units.loc[units.route == "L", "paralinguistic_score"]
    units.loc[units.route == "P", "leakage_score"] = units.loc[units.route == "P", "phone_like_score"]
    units.loc[units.route == "U", "leakage_score"] = units.loc[
        units.route == "U", ["linguistic_score", "paralinguistic_score"]
    ].max(axis=1)

    leaky = units[(units["issue_tags"] != "") | units["route_violation"]].copy()
    leaky = leaky.sort_values(["leakage_score", "max_factor_score"], ascending=False)

    route_rows: list[dict[str, Any]] = []
    for route, group in units.groupby("route", dropna=False):
        active = group[~group.get("dead", False).fillna(False)] if "dead" in group else group
        denom = max(len(group), 1)
        active_denom = max(len(active), 1)
        route_rows.append({
            "route": route,
            "units": int(len(group)),
            "active_units": int(len(active)),
            "dead_fraction": float(group.get("dead", False).fillna(False).mean()) if "dead" in group else 0.0,
            "phone_selective_fraction": float(group["phone_selective"].mean()),
            "speaker_selective_fraction": float(group["speaker_selective"].mean()),
            "prosody_selective_fraction": float(group["prosody_selective"].mean()),
            "metadata_paralinguistic_selective_fraction": float(group["metadata_paralinguistic_selective"].mean()),
            "mixed_phone_speaker_fraction": float(group["mixed_phone_speaker"].mean()),
            "mixed_any_fraction": float(group["mixed_linguistic_paralinguistic"].mean()),
            "route_violation_fraction": float(group["route_violation"].mean()),
            "active_phone_selective_fraction": float(active["phone_selective"].mean()) if active_denom else 0.0,
            "active_speaker_selective_fraction": float(active["speaker_selective"].mean()) if active_denom else 0.0,
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

    thesis = pd.DataFrame([{
        "score_threshold": score_threshold,
        "q_threshold": q_threshold,
        "L_phone_selective_fraction": route_value("L", "phone_selective_fraction"),
        "L_speaker_leak_fraction": route_value("L", "speaker_selective_fraction"),
        "L_paralinguistic_leak_fraction": route_value("L", "route_violation_fraction"),
        "P_speaker_selective_fraction": route_value("P", "speaker_selective_fraction"),
        "P_phone_leak_fraction": route_value("P", "phone_selective_fraction"),
        "U_info_fraction": route_value("U", "route_violation_fraction"),
        "mixed_phone_speaker_fraction_all": float(units["mixed_phone_speaker"].mean()),
        "mixed_any_fraction_all": float(units["mixed_linguistic_paralinguistic"].mean()),
        "route_violation_fraction_all": float(units["route_violation"].mean()),
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
    return float(-(p * np.log(p + 1e-12)).sum())


def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    ha, hb = _entropy(a), _entropy(b)
    mi = 0.0
    for av in np.unique(a):
        for bv in np.unique(b):
            pab = np.mean((a == av) & (b == bv))
            if pab:
                mi += pab * math.log(pab / (np.mean(a == av) * np.mean(b == bv)))
    return float(mi / max(math.sqrt(ha * hb), 1e-12))


def clustering_analysis(
    cache: FeatureCache, profiles: pd.DataFrame, output: Path, seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    numeric = [c for c in profiles.columns if c.endswith("_score") or c == "delta_L_minus_P"]
    x = profiles[numeric].to_numpy(dtype=np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    active = np.linalg.norm(x, axis=1) > 0
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


def _cos_rows(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xa, xb = x[a], x[b]
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
        zf = cache.dense(sample_frames)
        phone = cache.phones[sample_frames].astype(str)
        ui = frame_to_utterance(cache, sample_frames)
        speakers = metadata.iloc[ui]["speaker_id"].astype(str).to_numpy() if "speaker_id" in metadata else np.full(len(ui), "")
        for rid, name in ((0, "L"), (1, "P")):
            mask = cache.route == rid
            same_a, same_b, contrast_a, contrast_b = [], [], [], []
            for i in range(len(zf)):
                same = np.flatnonzero((phone == phone[i]) & (speakers != speakers[i]))
                contrast = np.flatnonzero((phone != phone[i]) & (speakers == speakers[i]))
                if len(same): same_a.append(i); same_b.append(int(rng.choice(same)))
                if len(contrast): contrast_a.append(i); contrast_b.append(int(rng.choice(contrast)))
            a = _cos_rows(zf[:, mask], np.asarray(same_a), np.asarray(same_b)) if same_a else np.zeros(0)
            b = _cos_rows(zf[:, mask], np.asarray(contrast_a), np.asarray(contrast_b)) if contrast_a else np.zeros(0)
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
    for start in range(0, cache.K, 256):
        sims = decoder[start:start+256] @ decoder.T
        for local in range(len(sims)):
            unit = start + local
            sims[local, unit] = -np.inf
            near = np.argpartition(sims[local], -5)[-5:]
            near = near[np.argsort(sims[local, near])[::-1]]
            for rank, other in enumerate(near, 1):
                rows.append({
                    "unit": unit, "neighbor": int(other), "rank": rank,
                    "decoder_cosine": float(sims[local, other]),
                    "unit_route": ROUTE_NAMES.get(int(cache.route[unit]), str(cache.route[unit])),
                    "neighbor_route": ROUTE_NAMES.get(int(cache.route[other]), str(cache.route[other])),
                })
    table = pd.DataFrame(rows)
    cross = table.unit_route != table.neighbor_route
    # Empirical coactivation is complementary to decoder geometry. Restrict to
    # frequently active units and a deterministic frame grid to bound memory.
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
        "mean_nearest_cosine": float(table[table.rank == 1].decoder_cosine.mean()),
        "cross_route_neighbor_fraction": float(cross.mean()),
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
