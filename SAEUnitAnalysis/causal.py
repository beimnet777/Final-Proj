from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analyses import cache_metadata
from .bundle import AnalysisBundle
from .evaluators import EvaluatorSuite, evaluate_frames, evaluate_utterances
from .extraction import FeatureCache
from .types import ResolvedModel
from .utils import AnalysisError, write_json


BUDGETS = (1, 2, 5, 10, 20, 50, 100)
SWAP_MODE_ORDER = (
    "baseline",
    "identity_P",
    "P_same_speaker",
    "P_from_donor",
    "L_from_donor",
    "matched_P_subset_from_donor",
    "matched_nonP_from_donor",
    "P_zero",
    "P_mean",
    "P_time_shuffled_from_donor",
    "random_route_P_from_donor",
)
SWAP_METRICS = (
    "phone_recipient_accuracy",
    "donor_speaker_match",
    "recipient_speaker_match",
    "donor_speaker_probability",
    "recipient_speaker_probability",
    "reconstruction_shift_mse",
)


def _decode(dense: np.ndarray, resolved: ResolvedModel) -> np.ndarray:
    weight = resolved.state["sae.dec_weight"].detach().float().cpu().numpy()
    bias = resolved.state["sae.b_pre"].detach().float().cpu().numpy()
    return dense @ weight.T + bias


def _stats(h: np.ndarray) -> np.ndarray:
    return h.mean(0)


def _test_indices(
    cache: FeatureCache, bundle: AnalysisBundle, suite: EvaluatorSuite | None = None,
) -> np.ndarray:
    metadata = cache_metadata(bundle, cache)
    name = str(bundle.spec.split_map.get("test", "test"))
    indices = np.flatnonzero(metadata["split"].astype(str).to_numpy() == name)
    if suite is not None and len(suite.evaluation_utterance_ids):
        allowed = set(suite.evaluation_utterance_ids.astype(str).tolist())
        indices = np.asarray(
            [i for i in indices if str(cache.utterance_ids[int(i)]) in allowed], dtype=int,
        )
    return indices


def _evaluate_intervention(
    cache: FeatureCache, bundle: AnalysisBundle, resolved: ResolvedModel,
    suite: EvaluatorSuite, ablate: np.ndarray | None = None,
    retain: np.ndarray | None = None, max_utterances: int = 0,
) -> dict[str, float]:
    metadata = cache_metadata(bundle, cache)
    indices = _test_indices(cache, bundle, suite)
    if max_utterances > 0: indices = indices[:max_utterances]
    frame_metrics = []
    stats, speakers, emotions = [], [], []
    for ui in indices:
        sl = cache.utterance_slice(int(ui))
        z = cache.dense(sl)
        if ablate is not None: z[:, ablate] = 0
        if retain is not None:
            keep = np.zeros(cache.K, dtype=bool); keep[retain] = True; z[:, ~keep] = 0
        h = _decode(z, resolved)
        frame_metrics.append(evaluate_frames(suite, h, cache.phones[sl], cache.f0[sl], cache.energy[sl], cache.voicing[sl]))
        stats.append(_stats(h))
        row = metadata.iloc[int(ui)]
        speakers.append(str(row.get("speaker_id", "")))
        emotions.append(str(row.get("emotion", "")))
    out: dict[str, float] = {}
    keys = set().union(*(m.keys() for m in frame_metrics)) if frame_metrics else set()
    for key in keys:
        vals = [m[key] for m in frame_metrics if key in m]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    if stats:
        out.update(evaluate_utterances(suite, np.asarray(stats), np.asarray(speakers), np.asarray(emotions)))
    return out


def causal_analysis(
    cache: FeatureCache, bundle: AnalysisBundle, resolved: ResolvedModel,
    suite: EvaluatorSuite, profiles: pd.DataFrame, output: Path,
    seed: int = 42, quick: bool = False,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    max_u = 32 if quick else int(bundle.spec.raw.get("causal_max_utterances", 0))
    baseline = _evaluate_intervention(cache, bundle, resolved, suite, max_utterances=max_u)

    def decorate(row: dict[str, Any], metrics: dict[str, float], primary: str) -> None:
        if primary in baseline and primary in metrics:
            row["target_delta"] = metrics[primary] - baseline[primary]
        collateral = []
        for key in metrics:
            if key not in baseline or key == primary or not np.isfinite(metrics[key]) or not np.isfinite(baseline[key]):
                continue
            delta = abs(metrics[key] - baseline[key])
            if key.endswith("rmse"):
                delta /= max(abs(baseline[key]), 1e-8)
            collateral.append(delta)
        row["collateral_damage"] = float(np.mean(collateral)) if collateral else 0.0
        if "target_delta" in row:
            row["causal_specificity"] = float(abs(row["target_delta"]) / (row["collateral_damage"] + 1e-8))
    families = [
        ("linguistic", 0, "PhoneScore" if "PhoneScore" in profiles else "linguistic_score", "phone_accuracy"),
        ("paralinguistic", 1, "SpeakerScore" if "SpeakerScore" in profiles else "paralinguistic_score", "speaker_accuracy"),
    ]
    rng = np.random.default_rng(seed)
    n_controls = 10 if quick else int(bundle.spec.raw.get("random_controls", 100))
    for family, route, score_col, primary in families:
        if primary not in baseline:
            continue
        candidates = profiles[profiles.route_id == route].sort_values(score_col, ascending=False).unit.to_numpy(dtype=int)
        if not len(candidates): continue
        for budget in BUDGETS:
            selected = candidates[:min(budget, len(candidates))]
            for mode in ("ablate", "retain"):
                metrics = _evaluate_intervention(
                    cache, bundle, resolved, suite,
                    ablate=selected if mode == "ablate" else None,
                    retain=selected if mode == "retain" else None,
                    max_utterances=max_u,
                )
                row = {"family": family, "mode": mode, "budget": len(selected), **metrics}
                decorate(row, metrics, primary)
                rows.append(row)
            # Matched random controls. Health matching is approximated by the
            # same route; activity-matched sampling is added by sorting candidates
            # before deterministic local windows.
            controls = []
            if len(candidates) >= len(selected):
                pool = candidates
                if "frame_frequency" in profiles and len(selected):
                    target_frequency = float(profiles.set_index("unit").loc[selected].frame_frequency.mean())
                    candidate_frequency = profiles.set_index("unit").loc[candidates].frame_frequency.to_numpy()
                    order = np.argsort(np.abs(candidate_frequency - target_frequency))
                    pool = candidates[order[:max(len(selected), min(len(candidates), 10 * len(selected)))]]
                for _ in range(n_controls):
                    random_units = rng.choice(pool, size=len(selected), replace=False)
                    metrics = _evaluate_intervention(cache, bundle, resolved, suite,
                                                     ablate=random_units, max_utterances=max_u)
                    if primary in baseline and primary in metrics:
                        controls.append(metrics[primary] - baseline[primary])
            rows.append({
                "family": family, "mode": "random_ablate", "budget": len(selected),
                "target_delta": float(np.mean(controls)) if controls else float("nan"),
                "target_delta_std": float(np.std(controls)) if controls else float("nan"),
                "controls": len(controls),
            })

        # Individual necessity tests expose whether a group-level result is
        # concentrated in a few units or distributed across the route.
        single_n = min(5 if quick else 25, len(candidates))
        for unit in candidates[:single_n]:
            metrics = _evaluate_intervention(cache, bundle, resolved, suite,
                                             ablate=np.asarray([unit]), max_utterances=max_u)
            row = {"family": family, "mode": "single_ablate", "budget": 1,
                   "unit": int(unit), **metrics}
            decorate(row, metrics, primary)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "tables" / "interventions.csv", index=False)
    try: table.to_parquet(output / "tables" / "interventions.parquet", index=False)
    except Exception: pass
    summary = {
        "baseline": baseline,
        "budgets": list(BUDGETS),
        "random_controls": n_controls,
        "evaluation_utterances": int(len(_test_indices(cache, bundle, suite))),
        "available_primary_metrics": [k for k in ("phone_accuracy", "speaker_accuracy") if k in baseline],
    }
    write_json(output / "causal.json", summary)
    return table, summary


def _resample(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) == length: return x
    if len(x) == 1: return np.repeat(x, length, axis=0)
    old = np.linspace(0, 1, len(x)); new = np.linspace(0, 1, length)
    # np.interp returns float64 regardless of input. Keep every intervention in
    # the same float32 domain as the baseline reconstruction; otherwise highly
    # saturated measurement probes can produce dtype-dependent probabilities.
    return np.stack(
        [np.interp(new, old, x[:, j]) for j in range(x.shape[1])], 1,
    ).astype(x.dtype, copy=False)


def _length_compatible_candidates(
    anchor: int,
    candidates: np.ndarray,
    lengths: np.ndarray,
    *,
    tolerance: float = .10,
) -> np.ndarray:
    """Return candidates within the declared duration tolerance."""
    candidates = np.asarray(candidates, dtype=int)
    if not len(candidates):
        return candidates
    relative = np.abs(lengths[candidates].astype(float) - float(lengths[anchor])) / max(
        float(lengths[anchor]), 1.0,
    )
    return candidates[relative <= float(tolerance)]


def build_swap_pairs(
    cache: FeatureCache,
    bundle: AnalysisBundle,
    suite: EvaluatorSuite,
    *,
    seed: int = 42,
    quick: bool = False,
    pair_manifest: Path | None = None,
) -> pd.DataFrame:
    """Create or validate a checkpoint-independent intervention pair registry.

    The registry stores utterance IDs rather than cache row numbers so the same
    recipient/donor design can be supplied to every compatible checkpoint.
    """
    metadata = cache_metadata(bundle, cache)
    test = _test_indices(cache, bundle, suite)
    id_to_index = {
        str(utterance_id): int(i) for i, utterance_id in enumerate(cache.utterance_ids)
    }
    required = {
        "pair", "recipient", "donor", "same_speaker_donor",
        "recipient_speaker", "donor_speaker",
    }
    if pair_manifest is not None:
        supplied = pd.read_csv(pair_manifest)
        missing = required - set(supplied.columns)
        if missing:
            raise AnalysisError(
                f"Swap pair manifest is missing columns: {sorted(missing)}"
            )
        unknown = (
            set(supplied["recipient"].astype(str))
            | set(supplied["donor"].astype(str))
            | set(supplied["same_speaker_donor"].dropna().astype(str))
        ) - set(id_to_index)
        if unknown:
            raise AnalysisError(
                f"Swap pair manifest contains {len(unknown)} utterances absent from this cache."
            )
        supplied = supplied.copy()
        allowed = set(cache.utterance_ids[test].astype(str).tolist())
        primary_ids = (
            set(supplied["recipient"].astype(str))
            | set(supplied["donor"].astype(str))
        )
        if not primary_ids <= allowed:
            raise AnalysisError(
                "Swap pair manifest includes recipient/donor utterances outside the reserved "
                "speaker-evaluator partition."
            )
        supplied["recipient_index"] = supplied["recipient"].astype(str).map(id_to_index)
        supplied["donor_index"] = supplied["donor"].astype(str).map(id_to_index)
        supplied["same_speaker_donor_index"] = (
            supplied["same_speaker_donor"].astype(str).map(id_to_index)
        )
        actual_speakers = metadata["speaker_id"].astype(str).to_numpy()
        for _, row in supplied.iterrows():
            a, b, c = (
                int(row["recipient_index"]), int(row["donor_index"]),
                int(row["same_speaker_donor_index"]),
            )
            if a == b or actual_speakers[a] == actual_speakers[b]:
                raise AnalysisError("Every primary swap pair must use different speakers.")
            if actual_speakers[c] != actual_speakers[a]:
                raise AnalysisError("same_speaker_donor does not match the recipient speaker.")
            relative = abs(int(cache.lengths[b]) - int(cache.lengths[a])) / max(
                int(cache.lengths[a]), 1,
            )
            if relative > .10 + 1e-8:
                raise AnalysisError("Swap pair manifest violates the 10% duration tolerance.")
        if "same_speaker_control_available" not in supplied:
            supplied["same_speaker_control_available"] = (
                supplied["same_speaker_donor"].astype(str)
                != supplied["recipient"].astype(str)
            )
        return supplied

    rng = np.random.default_rng(seed)
    speakers = metadata["speaker_id"].astype(str).to_numpy()
    test = np.asarray(sorted(test, key=lambda i: str(cache.utterance_ids[int(i)])), dtype=int)
    rows: list[dict[str, Any]] = []
    for a in test:
        different = test[(test != a) & (speakers[test] != speakers[a])]
        different = _length_compatible_candidates(a, different, cache.lengths)
        same = test[(test != a) & (speakers[test] == speakers[a])]
        same = _length_compatible_candidates(a, same, cache.lengths)
        if not len(different):
            continue
        # Choice is reproducible because candidates and anchors are sorted
        # before drawing. It therefore does not depend on checkpoint internals.
        b = int(different[int(rng.integers(len(different)))])
        same_available = bool(len(same))
        c = int(same[int(rng.integers(len(same)))]) if same_available else int(a)
        rows.append({
            "pair": int(len(rows)),
            "recipient": str(cache.utterance_ids[a]),
            "donor": str(cache.utterance_ids[b]),
            "same_speaker_donor": str(cache.utterance_ids[c]),
            "recipient_index": int(a),
            "donor_index": int(b),
            "same_speaker_donor_index": int(c),
            "recipient_speaker": str(speakers[a]),
            "donor_speaker": str(speakers[b]),
            "recipient_frames": int(cache.lengths[a]),
            "donor_frames": int(cache.lengths[b]),
            "same_speaker_donor_frames": int(cache.lengths[c]),
            "length_ratio": float(cache.lengths[b] / max(int(cache.lengths[a]), 1)),
            "same_speaker_length_ratio": float(
                cache.lengths[c] / max(int(cache.lengths[a]), 1)
            ),
            "same_speaker_control_available": same_available,
        })
        if quick and len(rows) >= 16:
            break
    return pd.DataFrame(rows)


def _route_mean(cache: FeatureCache, bundle: AnalysisBundle) -> np.ndarray:
    """Mean latent, including implicit sparse zeros, on the training split."""
    metadata = cache_metadata(bundle, cache)
    train_name = str(bundle.spec.split_map.get("train", "train"))
    selected = np.flatnonzero(metadata["split"].astype(str).to_numpy() == train_name)
    total = np.zeros(cache.K, dtype=np.float64)
    frames = 0
    for ui in selected:
        sl = cache.utterance_slice(int(ui))
        indices = cache.indices[sl].reshape(-1)
        values = cache.values[sl].reshape(-1).astype(np.float32)
        total += np.bincount(indices, weights=values, minlength=cache.K)
        frames += int(sl.stop - sl.start)
    return (total / max(frames, 1)).astype(np.float32)


def _matched_non_p_units(
    cache: FeatureCache,
    decoder_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Match non-P units to P units without route overlap.

    Matching uses log activity frequency and decoder-column norm. Hungarian
    assignment provides a unique control unit for every target whenever the
    complement has enough capacity.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("Matched swap controls require scipy.") from exc
    p_units = np.flatnonzero(cache.route == 1)
    other_units = np.flatnonzero(cache.route != 1)
    count = min(len(p_units), len(other_units))
    if count == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=int), {"matched_units": 0}
    frequency = np.bincount(
        cache.indices.reshape(-1), minlength=cache.K,
    ).astype(np.float64) / max(cache.n_frames, 1)
    norm = np.linalg.norm(decoder_weight.astype(np.float64), axis=0)
    features = np.stack([np.log1p(frequency), np.log1p(norm)], axis=1)
    centre = features.mean(axis=0)
    scale = features.std(axis=0)
    features = (features - centre) / np.where(scale > 1e-12, scale, 1.0)
    target = p_units
    candidates = other_units
    distance = ((features[target][:, None, :] - features[candidates][None, :, :]) ** 2).sum(axis=2)
    # Rectangular assignment selects the best unique subset from the larger
    # route.  When P owns more than half of the dictionary, a full-P versus
    # disjoint non-P comparison is mathematically impossible; the explicit
    # matched-P-subset intervention below is therefore the only valid target
    # for this capacity-matched control.
    target_rows, candidate_columns = linear_sum_assignment(distance)
    target = target[target_rows].astype(int)
    matched = candidates[candidate_columns].astype(int)
    return target, matched, {
        "matched_units": int(len(matched)),
        "full_p_units": int(len(p_units)),
        "full_non_p_units": int(len(other_units)),
        "target_p_fraction": float(len(target) / max(len(p_units), 1)),
        "full_p_capacity_match": bool(len(p_units) <= len(other_units)),
        "target_route": "P",
        "control_route": "non-P",
        "overlap": int(len(np.intersect1d(target, matched))),
        "matching_variables": ["log_frame_frequency", "log_decoder_column_norm"],
        "mean_match_distance": float(np.sqrt(distance[target_rows, candidate_columns]).mean()),
    }


def _speaker_transfer_metrics(
    suite: EvaluatorSuite,
    h_stats: np.ndarray,
    recipient_speaker: str,
    donor_speaker: str,
) -> dict[str, Any]:
    if suite.speaker is None or not len(suite.speaker_classes):
        return {}
    encoded_prediction = int(np.asarray(suite.speaker.predict(h_stats[None]))[0])
    predicted = str(suite.speaker_classes[encoded_prediction])
    out: dict[str, Any] = {
        "predicted_speaker": predicted,
        "recipient_speaker_match": float(predicted == str(recipient_speaker)),
        "donor_speaker_match": float(predicted == str(donor_speaker)),
    }
    if hasattr(suite.speaker, "predict_proba"):
        probability = np.asarray(suite.speaker.predict_proba(h_stats[None]))[0]
        encoded_classes = np.asarray(getattr(suite.speaker, "classes_", np.arange(len(probability))), dtype=int)
        column_for_class = {int(label): column for column, label in enumerate(encoded_classes.tolist())}
        for prefix, label in (("recipient", recipient_speaker), ("donor", donor_speaker)):
            matches = np.flatnonzero(suite.speaker_classes.astype(str) == str(label))
            if len(matches) and int(matches[0]) in column_for_class:
                out[f"{prefix}_speaker_probability"] = float(
                    probability[column_for_class[int(matches[0])]]
                )
    return out


def _bootstrap_mode_summary(
    table: pd.DataFrame,
    *,
    seed: int,
    repetitions: int = 1000,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    observed_modes = set(table["mode"].astype(str))
    interpolation_modes = sorted(
        [mode for mode in observed_modes if mode.startswith("P_interpolate_")],
        key=lambda mode: float(mode.rsplit("_", 1)[-1]),
    )
    mode_order = [mode for mode in SWAP_MODE_ORDER if mode in observed_modes]
    mode_order += [mode for mode in interpolation_modes if mode not in mode_order]
    for mode in mode_order:
        group = table[table["mode"] == mode]
        if group.empty:
            continue
        row: dict[str, Any] = {"mode": mode, "pairs": int(len(group))}
        if "interpolation_alpha" in group:
            alpha = pd.to_numeric(group["interpolation_alpha"], errors="coerce").dropna()
            if len(alpha):
                row["interpolation_alpha"] = float(alpha.iloc[0])
        for metric in SWAP_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            row[metric] = float(np.mean(values))
            if len(values) >= 2:
                valid = pd.to_numeric(group[metric], errors="coerce").notna()
                if (
                    {"recipient_speaker", "donor_speaker"} <= set(group.columns)
                    and group.loc[valid, "recipient_speaker"].nunique() >= 2
                    and group.loc[valid, "donor_speaker"].nunique() >= 2
                ):
                    draws = _two_way_cluster_draws(
                        values,
                        group.loc[valid, "recipient_speaker"].astype(str).to_numpy(),
                        group.loc[valid, "donor_speaker"].astype(str).to_numpy(),
                        rng=rng, repetitions=repetitions,
                    )
                    row["interval_method"] = "two_way_recipient_donor_speaker_bootstrap"
                else:
                    draws = rng.choice(
                        values, size=(int(repetitions), len(values)), replace=True,
                    ).mean(axis=1)
                    row["interval_method"] = "row_bootstrap_fallback"
                low, high = np.quantile(draws, [0.025, 0.975])
                row[f"{metric}_ci95_low"] = float(low)
                row[f"{metric}_ci95_high"] = float(high)
        rows.append(row)
    return pd.DataFrame(rows)


def _two_way_cluster_draws(
    values: np.ndarray,
    recipient_clusters: np.ndarray,
    donor_clusters: np.ndarray,
    *,
    rng: np.random.Generator,
    repetitions: int,
) -> np.ndarray:
    """Pigeonhole bootstrap for crossed recipient and donor speakers."""
    values = np.asarray(values, dtype=float)
    recipient_clusters = np.asarray(recipient_clusters).astype(str)
    donor_clusters = np.asarray(donor_clusters).astype(str)
    recipients = np.unique(recipient_clusters)
    donors = np.unique(donor_clusters)
    draws = np.empty(int(repetitions), dtype=float)
    for repetition in range(int(repetitions)):
        sampled_recipients = rng.choice(recipients, size=len(recipients), replace=True)
        sampled_donors = rng.choice(donors, size=len(donors), replace=True)
        recipient_count = {
            value: int(np.sum(sampled_recipients == value)) for value in recipients
        }
        donor_count = {value: int(np.sum(sampled_donors == value)) for value in donors}
        weights = np.asarray([
            recipient_count.get(recipient, 0) * donor_count.get(donor, 0)
            for recipient, donor in zip(recipient_clusters, donor_clusters)
        ], dtype=float)
        draws[repetition] = (
            float(np.average(values, weights=weights)) if weights.sum() else float(np.mean(values))
        )
    return draws


def paired_swap_contrasts(
    table: pd.DataFrame,
    *,
    seed: int,
    repetitions: int = 1000,
) -> pd.DataFrame:
    """Paired mode-minus-baseline effects with crossed-speaker uncertainty."""
    if table.empty or "baseline" not in set(table["mode"].astype(str)):
        return pd.DataFrame()
    identifiers = [
        column for column in (
            "pair", "recipient", "donor", "recipient_speaker", "donor_speaker",
        ) if column in table.columns
    ]
    baseline_columns = [metric for metric in SWAP_METRICS if metric in table.columns]
    baseline = table[table["mode"] == "baseline"][identifiers + baseline_columns].copy()
    baseline = baseline.rename(columns={metric: f"{metric}__baseline" for metric in baseline_columns})
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for mode, group in table[table["mode"] != "baseline"].groupby("mode", sort=False):
        current_columns = identifiers + [metric for metric in baseline_columns if metric in group.columns]
        merged = group[current_columns].merge(baseline, on=identifiers, how="inner", validate="one_to_one")
        for metric in baseline_columns:
            before = pd.to_numeric(merged[f"{metric}__baseline"], errors="coerce")
            after = pd.to_numeric(merged[metric], errors="coerce")
            valid = before.notna() & after.notna()
            if not valid.any():
                continue
            effects = (after[valid] - before[valid]).to_numpy(dtype=float)
            recipients = merged.loc[valid, "recipient_speaker"].astype(str).to_numpy()
            donors = merged.loc[valid, "donor_speaker"].astype(str).to_numpy()
            if len(np.unique(recipients)) >= 2 and len(np.unique(donors)) >= 2:
                draws = _two_way_cluster_draws(
                    effects, recipients, donors, rng=rng, repetitions=repetitions,
                )
                interval_method = "paired_two_way_recipient_donor_speaker_bootstrap"
            else:
                draws = rng.choice(
                    effects, size=(int(repetitions), len(effects)), replace=True,
                ).mean(axis=1)
                interval_method = "paired_row_bootstrap_fallback"
            low, high = np.quantile(draws, [.025, .975])
            row: dict[str, Any] = {
                "mode": str(mode),
                "metric": metric,
                "pairs": int(len(effects)),
                "recipient_speakers": int(len(np.unique(recipients))),
                "donor_speakers": int(len(np.unique(donors))),
                "baseline_mean": float(before[valid].mean()),
                "mode_mean": float(after[valid].mean()),
                "paired_effect": float(np.mean(effects)),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "positive_pair_fraction": float(np.mean(effects > 0)),
                "interval_method": interval_method,
            }
            if "interpolation_alpha" in group:
                alpha = pd.to_numeric(group["interpolation_alpha"], errors="coerce").dropna()
                if len(alpha):
                    row["interpolation_alpha"] = float(alpha.iloc[0])
            rows.append(row)
    return pd.DataFrame(rows)


def build_swap_grid(
    pairs: pd.DataFrame,
    *,
    speakers: int = 5,
    contents: int = 5,
) -> pd.DataFrame:
    """Small deterministic content-by-speaker registry for later audio demos."""
    if pairs.empty:
        return pd.DataFrame()
    recipients = (
        pairs.sort_values(["recipient_speaker", "recipient"], kind="stable")
        .drop_duplicates("recipient_speaker")
        .head(int(contents))
    )
    recipient_speakers = set(recipients["recipient_speaker"].astype(str))
    donor_pool = pairs.sort_values(["donor_speaker", "donor"], kind="stable")
    different = donor_pool[~donor_pool["donor_speaker"].astype(str).isin(recipient_speakers)]
    donors = different.drop_duplicates("donor_speaker").head(int(speakers))
    if len(donors) < int(speakers):
        fallback = donor_pool[
            ~donor_pool["donor_speaker"].astype(str).isin(
                set(donors["donor_speaker"].astype(str))
            )
        ].drop_duplicates("donor_speaker")
        donors = pd.concat([donors, fallback], ignore_index=True).head(int(speakers))
    rows: list[dict[str, Any]] = []
    for row_index, (_, recipient) in enumerate(recipients.iterrows()):
        for column_index, (_, donor) in enumerate(donors.iterrows()):
            rows.append({
                "grid_row": int(row_index),
                "grid_column": int(column_index),
                "recipient": str(recipient["recipient"]),
                "recipient_speaker": str(recipient["recipient_speaker"]),
                "donor": str(donor["donor"]),
                "donor_speaker": str(donor["donor_speaker"]),
                "same_speaker_cell": bool(
                    str(recipient["recipient_speaker"]) == str(donor["donor_speaker"])
                ),
            })
    return pd.DataFrame(rows)


def swap_analysis(
    cache: FeatureCache, bundle: AnalysisBundle, resolved: ResolvedModel,
    suite: EvaluatorSuite, output: Path, seed: int = 42, quick: bool = False,
    pair_manifest: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Swap L/P routes and quantify recipient content and speaker identity.

    Baseline and every intervention are scored against both recipient and donor
    speaker identities. Phone accuracy always uses the recipient alignment.
    """
    metadata = cache_metadata(bundle, cache)
    rng = np.random.default_rng(seed)
    pairs = build_swap_pairs(
        cache, bundle, suite, seed=seed, quick=quick, pair_manifest=pair_manifest,
    )
    pair_export = pairs.drop(
        columns=[
            "recipient_index", "donor_index", "same_speaker_donor_index",
        ], errors="ignore",
    )
    pair_export.to_csv(output / "tables" / "swap_pairs.csv", index=False)
    rows = []
    L, P = cache.route == 0, cache.route == 1
    weight = resolved.state["sae.dec_weight"].detach().float().cpu().numpy()
    route_mean = _route_mean(cache, bundle)
    matched_p_units, matched_non_p, matching_summary = _matched_non_p_units(cache, weight)
    interpolation_alphas = (0.0, .25, .5, .75, 1.0)
    for _, pair_row in pairs.iterrows():
        pair_index = int(pair_row["pair"])
        a = int(pair_row["recipient_index"])
        b = int(pair_row["donor_index"])
        c = int(pair_row["same_speaker_donor_index"])
        za = cache.dense(cache.utterance_slice(a))
        zb = cache.dense(cache.utterance_slice(b))
        zc = cache.dense(cache.utterance_slice(c))
        random_route = rng.permutation(cache.route)
        rP = random_route == 1
        baseline_h = _decode(za, resolved)

        def replace_route(
            units: np.ndarray,
            donor_z: np.ndarray | None = None,
            *,
            replacement: np.ndarray | None = None,
            alpha: float = 1.0,
        ) -> np.ndarray:
            if replacement is None:
                if donor_z is None:
                    raise ValueError("replace_route requires donor_z or replacement.")
                replacement = _resample(donor_z[:, units], len(za))
            elif replacement.ndim == 1:
                selected_count = int(np.count_nonzero(units)) if units.dtype == bool else len(units)
                replacement = np.broadcast_to(replacement[None], (len(za), selected_count))
            delta = float(alpha) * (replacement - za[:, units])
            return baseline_h + delta @ weight[:, units].T

        # Algebraically identical to decoding each dense hybrid, while reusing
        # the recipient reconstruction shared by all interventions.
        p_hybrid_h = replace_route(P, zb)
        l_hybrid_h = replace_route(L, zb)
        same_speaker_hybrid_h = replace_route(P, zc)
        random_hybrid_h = replace_route(rP, zb)
        matched_p_hybrid_h = replace_route(matched_p_units, zb)
        matched_non_p_hybrid_h = replace_route(matched_non_p, zb)
        zero_p_h = replace_route(P, replacement=np.zeros(int(P.sum()), dtype=np.float32))
        mean_p_h = replace_route(P, replacement=route_mean[P])
        donor_p = _resample(zb[:, P], len(za))
        shuffled_p = donor_p[rng.permutation(len(donor_p))]
        time_shuffled_p_h = replace_route(P, replacement=shuffled_p)
        sl_a = cache.utterance_slice(a)
        recipient = metadata.iloc[a]
        donor = metadata.iloc[b]
        recipient_speaker = str(recipient.get("speaker_id", ""))
        donor_speaker = str(donor.get("speaker_id", ""))
        interventions = (
            ("baseline", baseline_h, float("nan")),
            ("identity_P", replace_route(P, za), float("nan")),
            ("P_same_speaker", same_speaker_hybrid_h, float("nan")),
            ("P_from_donor", p_hybrid_h, float("nan")),
            ("L_from_donor", l_hybrid_h, float("nan")),
            ("matched_P_subset_from_donor", matched_p_hybrid_h, float("nan")),
            ("matched_nonP_from_donor", matched_non_p_hybrid_h, float("nan")),
            ("P_zero", zero_p_h, float("nan")),
            ("P_mean", mean_p_h, float("nan")),
            ("P_time_shuffled_from_donor", time_shuffled_p_h, float("nan")),
            ("random_route_P_from_donor", random_hybrid_h, float("nan")),
        )
        interpolation = tuple(
            (
                f"P_interpolate_{alpha:.2f}",
                replace_route(P, zb, alpha=alpha),
                float(alpha),
            )
            for alpha in interpolation_alphas
        )
        for mode, h, interpolation_alpha in interventions + interpolation:
            frame = evaluate_frames(
                suite, h, cache.phones[sl_a],
                cache.f0[sl_a], cache.energy[sl_a], cache.voicing[sl_a],
            )
            phone_accuracy = frame.pop("phone_accuracy", float("nan"))
            speaker_metrics = _speaker_transfer_metrics(
                suite, _stats(h), recipient_speaker, donor_speaker,
            )
            rows.append({
                "pair": int(pair_index),
                "recipient": cache.utterance_ids[a], "donor": cache.utterance_ids[b],
                "recipient_speaker": recipient_speaker,
                "donor_speaker": donor_speaker,
                "mode": mode,
                "interpolation_alpha": interpolation_alpha,
                "length_ratio": float(pair_row["length_ratio"]),
                "phone_recipient_accuracy": float(phone_accuracy),
                "reconstruction_shift_mse": float(np.mean((h-baseline_h)**2)),
                **speaker_metrics, **frame,
            })
    table = pd.DataFrame(rows)
    table.to_csv(output / "tables" / "swaps.csv", index=False)
    try: table.to_parquet(output / "tables" / "swaps.parquet", index=False)
    except Exception: pass
    mode_summary = _bootstrap_mode_summary(table, seed=seed + 909) if len(table) else pd.DataFrame()
    mode_summary.to_csv(output / "tables" / "swap_mode_summary.csv", index=False)
    contrasts = paired_swap_contrasts(table, seed=seed + 1909) if len(table) else pd.DataFrame()
    contrasts.to_csv(output / "tables" / "swap_pair_contrasts.csv", index=False)
    grid = build_swap_grid(pairs, speakers=5, contents=5)
    grid.to_csv(output / "tables" / "swap_content_speaker_grid.csv", index=False)
    pair_count = int(table[["recipient", "donor"]].drop_duplicates().shape[0]) if len(table) else 0
    summary = {
        "pairs": pair_count,
        "rows": int(len(table)),
        "evaluation_utterances": int(len(_test_indices(cache, bundle, suite))),
        "protocol": "registered_recipient_L_plus_donor_P_with_complementary_and_matched_controls",
        "phone_target": "recipient alignment",
        "speaker_targets": ["recipient identity", "donor identity"],
        "bootstrap_repetitions": 1000,
        "primary_interval": "paired two-way recipient/donor-speaker cluster bootstrap",
        "pair_manifest": str(output / "tables" / "swap_pairs.csv"),
        "same_speaker_controls_available": int(
            pairs.get("same_speaker_control_available", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "matched_non_p_control": matching_summary,
        "interpolation_alphas": list(interpolation_alphas),
        "mode_means": mode_summary.to_dict(orient="records"),
        "contrasts": contrasts.to_dict(orient="records"),
    }
    if len(mode_summary):
        indexed = mode_summary.set_index("mode")
        if "baseline" in indexed.index and "P_from_donor" in indexed.index:
            for metric in (
                "phone_recipient_accuracy", "donor_speaker_match",
                "recipient_speaker_match", "donor_speaker_probability",
                "recipient_speaker_probability",
            ):
                if metric in indexed.columns:
                    summary.setdefault("legacy_mean_differences", {})[
                        f"P_from_donor_minus_baseline__{metric}"
                    ] = float(
                        indexed.loc["P_from_donor", metric] - indexed.loc["baseline", metric]
                    )
    write_json(output / "swap.json", summary)
    return table, mode_summary, contrasts, grid, summary
