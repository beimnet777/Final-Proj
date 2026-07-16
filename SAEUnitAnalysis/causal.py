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
from .utils import write_json


BUDGETS = (1, 2, 5, 10, 20, 50, 100)


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
    return np.stack([np.interp(new, old, x[:, j]) for j in range(x.shape[1])], 1)


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
    metrics = [
        "phone_recipient_accuracy", "donor_speaker_match", "recipient_speaker_match",
        "donor_speaker_probability", "recipient_speaker_probability",
        "reconstruction_shift_mse",
    ]
    rows = []
    mode_order = ["baseline", "P_from_donor", "L_from_donor", "random_route_P_from_donor"]
    rng = np.random.default_rng(seed)
    for mode in mode_order:
        group = table[table["mode"] == mode]
        if group.empty:
            continue
        row: dict[str, Any] = {"mode": mode, "pairs": int(len(group))}
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            row[metric] = float(np.mean(values))
            if len(values) >= 2:
                draws = rng.choice(values, size=(int(repetitions), len(values)), replace=True).mean(axis=1)
                low, high = np.quantile(draws, [0.025, 0.975])
                row[f"{metric}_ci95_low"] = float(low)
                row[f"{metric}_ci95_high"] = float(high)
        rows.append(row)
    return pd.DataFrame(rows)


def swap_analysis(
    cache: FeatureCache, bundle: AnalysisBundle, resolved: ResolvedModel,
    suite: EvaluatorSuite, output: Path, seed: int = 42, quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Swap L/P routes and quantify recipient content and speaker identity.

    Baseline and every intervention are scored against both recipient and donor
    speaker identities. Phone accuracy always uses the recipient alignment.
    """
    metadata = cache_metadata(bundle, cache)
    test = _test_indices(cache, bundle, suite)
    rng = np.random.default_rng(seed)
    pairs = []
    # Speaker-transfer pairs use different speakers and similar frame lengths.
    for a in test:
        row_a = metadata.iloc[int(a)]
        candidates = [int(b) for b in test if b != a
                      and str(metadata.iloc[int(b)].get("speaker_id", "")) != str(row_a.get("speaker_id", ""))
                      and abs(int(cache.lengths[b])-int(cache.lengths[a])) / max(int(cache.lengths[a]), 1) <= .10]
        if candidates:
            pairs.append((int(a), int(rng.choice(candidates))))
        if quick and len(pairs) >= 16: break
    rows = []
    L, P = cache.route == 0, cache.route == 1
    weight = resolved.state["sae.dec_weight"].detach().float().cpu().numpy()
    for pair_index, (a, b) in enumerate(pairs):
        za = cache.dense(cache.utterance_slice(a)); zb = cache.dense(cache.utterance_slice(b))
        random_route = rng.permutation(cache.route)
        rP = random_route == 1
        baseline_h = _decode(za, resolved)

        def replace_route(units: np.ndarray) -> np.ndarray:
            donor = _resample(zb[:, units], len(za))
            delta = donor - za[:, units]
            return baseline_h + delta @ weight[:, units].T

        # Algebraically identical to decoding each dense hybrid, while reusing
        # the recipient reconstruction shared by all interventions.
        p_hybrid_h = replace_route(P)
        l_hybrid_h = replace_route(L)
        random_hybrid_h = replace_route(rP)
        sl_a = cache.utterance_slice(a)
        recipient = metadata.iloc[a]
        donor = metadata.iloc[b]
        recipient_speaker = str(recipient.get("speaker_id", ""))
        donor_speaker = str(donor.get("speaker_id", ""))
        interventions = (
            ("baseline", baseline_h),
            ("P_from_donor", p_hybrid_h),
            ("L_from_donor", l_hybrid_h),
            ("random_route_P_from_donor", random_hybrid_h),
        )
        for mode, h in interventions:
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
                "length_ratio": float(cache.lengths[b] / max(cache.lengths[a], 1)),
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
    pair_count = int(table[["recipient", "donor"]].drop_duplicates().shape[0]) if len(table) else 0
    summary = {
        "pairs": pair_count,
        "rows": int(len(table)),
        "evaluation_utterances": int(len(test)),
        "protocol": "recipient_L_plus_donor_P_with_L_control_and_overlapping_shuffled_mask_diagnostic",
        "phone_target": "recipient alignment",
        "speaker_targets": ["recipient identity", "donor identity"],
        "bootstrap_repetitions": 1000,
        "mode_means": mode_summary.to_dict(orient="records"),
        "contrasts": {},
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
                    summary["contrasts"][f"P_from_donor_minus_baseline__{metric}"] = float(
                        indexed.loc["P_from_donor", metric] - indexed.loc["baseline", metric]
                    )
    write_json(output / "swap.json", summary)
    return table, mode_summary, summary
