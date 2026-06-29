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
    return np.concatenate([h.mean(0), h.std(0)])


def _test_indices(cache: FeatureCache, bundle: AnalysisBundle) -> np.ndarray:
    metadata = cache_metadata(bundle, cache)
    name = str(bundle.spec.split_map.get("test", "test"))
    return np.flatnonzero(metadata["split"].astype(str).to_numpy() == name)


def _evaluate_intervention(
    cache: FeatureCache, bundle: AnalysisBundle, resolved: ResolvedModel,
    suite: EvaluatorSuite, ablate: np.ndarray | None = None,
    retain: np.ndarray | None = None, max_utterances: int = 0,
) -> dict[str, float]:
    metadata = cache_metadata(bundle, cache)
    indices = _test_indices(cache, bundle)
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
        collateral = [abs(metrics[k] - baseline[k]) for k in metrics
                      if k in baseline and k != primary and np.isfinite(metrics[k]) and np.isfinite(baseline[k])]
        row["collateral_damage"] = float(np.mean(collateral)) if collateral else 0.0
        if "target_delta" in row:
            row["causal_specificity"] = float(abs(row["target_delta"]) / (row["collateral_damage"] + 1e-8))
    families = [
        ("linguistic", 0, "linguistic_score", "phone_accuracy"),
        ("paralinguistic", 1, "paralinguistic_score", "speaker_accuracy"),
    ]
    rng = np.random.default_rng(seed)
    n_controls = 10 if quick else int(bundle.spec.raw.get("random_controls", 100))
    for family, route, score_col, primary in families:
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
    summary = {"baseline": baseline, "budgets": list(BUDGETS), "random_controls": n_controls}
    write_json(output / "causal.json", summary)
    return table, summary


def _resample(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) == length: return x
    if len(x) == 1: return np.repeat(x, length, axis=0)
    old = np.linspace(0, 1, len(x)); new = np.linspace(0, 1, length)
    return np.stack([np.interp(new, old, x[:, j]) for j in range(x.shape[1])], 1)


def swap_analysis(
    cache: FeatureCache, bundle: AnalysisBundle, resolved: ResolvedModel,
    suite: EvaluatorSuite, output: Path, seed: int = 42, quick: bool = False,
) -> tuple[pd.DataFrame, dict]:
    metadata = cache_metadata(bundle, cache)
    test = _test_indices(cache, bundle)
    rng = np.random.default_rng(seed)
    pairs = []
    # Speaker-transfer pairs control emotion when possible.
    for a in test:
        row_a = metadata.iloc[int(a)]
        candidates = [int(b) for b in test if b != a
                      and str(metadata.iloc[int(b)].get("speaker_id", "")) != str(row_a.get("speaker_id", ""))
                      and abs(int(cache.lengths[b])-int(cache.lengths[a])) / max(int(cache.lengths[a]), 1) <= .10]
        if "emotion" in metadata:
            same_emotion = [b for b in candidates if str(metadata.iloc[b].get("emotion", "")) == str(row_a.get("emotion", ""))]
            candidates = same_emotion or candidates
        if candidates:
            pairs.append((int(a), int(rng.choice(candidates)), "speaker"))
        if quick and len(pairs) >= 16: break
    # Emotion-transfer pairs hold speaker fixed and change the emotion label.
    if "emotion" in metadata and "speaker_id" in metadata:
        emotion_pairs = []
        for a in test:
            row_a = metadata.iloc[int(a)]
            candidates = [int(b) for b in test if b != a
                          and str(metadata.iloc[int(b)].get("speaker_id", "")) == str(row_a.get("speaker_id", ""))
                          and str(metadata.iloc[int(b)].get("emotion", "")) != str(row_a.get("emotion", ""))
                          and abs(int(cache.lengths[b])-int(cache.lengths[a])) / max(int(cache.lengths[a]), 1) <= .10]
            if candidates: emotion_pairs.append((int(a), int(rng.choice(candidates)), "emotion"))
            if quick and len(emotion_pairs) >= 16: break
        pairs.extend(emotion_pairs)
    rows = []
    L, P, U = cache.route == 0, cache.route == 1, cache.route == 2
    for a, b, transfer_type in pairs:
        za = cache.dense(cache.utterance_slice(a)); zb = cache.dense(cache.utterance_slice(b))
        hybrid = np.zeros_like(za)
        hybrid[:, L] = za[:, L]
        hybrid[:, P] = _resample(zb[:, P], len(za))
        hybrid[:, U] = za[:, U]
        random_route = rng.permutation(cache.route)
        rL, rP, rU = random_route == 0, random_route == 1, random_route == 2
        random_hybrid = np.zeros_like(za)
        random_hybrid[:, rL] = za[:, rL]
        random_hybrid[:, rP] = _resample(zb[:, rP], len(za))
        random_hybrid[:, rU] = za[:, rU]
        baseline_h = _decode(za, resolved)
        sl_a, sl_b = cache.utterance_slice(a), cache.utterance_slice(b)
        donor_f0 = _resample(cache.f0[sl_b][:, None], len(za))[:, 0]
        donor_energy = _resample(cache.energy[sl_b][:, None], len(za))[:, 0]
        donor_voicing = _resample(cache.voicing[sl_b][:, None], len(za))[:, 0]
        donor = metadata.iloc[b]
        for mode, z_now in (("baseline", za), ("learned_route", hybrid), ("random_route", random_hybrid)):
            h = baseline_h if mode == "baseline" else _decode(z_now, resolved)
            frame = evaluate_frames(suite, h, cache.phones[sl_a], donor_f0, donor_energy, donor_voicing)
            utter = evaluate_utterances(suite, _stats(h)[None],
                                        np.asarray([str(donor.get("speaker_id", ""))]),
                                        np.asarray([str(donor.get("emotion", ""))]))
            rows.append({
                "recipient": cache.utterance_ids[a], "donor": cache.utterance_ids[b],
                "transfer": transfer_type, "mode": mode,
                "length_ratio": float(cache.lengths[b] / max(cache.lengths[a], 1)),
                "reconstruction_shift_mse": float(np.mean((h-baseline_h)**2)), **frame, **utter,
            })
    table = pd.DataFrame(rows)
    table.to_csv(output / "tables" / "swaps.csv", index=False)
    try: table.to_parquet(output / "tables" / "swaps.parquet", index=False)
    except Exception: pass
    summary = {"pairs": len(table)}
    for col in table.select_dtypes(include=[np.number]).columns:
        summary[f"mean_{col}"] = float(table[col].mean())
    write_json(output / "swap.json", summary)
    return table, summary
