from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analyses import cache_metadata, _split_values
from .bundle import AnalysisBundle
from .extraction import FeatureCache
from .utils import AnalysisError, write_json


FACTOR_NAMES = ("phone", "speaker_id")
VIEW_NAMES = ("full", "L", "P")


def _entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def _encode_labels(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, encoded = np.unique(np.asarray(values).astype(str), return_inverse=True)
    return encoded.astype(np.int32), labels


def _phone_segments(
    cache: FeatureCache,
    bundle: AnalysisBundle,
    evaluation_split: str,
) -> pd.DataFrame:
    metadata = cache_metadata(bundle, cache)
    if "speaker_id" not in metadata:
        raise AnalysisError("Speech factor metrics require speaker_id metadata.")
    split_values = _split_values(bundle, evaluation_split)
    split_values = set(metadata["split"].astype(str)) if split_values is None else split_values
    rows: list[dict[str, Any]] = []
    for utterance_index, meta in metadata.iterrows():
        if str(meta["split"]) not in split_values:
            continue
        offset = int(cache.offsets[utterance_index])
        length = int(cache.lengths[utterance_index])
        if length <= 0:
            continue
        phones = np.asarray(cache.phones[offset:offset + length]).astype("U32")
        starts = np.flatnonzero(np.r_[True, phones[1:] != phones[:-1]])
        ends = np.r_[starts[1:], len(phones)]
        for local_start, local_end in zip(starts, ends):
            phone = str(phones[local_start]).upper()
            if phone in {"<UNALIGNED>", "", "NAN", "NONE"}:
                continue
            rows.append({
                "segment_id": len(rows),
                "utterance_index": int(utterance_index),
                "utterance_id": str(cache.utterance_ids[utterance_index]),
                "speaker_id": str(meta["speaker_id"]),
                "phone": phone,
                "start_frame": offset + int(local_start),
                "end_frame": offset + int(local_end),
                "frames": int(local_end - local_start),
                "split": str(meta["split"]),
            })
    return pd.DataFrame(rows)


def _joint_balanced_sample(
    segments: pd.DataFrame,
    max_segments: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cap observed speaker-phone cells, retaining rare cells before common ones."""
    if segments.empty:
        return segments.copy(), {"method": "none", "available": 0, "sampled": 0}
    frame = segments.copy()
    rng = np.random.default_rng(seed)
    frame["_random"] = rng.random(len(frame))
    groups = list(frame.groupby(["speaker_id", "phone"], sort=True, observed=True))
    if max_segments <= 0 or len(frame) <= max_segments:
        sampled = frame
        cap = int(max(len(group) for _, group in groups))
    else:
        cap = max(1, int(math.ceil(max_segments / max(len(groups), 1))))
        sampled = pd.concat(
            [group.nsmallest(cap, "_random") for _, group in groups],
            ignore_index=False,
        )
        if len(sampled) > max_segments:
            # Round-robin cell rank keeps coverage when the final global trim is needed.
            sampled = sampled.sort_values(["_random"]).copy()
            sampled["_cell_rank"] = sampled.groupby(
                ["speaker_id", "phone"], observed=True
            ).cumcount()
            sampled = sampled.sort_values(["_cell_rank", "_random"]).head(max_segments)
        sampled = sampled.drop(columns=["_cell_rank"], errors="ignore")
    sampled = sampled.sort_values(["utterance_id", "start_frame"]).reset_index(drop=True)
    cell_counts = sampled.groupby(["speaker_id", "phone"], observed=True).size()
    summary = {
        "method": "observed_speaker_phone_cell_cap",
        "available": int(len(frame)),
        "sampled": int(len(sampled)),
        "observed_cells_available": int(len(groups)),
        "observed_cells_sampled": int(len(cell_counts)),
        "cell_cap": int(cap),
        "sampled_cell_min": int(cell_counts.min()) if len(cell_counts) else 0,
        "sampled_cell_median": float(cell_counts.median()) if len(cell_counts) else 0.0,
        "sampled_cell_max": int(cell_counts.max()) if len(cell_counts) else 0,
    }
    return sampled.drop(columns=["_random"], errors="ignore"), summary


def _segment_matrix(
    cache: FeatureCache,
    segments: pd.DataFrame,
    segment_topk: int,
):
    """Mean-pool frames in each phone run and retain the strongest mean activations."""
    from scipy import sparse

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    topk = max(1, min(int(segment_topk), cache.K))
    for row, segment in segments.iterrows():
        start, end = int(segment.start_frame), int(segment.end_frame)
        indices = cache.indices[start:end].reshape(-1).astype(np.int32)
        values = cache.values[start:end].reshape(-1).astype(np.float32)
        pooled = np.bincount(indices, weights=values, minlength=cache.K).astype(np.float32)
        pooled /= max(end - start, 1)
        active = np.flatnonzero(pooled > 0)
        if len(active) > topk:
            chosen = np.argpartition(pooled[active], -topk)[-topk:]
            active = active[chosen]
        active = np.sort(active)
        row_parts.append(np.full(len(active), int(row), dtype=np.int32))
        col_parts.append(active.astype(np.int32, copy=False))
        value_parts.append(pooled[active])
    rows = np.concatenate(row_parts) if row_parts else np.asarray([], dtype=np.int32)
    cols = np.concatenate(col_parts) if col_parts else np.asarray([], dtype=np.int32)
    values = np.concatenate(value_parts) if value_parts else np.asarray([], dtype=np.float32)
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(segments), cache.K), dtype=np.float32,
    )


def _discretize_sparse(x, positive_bins: int = 4):
    """Use an exact inactive bin plus equal-mass bins among positive activations."""
    from scipy import sparse

    csc = x.tocsc(copy=True)
    binned = np.empty(len(csc.data), dtype=np.uint8)
    effective_bins = np.ones(csc.shape[1], dtype=np.uint8)
    for column in range(csc.shape[1]):
        start, end = int(csc.indptr[column]), int(csc.indptr[column + 1])
        values = csc.data[start:end]
        if len(values) == 0:
            continue
        if positive_bins <= 1 or len(values) < 2:
            binned[start:end] = 1
            effective_bins[column] = 2
            continue
        quantiles = np.linspace(0, 1, positive_bins + 1)[1:-1]
        edges = np.unique(np.quantile(values, quantiles))
        binned[start:end] = 1 + np.searchsorted(edges, values, side="right")
        effective_bins[column] = int(2 + len(edges))
    return sparse.csc_matrix(
        (binned, csc.indices.copy(), csc.indptr.copy()), shape=csc.shape, dtype=np.uint8,
    ), effective_bins


def _mutual_information_scores(
    binned,
    labels: np.ndarray,
    columns: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Categorical MI for sparse zero-aware binned columns."""
    matrix = binned.tocsc(copy=False)
    y = np.asarray(labels, dtype=np.int32)
    columns = np.arange(matrix.shape[1], dtype=np.int32) if columns is None else np.asarray(columns, dtype=np.int32)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    classes = int(y.max()) + 1 if len(y) else 0
    class_total = np.bincount(y, weights=w, minlength=classes).astype(np.float64)
    total = float(class_total.sum())
    result = np.zeros(len(columns), dtype=np.float64)
    if total <= 0 or classes < 2:
        return result
    for result_index, column in enumerate(columns):
        start, end = int(matrix.indptr[column]), int(matrix.indptr[column + 1])
        row_ids = matrix.indices[start:end]
        bins = matrix.data[start:end].astype(np.int32)
        max_bin = max(int(bins.max()) if len(bins) else 0, 1)
        counts = np.zeros((classes, max_bin + 1), dtype=np.float64)
        if len(row_ids):
            np.add.at(counts, (y[row_ids], bins), w[row_ids])
            counts[:, 0] = class_total - np.bincount(
                y[row_ids], weights=w[row_ids], minlength=classes,
            )
        else:
            counts[:, 0] = class_total
        bin_total = counts.sum(axis=0)
        expected = np.outer(class_total, bin_total) / total
        valid = counts > 0
        result[result_index] = float(
            (counts[valid] / total * np.log(counts[valid] / expected[valid])).sum()
        )
    return result


def _gap(values: np.ndarray) -> tuple[float, int, int]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), -1, -1
    order = np.argsort(values)[::-1]
    first = int(order[0])
    second = int(order[1]) if len(order) > 1 else first
    return float(values[first] - (values[second] if len(order) > 1 else 0.0)), first, second


def _view_units(cache: FeatureCache) -> dict[str, np.ndarray]:
    return {
        "full": np.arange(cache.K, dtype=np.int32),
        "L": np.flatnonzero(cache.route == 0).astype(np.int32),
        "P": np.flatnonzero(cache.route == 1).astype(np.int32),
    }


def _cluster_weights(clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    encoded, names = _encode_labels(clusters)
    draws = rng.integers(0, len(names), size=len(names))
    cluster_counts = np.bincount(draws, minlength=len(names)).astype(np.float64)
    return cluster_counts[encoded]


def _mig_rows(
    binned,
    labels: dict[str, np.ndarray],
    raw_labels: dict[str, np.ndarray],
    segments: pd.DataFrame,
    views: dict[str, np.ndarray],
    seed: int,
    bootstrap_repetitions: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    normalized: dict[str, np.ndarray] = {}
    candidates: dict[tuple[str, str], np.ndarray] = {}
    for factor, y in labels.items():
        entropy = _entropy(np.bincount(y) / max(len(y), 1))
        mi = _mutual_information_scores(binned, y)
        normalized[factor] = mi / entropy if entropy > 0 else np.zeros_like(mi)
        for view, units in views.items():
            scores = normalized[factor][units]
            gap, first_local, second_local = _gap(scores)
            first = int(units[first_local]) if first_local >= 0 else -1
            second = int(units[second_local]) if second_local >= 0 else -1
            top_n = min(32, len(units))
            candidates[(factor, view)] = units[np.argsort(scores)[::-1][:top_n]]
            rows.append({
                "metric": "MIG", "component": "factor_gap", "target": factor,
                "view": view, "value": gap, "ci95_low": np.nan, "ci95_high": np.nan,
                "control": "observed", "top_unit": first, "second_unit": second,
                "top_score": float(scores[first_local]) if first_local >= 0 else np.nan,
                "second_score": float(scores[second_local]) if second_local >= 0 else np.nan,
                "interval_method": "target_cluster_bootstrap",
            })

    draws: dict[tuple[str, str], list[float]] = {key: [] for key in candidates}
    if bootstrap_repetitions > 0:
        rng = np.random.default_rng(seed + 1103)
        candidate_union = np.unique(np.concatenate(list(candidates.values())))
        union_lookup = {int(unit): index for index, unit in enumerate(candidate_union)}
        for _ in range(bootstrap_repetitions):
            for factor, y in labels.items():
                clusters = (
                    segments["utterance_id"].to_numpy()
                    if factor == "phone" else segments["speaker_id"].to_numpy()
                )
                weights = _cluster_weights(clusters, rng)
                mi = _mutual_information_scores(binned, y, candidate_union, weights)
                entropy = _entropy(
                    np.bincount(y, weights=weights, minlength=int(y.max()) + 1)
                    / max(float(weights.sum()), 1.0)
                )
                scores = mi / entropy if entropy > 0 else np.zeros_like(mi)
                for view in views:
                    units = candidates[(factor, view)]
                    local = np.asarray([union_lookup[int(unit)] for unit in units], dtype=int)
                    draws[(factor, view)].append(_gap(scores[local])[0])
        row_lookup = {
            (str(row["target"]), str(row["view"])): row
            for row in rows if row["control"] == "observed"
        }
        for key, values in draws.items():
            if values:
                row_lookup[key]["ci95_low"] = float(np.quantile(values, .025))
                row_lookup[key]["ci95_high"] = float(np.quantile(values, .975))

    observed_lookup = {
        (str(row["target"]), str(row["view"])): row
        for row in rows if row["control"] == "observed"
    }
    for factor, positive, negative in (
        ("phone", "L", "P"), ("speaker_id", "P", "L"),
    ):
        value = float(
            observed_lookup[(factor, positive)]["value"]
            - observed_lookup[(factor, negative)]["value"]
        )
        contrast_draws = np.asarray(draws[(factor, positive)]) - np.asarray(draws[(factor, negative)])
        rows.append({
            "metric": "MIG", "component": "route_contrast", "target": factor,
            "view": f"{positive}-{negative}", "value": value,
            "ci95_low": float(np.quantile(contrast_draws, .025)) if len(contrast_draws) else np.nan,
            "ci95_high": float(np.quantile(contrast_draws, .975)) if len(contrast_draws) else np.nan,
            "control": "observed", "top_unit": -1, "second_unit": -1,
            "top_score": np.nan, "second_score": np.nan,
            "interval_method": "paired_target_cluster_bootstrap",
        })

    rng = np.random.default_rng(seed + 1601)
    for factor, y in labels.items():
        if factor == "speaker_id":
            utterances = segments[["utterance_id", "speaker_id"]].drop_duplicates("utterance_id")
            permuted = rng.permutation(utterances["speaker_id"].to_numpy())
            mapping = dict(zip(utterances["utterance_id"], permuted))
            shuffled_raw = segments["utterance_id"].map(mapping).to_numpy()
        else:
            shuffled_raw = rng.permutation(raw_labels[factor])
        shuffled, _ = _encode_labels(shuffled_raw)
        entropy = _entropy(np.bincount(shuffled) / max(len(shuffled), 1))
        mi = _mutual_information_scores(binned, shuffled)
        score = mi / entropy if entropy > 0 else np.zeros_like(mi)
        for view, units in views.items():
            gap, first_local, second_local = _gap(score[units])
            rows.append({
                "metric": "MIG", "component": "factor_gap", "target": factor,
                "view": view, "value": gap, "ci95_low": np.nan, "ci95_high": np.nan,
                "control": "label_shuffle",
                "top_unit": int(units[first_local]) if first_local >= 0 else -1,
                "second_unit": int(units[second_local]) if second_local >= 0 else -1,
                "top_score": float(score[units[first_local]]) if first_local >= 0 else np.nan,
                "second_score": float(score[units[second_local]]) if second_local >= 0 else np.nan,
                "interval_method": "none",
            })
    return rows, candidates


def _stratified_utterance_split(
    segments: pd.DataFrame,
    seed: int,
    test_fraction: float = .30,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    utterances = segments[["utterance_id", "speaker_id"]].drop_duplicates("utterance_id")
    train_utterances: list[str] = []
    test_utterances: list[str] = []
    for _, group in utterances.groupby("speaker_id", sort=True, observed=True):
        ids = group["utterance_id"].astype(str).to_numpy().copy()
        rng.shuffle(ids)
        if len(ids) < 2:
            train_utterances.extend(ids.tolist())
            continue
        n_test = min(len(ids) - 1, max(1, int(round(test_fraction * len(ids)))))
        test_utterances.extend(ids[:n_test].tolist())
        train_utterances.extend(ids[n_test:].tolist())
    train = np.flatnonzero(segments["utterance_id"].isin(train_utterances).to_numpy())
    test = np.flatnonzero(segments["utterance_id"].isin(test_utterances).to_numpy())
    return train.astype(np.int32), test.astype(np.int32)


def _univariate_predictability(
    binned,
    labels: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    columns: np.ndarray | None = None,
) -> np.ndarray:
    matrix = binned.tocsc(copy=False)
    columns = np.arange(matrix.shape[1], dtype=np.int32) if columns is None else np.asarray(columns, dtype=np.int32)
    n_classes = int(labels.max()) + 1
    global_majority = int(np.bincount(labels[train_rows], minlength=n_classes).argmax())
    train_selected = np.zeros(matrix.shape[0], dtype=bool)
    test_selected = np.zeros(matrix.shape[0], dtype=bool)
    train_selected[train_rows] = True
    test_selected[test_rows] = True
    train_class_total = np.bincount(labels[train_rows], minlength=n_classes)
    test_class_total = np.bincount(labels[test_rows], minlength=n_classes).astype(np.float64)
    valid_classes = test_class_total > 0
    result = np.zeros(len(columns), dtype=np.float64)
    for result_index, column in enumerate(columns):
        start, end = int(matrix.indptr[column]), int(matrix.indptr[column + 1])
        column_rows = matrix.indices[start:end]
        bins = matrix.data[start:end].astype(np.int32)
        max_bin = max(int(bins.max()) if len(bins) else 0, 1)
        train_counts = np.zeros((n_classes, max_bin + 1), dtype=np.int64)
        test_counts = np.zeros((n_classes, max_bin + 1), dtype=np.int64)
        train_keep = train_selected[column_rows]
        test_keep = test_selected[column_rows]
        train_active_rows, train_active_bins = column_rows[train_keep], bins[train_keep]
        test_active_rows, test_active_bins = column_rows[test_keep], bins[test_keep]
        if len(train_active_rows):
            np.add.at(train_counts, (labels[train_active_rows], train_active_bins), 1)
            train_active_by_class = np.bincount(labels[train_active_rows], minlength=n_classes)
        else:
            train_active_by_class = np.zeros(n_classes, dtype=int)
        if len(test_active_rows):
            np.add.at(test_counts, (labels[test_active_rows], test_active_bins), 1)
            test_active_by_class = np.bincount(labels[test_active_rows], minlength=n_classes)
        else:
            test_active_by_class = np.zeros(n_classes, dtype=int)
        train_counts[:, 0] = train_class_total - train_active_by_class
        test_counts[:, 0] = test_class_total.astype(np.int64) - test_active_by_class
        predictions = np.full(train_counts.shape[1], global_majority, dtype=np.int32)
        nonempty = train_counts.sum(axis=0) > 0
        predictions[nonempty] = train_counts[:, nonempty].argmax(axis=0)
        correct = np.zeros(n_classes, dtype=np.float64)
        for bin_index, prediction in enumerate(predictions):
            if bin_index < test_counts.shape[1]:
                correct[prediction] += test_counts[prediction, bin_index]
        recalls = np.divide(correct, test_class_total, out=np.zeros_like(correct), where=valid_classes)
        result[result_index] = float(recalls[valid_classes].mean()) if valid_classes.any() else 0.0
    return result


def _dci_scores(importance: np.ndarray) -> tuple[float, float, np.ndarray]:
    matrix = np.maximum(np.asarray(importance, dtype=np.float64), 0.0)
    dimensions, factors = matrix.shape
    code_weight = matrix.sum(axis=1)
    total = float(code_weight.sum())
    if total <= 0 or factors < 2:
        return 0.0, 0.0, np.zeros(factors, dtype=np.float64)
    factor_given_code = np.divide(
        matrix, code_weight[:, None], out=np.zeros_like(matrix), where=code_weight[:, None] > 0,
    )
    code_disentanglement = np.asarray([
        1.0 - _entropy(row) / math.log(factors) if row.sum() > 0 else 0.0
        for row in factor_given_code
    ])
    disentanglement = float(np.sum(code_weight / total * code_disentanglement))
    completeness = np.zeros(factors, dtype=np.float64)
    if dimensions > 1:
        for factor in range(factors):
            weights = matrix[:, factor]
            if weights.sum() > 0:
                completeness[factor] = 1.0 - _entropy(weights / weights.sum()) / math.log(dimensions)
    return disentanglement, float(completeness.mean()), completeness


def _predictor_rows(
    x,
    binned,
    labels: dict[str, np.ndarray],
    segments: pd.DataFrame,
    views: dict[str, np.ndarray],
    cache: FeatureCache,
    seed: int,
    repeats: int,
    estimators: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.metrics import balanced_accuracy_score
    except ImportError as exc:
        raise AnalysisError(
            "Speech factor metrics require scikit-learn; install SAEUnitAnalysis/requirements.txt."
        ) from exc

    metric_draws: dict[tuple[str, str, str], list[float]] = {}
    importance_draws: dict[tuple[str, str], list[np.ndarray]] = {}
    repeat_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 2303)

    for repeat in range(max(1, repeats)):
        train_rows, test_rows = _stratified_utterance_split(segments, seed + 101 * repeat)
        if len(train_rows) == 0 or len(test_rows) == 0:
            raise AnalysisError("Speech factor metrics need at least two utterances per evaluated speaker.")
        shuffled_labels: dict[str, np.ndarray] = {}
        for factor, y in labels.items():
            if factor == "speaker_id":
                utterances = segments[["utterance_id", "speaker_id"]].drop_duplicates("utterance_id")
                shuffled_values = rng.permutation(utterances["speaker_id"].to_numpy())
                mapping = dict(zip(utterances["utterance_id"], shuffled_values))
                raw = segments["utterance_id"].map(mapping).to_numpy()
                shuffled_labels[factor], _ = _encode_labels(raw)
            else:
                shuffled_labels[factor] = rng.permutation(y)

        # SAP is a coordinate-level diagnostic. Compute every unit once per
        # factor/split, then take the two best coordinates within each view.
        sap_all = {
            factor: _univariate_predictability(binned, y, train_rows, test_rows)
            for factor, y in labels.items()
        }
        sap_null_all = (
            {
                factor: _univariate_predictability(
                    binned, shuffled_labels[factor], train_rows, test_rows,
                )
                for factor in FACTOR_NAMES
            }
            if repeat == 0 else {}
        )

        for view, units in views.items():
            if len(units) < 2:
                continue
            view_x = x[:, units].tocsc()
            importance = np.zeros((len(units), len(FACTOR_NAMES)), dtype=np.float64)
            informativeness: dict[str, float] = {}
            for factor_index, factor in enumerate(FACTOR_NAMES):
                y = labels[factor]
                classifier = ExtraTreesClassifier(
                    n_estimators=max(8, int(estimators)), max_depth=18,
                    min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
                    n_jobs=-1, random_state=seed + 1009 * repeat + 17 * factor_index,
                )
                classifier.fit(view_x[train_rows], y[train_rows])
                prediction = classifier.predict(view_x[test_rows])
                accuracy = float(balanced_accuracy_score(y[test_rows], prediction))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    feature_importance = classifier.feature_importances_
                chance = 1.0 / max(len(np.unique(y)), 1)
                evidence = max(0.0, (accuracy - chance) / max(1.0 - chance, 1e-12))
                importance[:, factor_index] = np.nan_to_num(
                    feature_importance, nan=0.0, posinf=0.0, neginf=0.0,
                ) * evidence
                informativeness[factor] = accuracy
                sap_gap, _, _ = _gap(sap_all[factor][units])
                metric_draws.setdefault(("SAP", factor, view), []).append(sap_gap)
                metric_draws.setdefault(("DCI_informativeness", factor, view), []).append(accuracy)
                repeat_rows.extend([
                    {
                        "repeat": repeat, "metric": "SAP", "component": "factor_gap",
                        "target": factor, "view": view, "value": sap_gap,
                    },
                    {
                        "repeat": repeat, "metric": "DCI", "component": "informativeness",
                        "target": factor, "view": view, "value": accuracy,
                    },
                ])
            dci_disentanglement, dci_completeness, factor_completeness = _dci_scores(importance)
            metric_draws.setdefault(("DCI_disentanglement", "all", view), []).append(dci_disentanglement)
            metric_draws.setdefault(("DCI_completeness", "mean", view), []).append(dci_completeness)
            for factor_index, factor in enumerate(FACTOR_NAMES):
                metric_draws.setdefault(("DCI_completeness", factor, view), []).append(
                    float(factor_completeness[factor_index])
                )
            importance_draws.setdefault((view, "observed"), []).append(importance)
            repeat_rows.extend([
                {"repeat": repeat, "metric": "DCI", "component": "disentanglement",
                 "target": "all", "view": view, "value": dci_disentanglement},
                {"repeat": repeat, "metric": "DCI", "component": "completeness",
                 "target": "mean", "view": view, "value": dci_completeness},
            ])

            if repeat == 0:
                null_importance = np.zeros_like(importance)
                for factor_index, factor in enumerate(FACTOR_NAMES):
                    y_null = shuffled_labels[factor]
                    classifier = ExtraTreesClassifier(
                        n_estimators=max(8, int(estimators)), max_depth=18,
                        min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
                        n_jobs=-1, random_state=seed + 7001 + factor_index,
                    )
                    classifier.fit(view_x[train_rows], y_null[train_rows])
                    prediction = classifier.predict(view_x[test_rows])
                    accuracy = float(balanced_accuracy_score(y_null[test_rows], prediction))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        feature_importance = classifier.feature_importances_
                    chance = 1.0 / max(len(np.unique(y_null)), 1)
                    evidence = max(0.0, (accuracy - chance) / max(1.0 - chance, 1e-12))
                    null_importance[:, factor_index] = np.nan_to_num(
                        feature_importance, nan=0.0, posinf=0.0, neginf=0.0,
                    ) * evidence
                    sap_null = sap_null_all[factor][units]
                    metric_draws.setdefault(("SAP_null", factor, view), []).append(_gap(sap_null)[0])
                    metric_draws.setdefault(("DCI_informativeness_null", factor, view), []).append(accuracy)
                null_d, null_c, null_fc = _dci_scores(null_importance)
                metric_draws.setdefault(("DCI_disentanglement_null", "all", view), []).append(null_d)
                metric_draws.setdefault(("DCI_completeness_null", "mean", view), []).append(null_c)
                for factor_index, factor in enumerate(FACTOR_NAMES):
                    metric_draws.setdefault(("DCI_completeness_null", factor, view), []).append(
                        float(null_fc[factor_index])
                    )

    rows: list[dict[str, Any]] = []
    for (metric_key, target, view), values in metric_draws.items():
        is_null = metric_key.endswith("_null")
        clean = metric_key.removesuffix("_null")
        if clean.startswith("DCI_"):
            metric, component = "DCI", clean.removeprefix("DCI_")
        else:
            metric, component = clean, "factor_gap"
        observed = np.asarray(values, dtype=np.float64)
        rows.append({
            "metric": metric, "component": component, "target": target, "view": view,
            "value": float(observed.mean()),
            "ci95_low": float(np.quantile(observed, .025)) if len(observed) > 1 else np.nan,
            "ci95_high": float(np.quantile(observed, .975)) if len(observed) > 1 else np.nan,
            "control": "label_shuffle" if is_null else "observed",
            "top_unit": -1, "second_unit": -1, "top_score": np.nan, "second_score": np.nan,
            "interval_method": "repeated_utterance_group_holdout" if len(observed) > 1 else "none",
        })

    for metric_key in ("SAP", "DCI_informativeness"):
        for factor, positive, negative in (
            ("phone", "L", "P"), ("speaker_id", "P", "L"),
        ):
            positive_values = np.asarray(metric_draws[(metric_key, factor, positive)], dtype=np.float64)
            negative_values = np.asarray(metric_draws[(metric_key, factor, negative)], dtype=np.float64)
            contrast = positive_values - negative_values
            metric = "DCI" if metric_key.startswith("DCI_") else metric_key
            rows.append({
                "metric": metric, "component": "route_contrast", "target": factor,
                "view": f"{positive}-{negative}", "value": float(contrast.mean()),
                "ci95_low": float(np.quantile(contrast, .025)) if len(contrast) > 1 else np.nan,
                "ci95_high": float(np.quantile(contrast, .975)) if len(contrast) > 1 else np.nan,
                "control": "observed", "top_unit": -1, "second_unit": -1,
                "top_score": np.nan, "second_score": np.nan,
                "interval_method": "paired_repeated_utterance_group_holdout",
            })

    importance_rows: list[dict[str, Any]] = []
    for (view, control), arrays in importance_draws.items():
        mean_importance = np.mean(arrays, axis=0)
        units = views[view]
        for local, unit in enumerate(units):
            importance_rows.append({
                "view": view, "unit": int(unit),
                "route": "L" if cache.route[unit] == 0 else "P" if cache.route[unit] == 1 else "U",
                "phone_importance": float(mean_importance[local, 0]),
                "speaker_importance": float(mean_importance[local, 1]),
                "total_importance": float(mean_importance[local].sum()),
            })
    return rows, pd.DataFrame(importance_rows), pd.DataFrame(repeat_rows)


def _add_mean_rows(table: pd.DataFrame) -> pd.DataFrame:
    observed = table[
        (table["control"] == "observed")
        & (table["component"] == "factor_gap")
        & (table["target"].isin(FACTOR_NAMES))
    ]
    additions = []
    for (metric, view), group in observed.groupby(["metric", "view"], observed=True):
        if set(group.target) != set(FACTOR_NAMES):
            continue
        additions.append({
            "metric": metric, "component": "mean_factor_gap", "target": "mean",
            "view": view, "value": float(group.value.mean()),
            "ci95_low": np.nan, "ci95_high": np.nan, "control": "observed",
            "top_unit": -1, "second_unit": -1, "top_score": np.nan, "second_score": np.nan,
            "interval_method": "not_pooled",
        })
    return pd.concat([table, pd.DataFrame(additions)], ignore_index=True) if additions else table


def _prediction_mig(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Predictive lower bound on normalized MI for a complete route vector."""
    from sklearn.metrics import mutual_info_score

    counts = np.bincount(np.asarray(y_true, dtype=np.int32))
    entropy = _entropy(counts / max(int(counts.sum()), 1))
    if entropy <= 0:
        return 0.0
    return float(mutual_info_score(y_true, y_pred) / entropy)


def _shuffle_factor_labels(
    factor: str,
    labels: np.ndarray,
    segments: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle labels without leaving adjacent segments as independent controls."""
    if factor != "speaker_id":
        return rng.permutation(labels).astype(np.int32)
    utterances = segments[["utterance_id"]].drop_duplicates("utterance_id")
    utterance_labels = rng.choice(
        np.unique(labels), size=len(utterances), replace=True,
    ).astype(np.int32)
    mapping = dict(zip(utterances["utterance_id"].astype(str), utterance_labels))
    return segments["utterance_id"].astype(str).map(mapping).to_numpy(dtype=np.int32)


def _route_subspace_rows(
    x,
    labels: dict[str, np.ndarray],
    segments: pd.DataFrame,
    views: dict[str, np.ndarray],
    *,
    seed: int,
    repeats: int,
    estimators: int,
    null_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate factors from whole routes rather than individual SAE coordinates.

    DCI uses nonlinear ExtraTrees informativeness. Grouped SAP uses a linear
    SVM trained by SGD. Grouped MIG is normalized mutual information between the
    true factor and the held-out nonlinear prediction, which is a conservative
    predictive lower bound on I(factor; route).
    """
    try:
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.preprocessing import MaxAbsScaler
    except ImportError as exc:
        raise AnalysisError(
            "Grouped route metrics require scikit-learn; install SAEUnitAnalysis/requirements.txt."
        ) from exc

    activity = np.asarray(x.getnnz(axis=0)).reshape(-1)
    route_views = {name: np.asarray(views[name], dtype=np.int32) for name in ("L", "P")}
    matched_count = min(len(route_views["L"]), len(route_views["P"]))
    matched_views: dict[str, np.ndarray] = {}
    for route, units in route_views.items():
        order = np.lexsort((units, -activity[units]))
        matched_views[route] = np.sort(units[order[:matched_count]])
    capacity_views = {
        "all_observed": {
            "L": route_views["L"],
            "P": route_views["P"],
        },
        "matched_active_units": matched_views,
    }

    draw_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed + 3907)
    factors = tuple(labels)
    for repeat in range(max(2, int(repeats))):
        train_rows, test_rows = _stratified_utterance_split(segments, seed + 101 * repeat)
        if len(train_rows) == 0 or len(test_rows) == 0:
            raise AnalysisError("Grouped route metrics need non-empty grouped train/test splits.")
        controls = ["observed"]
        if repeat < max(1, int(null_repeats)):
            controls.append("label_shuffle")
        shuffled = {
            factor: _shuffle_factor_labels(factor, y, segments, rng)
            for factor, y in labels.items()
        }
        for capacity_mode, mode_views in capacity_views.items():
            for view, units in mode_views.items():
                view_x = x[:, units].tocsr()
                scaler = MaxAbsScaler(copy=True)
                linear_train = scaler.fit_transform(view_x[train_rows])
                linear_test = scaler.transform(view_x[test_rows])
                for control in controls:
                    for factor_index, factor in enumerate(factors):
                        y = labels[factor] if control == "observed" else shuffled[factor]
                        chance = 1.0 / max(len(np.unique(y)), 1)
                        nonlinear = ExtraTreesClassifier(
                            n_estimators=max(12, int(estimators)), max_depth=18,
                            min_samples_leaf=2, max_features="sqrt",
                            class_weight="balanced", n_jobs=-1,
                            random_state=(
                                seed + 1009 * repeat + 31 * factor_index
                                + (0 if control == "observed" else 100_003)
                                + (0 if capacity_mode == "all_observed" else 200_003)
                            ),
                        )
                        nonlinear.fit(view_x[train_rows], y[train_rows])
                        nonlinear_prediction = nonlinear.predict(view_x[test_rows])
                        nonlinear_accuracy = float(
                            balanced_accuracy_score(y[test_rows], nonlinear_prediction)
                        )
                        linear = SGDClassifier(
                            loss="hinge", alpha=1e-4, class_weight="balanced",
                            max_iter=300, tol=1e-3, average=True, n_jobs=-1,
                            random_state=(
                                seed + 2029 * repeat + 43 * factor_index
                                + (0 if control == "observed" else 300_007)
                                + (0 if capacity_mode == "all_observed" else 400_009)
                            ),
                        )
                        linear.fit(linear_train, y[train_rows])
                        linear_prediction = linear.predict(linear_test)
                        linear_accuracy = float(
                            balanced_accuracy_score(y[test_rows], linear_prediction)
                        )
                        common = {
                            "repeat": int(repeat), "target": factor, "view": view,
                            "capacity_mode": capacity_mode, "control": control,
                            "chance": float(chance), "route_units": int(len(units)),
                        }
                        draw_rows.extend([
                            {
                                **common, "metric": "MIG", "component": "informativeness",
                                "value": _prediction_mig(y[test_rows], nonlinear_prediction),
                                "estimator": "normalized_MI_of_ExtraTrees_prediction",
                            },
                            {
                                **common, "metric": "SAP", "component": "informativeness",
                                "value": linear_accuracy,
                                "estimator": "SGD_linear_SVM_balanced_accuracy",
                            },
                            {
                                **common, "metric": "DCI", "component": "informativeness",
                                "value": nonlinear_accuracy,
                                "estimator": "ExtraTrees_balanced_accuracy",
                            },
                        ])

    draws = pd.DataFrame(draw_rows)
    summary_rows: list[dict[str, Any]] = []
    group_columns = [
        "metric", "component", "target", "view", "capacity_mode", "control",
        "chance", "route_units", "estimator",
    ]
    for keys, group in draws.groupby(group_columns, sort=False, observed=True, dropna=False):
        values = group["value"].to_numpy(dtype=np.float64)
        row = dict(zip(group_columns, keys))
        row.update({
            "value": float(values.mean()),
            "ci95_low": float(np.quantile(values, .025)) if len(values) > 1 else float(values[0]),
            "ci95_high": float(np.quantile(values, .975)) if len(values) > 1 else float(values[0]),
            "interval_method": "repeated_utterance_group_holdout",
            "repetitions": int(len(values)),
            "scope": "route_subspace",
            "status": "primary" if row["capacity_mode"] == "all_observed" else "capacity_control",
        })
        summary_rows.append(row)

    # Paired desired-route minus other-route contrasts are the primary
    # disentanglement quantities. They never compare units within a route.
    for (metric, target, capacity_mode, control), group in draws[
        draws["view"].isin(["L", "P"])
    ].groupby(["metric", "target", "capacity_mode", "control"], sort=False, observed=True):
        desired, other = ("L", "P") if target == "phone" else ("P", "L")
        wide = group.pivot_table(index="repeat", columns="view", values="value", aggfunc="first")
        if desired not in wide or other not in wide:
            continue
        contrast = (wide[desired] - wide[other]).dropna().to_numpy(dtype=np.float64)
        source = group.iloc[0]
        summary_rows.append({
            "metric": metric, "component": "route_contrast", "target": target,
            "view": f"{desired}-{other}", "capacity_mode": capacity_mode,
            "control": control, "chance": 0.0,
            "route_units": int(min(
                group[group.view == "L"].route_units.min(),
                group[group.view == "P"].route_units.min(),
            )),
            "estimator": source.estimator,
            "value": float(contrast.mean()),
            "ci95_low": float(np.quantile(contrast, .025)),
            "ci95_high": float(np.quantile(contrast, .975)),
            "interval_method": "paired_repeated_utterance_group_holdout",
            "repetitions": int(len(contrast)), "scope": "route_subspace",
            "status": "primary" if capacity_mode == "all_observed" else "capacity_control",
        })

    # Grouped DCI: the two rows of the importance matrix are the L and P
    # subspaces, and the columns are phone and speaker. Evidence is each
    # route's chance-corrected held-out informativeness.
    dci_draws = draws[
        (draws.metric == "DCI") & (draws.control == "observed")
        & (draws.view.isin(["L", "P"]))
    ]
    structure_draws: list[dict[str, Any]] = []
    for (capacity_mode, repeat), group in dci_draws.groupby(
        ["capacity_mode", "repeat"], sort=False, observed=True,
    ):
        matrix = np.zeros((2, 2), dtype=np.float64)
        for route_index, route in enumerate(("L", "P")):
            for factor_index, factor in enumerate(("phone", "speaker_id")):
                match = group[(group.view == route) & (group.target == factor)]
                if match.empty:
                    continue
                item = match.iloc[0]
                matrix[route_index, factor_index] = max(
                    0.0, (float(item.value) - float(item.chance)) / max(1.0 - float(item.chance), 1e-12)
                )
        disentanglement, completeness_mean, completeness = _dci_scores(matrix)
        shares = []
        for factor_index, desired_index in ((0, 0), (1, 1)):
            total = float(matrix[:, factor_index].sum())
            shares.append(float(matrix[desired_index, factor_index] / total) if total > 0 else 0.0)
        structure_draws.extend([
            {"capacity_mode": capacity_mode, "repeat": repeat, "component": "route_disentanglement", "target": "all", "value": disentanglement},
            {"capacity_mode": capacity_mode, "repeat": repeat, "component": "route_completeness", "target": "mean", "value": completeness_mean},
            {"capacity_mode": capacity_mode, "repeat": repeat, "component": "route_completeness", "target": "phone", "value": float(completeness[0])},
            {"capacity_mode": capacity_mode, "repeat": repeat, "component": "route_completeness", "target": "speaker_id", "value": float(completeness[1])},
            {"capacity_mode": capacity_mode, "repeat": repeat, "component": "directional_alignment", "target": "mean", "value": float(np.mean(shares))},
            {"capacity_mode": capacity_mode, "repeat": repeat, "component": "directional_alignment", "target": "phone", "value": shares[0]},
            {"capacity_mode": capacity_mode, "repeat": repeat, "component": "directional_alignment", "target": "speaker_id", "value": shares[1]},
        ])
    structure = pd.DataFrame(structure_draws)
    for (capacity_mode, component, target), group in structure.groupby(
        ["capacity_mode", "component", "target"], sort=False, observed=True,
    ):
        values = group.value.to_numpy(dtype=np.float64)
        summary_rows.append({
            "metric": "DCI", "component": component, "target": target,
            "view": "L|P", "capacity_mode": capacity_mode, "control": "observed",
            "chance": 0.5 if component == "directional_alignment" else 0.0,
            "route_units": int(matched_count if capacity_mode == "matched_active_units" else len(route_views["L"]) + len(route_views["P"])),
            "estimator": "chance_corrected_route_evidence_matrix",
            "value": float(values.mean()), "ci95_low": float(np.quantile(values, .025)),
            "ci95_high": float(np.quantile(values, .975)),
            "interval_method": "repeated_utterance_group_holdout",
            "repetitions": int(len(values)), "scope": "route_subspace",
            "status": "primary" if capacity_mode == "all_observed" else "capacity_control",
        })

    metrics = pd.DataFrame(summary_rows)
    evidence = metrics[
        (metrics.metric == "DCI") & (metrics.component == "informativeness")
        & (metrics.control == "observed")
    ].copy()
    evidence["chance_corrected_evidence"] = np.maximum(
        0.0,
        (evidence.value - evidence.chance) / np.maximum(1.0 - evidence.chance, 1e-12),
    )
    return metrics, evidence, pd.concat([draws, structure.assign(
        metric="DCI", view="L|P", control="observed", scope="route_subspace"
    )], ignore_index=True, sort=False)


def _headline(table: pd.DataFrame, metric: str, target: str, positive: str, negative: str) -> float | None:
    contrast = table[
        (table.metric == metric) & (table.target == target)
        & (table.control == "observed") & (table.component == "route_contrast")
        & (table.view == f"{positive}-{negative}")
    ]
    if len(contrast):
        return float(contrast.iloc[0].value)
    subset = table[
        (table.metric == metric) & (table.target == target)
        & (table.control == "observed")
    ]
    if metric == "DCI":
        subset = subset[subset.component == "informativeness"]
    else:
        subset = subset[subset.component == "factor_gap"]
    values = subset.set_index("view").value.to_dict()
    if positive not in values or negative not in values:
        return None
    return float(values[positive] - values[negative])


def speech_factor_metrics(
    cache: FeatureCache,
    bundle: AnalysisBundle,
    output: Path,
    *,
    seed: int = 42,
    quick: bool = False,
    evaluation_split: str = "test",
    max_segments: int | None = None,
    positive_bins: int = 4,
    bootstrap_repetitions: int | None = None,
    dci_repeats: int | None = None,
    dci_estimators: int | None = None,
    null_repeats: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute grouped MIG, SAP and DCI for the complete L/P subspaces.

    The intended object of disentanglement is the partition z=(z_L,z_P), not
    independence among individual coordinates inside either subspace. Historic
    coordinate-gap results are archived as deprecated artifacts on upgrade.
    """
    max_segments = int(max_segments if max_segments is not None else (2_000 if quick else 30_000))
    # Retained in the public signature for compatibility with older commands.
    bootstrap_repetitions = int(
        bootstrap_repetitions if bootstrap_repetitions is not None else (20 if quick else 0)
    )
    dci_repeats = int(dci_repeats if dci_repeats is not None else (2 if quick else 10))
    dci_estimators = int(dci_estimators if dci_estimators is not None else (16 if quick else 48))
    null_repeats = int(null_repeats if null_repeats is not None else (1 if quick else 2))

    tables_dir = output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    existing_metrics = tables_dir / "speech_factor_metrics.csv"
    deprecated_metrics = tables_dir / "deprecated_unit_compactness_metrics.csv"
    if existing_metrics.exists() and not deprecated_metrics.exists():
        previous = pd.read_csv(existing_metrics)
        if "scope" not in previous or not previous["scope"].eq("route_subspace").all():
            previous["status"] = "deprecated_within_route_coordinate_metric"
            previous.to_csv(deprecated_metrics, index=False)
    existing_importance = tables_dir / "dci_unit_importances.csv"
    deprecated_importance = tables_dir / "deprecated_dci_unit_importances.csv"
    if existing_importance.exists() and not deprecated_importance.exists():
        pd.read_csv(existing_importance).to_csv(deprecated_importance, index=False)

    segments_available = _phone_segments(cache, bundle, evaluation_split)
    if segments_available.empty:
        raise AnalysisError(
            f"Speech factor metrics found no aligned phone segments in {evaluation_split!r}."
        )
    if segments_available.phone.nunique() < 2 or segments_available.speaker_id.nunique() < 2:
        raise AnalysisError("Speech factor metrics need at least two phones and two speakers.")
    segments, balance = _joint_balanced_sample(segments_available, max_segments, seed)
    if len(segments) < 20:
        raise AnalysisError("Speech factor metrics need at least 20 balanced phone segments.")
    segment_topk = min(cache.indices.shape[1], cache.K)
    x = _segment_matrix(cache, segments, segment_topk)
    raw_labels = {
        "phone": segments.phone.astype(str).to_numpy(),
        "speaker_id": segments.speaker_id.astype(str).to_numpy(),
    }
    labels = {name: _encode_labels(values)[0] for name, values in raw_labels.items()}
    active_units = np.flatnonzero(np.asarray(x.getnnz(axis=0)).reshape(-1) > 0).astype(np.int32)
    views = {
        name: np.intersect1d(units, active_units, assume_unique=True).astype(np.int32)
        for name, units in _view_units(cache).items()
    }
    views = {name: units for name, units in views.items() if len(units) >= 2}
    if not {"L", "P"}.issubset(views):
        raise AnalysisError("Speech factor metrics require at least two assigned units in both L and P.")

    metrics, importance, repeats = _route_subspace_rows(
        x, labels, segments, views, seed=seed, repeats=dci_repeats,
        estimators=dci_estimators, null_repeats=null_repeats,
    )
    metrics.insert(0, "speech_adapted", True)
    metrics["observations"] = int(len(segments))
    metrics["phones"] = int(segments.phone.nunique())
    metrics["speakers"] = int(segments.speaker_id.nunique())
    metrics["observed_units_in_view"] = metrics["route_units"].astype(int)
    metrics.to_csv(output / "tables" / "speech_factor_metrics.csv", index=False)
    importance.to_csv(output / "tables" / "route_dci_evidence.csv", index=False)
    repeats.to_csv(output / "tables" / "speech_factor_metric_repeats.csv", index=False)
    try:
        metrics.to_parquet(output / "tables" / "speech_factor_metrics.parquet", index=False)
        importance.to_parquet(output / "tables" / "route_dci_evidence.parquet", index=False)
    except Exception:
        pass

    headline = {
        "MIG_phone_L_minus_P": _headline(metrics, "MIG", "phone", "L", "P"),
        "MIG_speaker_P_minus_L": _headline(metrics, "MIG", "speaker_id", "P", "L"),
        "SAP_phone_L_minus_P": _headline(metrics, "SAP", "phone", "L", "P"),
        "SAP_speaker_P_minus_L": _headline(metrics, "SAP", "speaker_id", "P", "L"),
        "DCI_phone_informativeness_L_minus_P": _headline(metrics, "DCI", "phone", "L", "P"),
        "DCI_speaker_informativeness_P_minus_L": _headline(metrics, "DCI", "speaker_id", "P", "L"),
    }
    headline_intervals = {}
    headline_specs = {
        "MIG_phone_L_minus_P": ("MIG", "phone", "L-P"),
        "MIG_speaker_P_minus_L": ("MIG", "speaker_id", "P-L"),
        "SAP_phone_L_minus_P": ("SAP", "phone", "L-P"),
        "SAP_speaker_P_minus_L": ("SAP", "speaker_id", "P-L"),
        "DCI_phone_informativeness_L_minus_P": ("DCI", "phone", "L-P"),
        "DCI_speaker_informativeness_P_minus_L": ("DCI", "speaker_id", "P-L"),
    }
    for name, (metric, target, view) in headline_specs.items():
        match = metrics[
            (metrics.metric == metric) & (metrics.target == target)
            & (metrics.component == "route_contrast") & (metrics.view == view)
            & (metrics.control == "observed")
        ]
        if len(match):
            row = match.iloc[0]
            headline_intervals[name] = {
                "value": float(row.value),
                "ci95_low": float(row.ci95_low) if pd.notna(row.ci95_low) else None,
                "ci95_high": float(row.ci95_high) if pd.notna(row.ci95_high) else None,
                "interval_method": str(row.interval_method),
            }
    summary = {
        "status": "ok",
        "speech_adapted": True,
        "scope": "route_subspace",
        "evaluation_split": evaluation_split,
        "available_segments": int(len(segments_available)),
        "sampled_segments": int(len(segments)),
        "sampled_utterances": int(segments.utterance_id.nunique()),
        "phones": int(segments.phone.nunique()),
        "speakers": int(segments.speaker_id.nunique()),
        "balance": balance,
        "segment_representation": f"mean_pooled_then_top_{segment_topk}",
        "route_metric_definitions": {
            "MIG": "I(target; held-out ExtraTrees prediction from route) / H(target)",
            "SAP": "held-out RidgeClassifier balanced accuracy from the complete route",
            "DCI_informativeness": "held-out ExtraTrees balanced accuracy from the complete route",
            "DCI_structure": "standard DCI entropy equations on the 2-route x 2-factor chance-corrected evidence matrix",
        },
        "interval_method": "repeated_utterance_group_holdout_stability",
        "repeats": int(dci_repeats),
        "null_repeats": int(null_repeats),
        "DCI_predictor": "ExtraTreesClassifier",
        "DCI_estimators_per_factor": int(dci_estimators),
        "SAP_predictor": "MaxAbsScaler_plus_SGD_linear_SVM",
        "capacity_control": "same_number_of_most_active_observed_units_in_L_and_P",
        "observed_units_by_view": {name: int(len(units)) for name, units in views.items()},
        "headline_route_contrasts": headline,
        "headline_route_contrast_intervals": headline_intervals,
        "interpretation": (
            "The evaluated object is the L/P subspace partition. Positive phone L-P and "
            "speaker P-L contrasts mean that the intended complete route carries more "
            "held-out factor information; no within-route unit independence is claimed."
        ),
        "caveats": [
            "Natural speech is not a complete speaker-by-phone factorial design; observed cells are capped to reduce imbalance.",
            "Grouped MIG is a predictive lower bound based on held-out predictions, not a high-dimensional plug-in MI estimate.",
            "Intervals summarize repeated grouped train/test splits and are estimator-stability intervals, not population bootstrap confidence intervals.",
            "MIG, SAP and DCI are post-hoc labelled metrics; latent swaps remain the causal evidence.",
        ],
        "deprecated_artifact": str(deprecated_metrics) if deprecated_metrics.exists() else None,
    }
    write_json(output / "speech_factor_metrics.json", summary)
    return metrics, importance, repeats, summary
