from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analyses import cache_metadata, frame_to_utterance
from .bundle import AnalysisBundle
from .extraction import FeatureCache
from .types import ResolvedModel
from .utils import AnalysisError, fingerprint, write_json


@dataclass
class EvaluatorSuite:
    phone: Any | None
    speaker: Any | None
    emotion: Any | None
    prosody: Any | None
    phone_classes: np.ndarray
    speaker_classes: np.ndarray
    emotion_classes: np.ndarray
    evaluation_utterance_ids: np.ndarray


def _imports():
    try:
        from sklearn.linear_model import Ridge, SGDClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.pipeline import make_pipeline
        import joblib
    except ImportError as exc:
        raise AnalysisError(
            "Causal analyses require scikit-learn and joblib; install "
            "SAEUnitAnalysis/requirements.txt."
        ) from exc
    return Ridge, SGDClassifier, LabelEncoder, StandardScaler, make_pipeline, joblib


def train_evaluators(
    cache: FeatureCache, bundle: AnalysisBundle, cache_dir: Path,
    resolved: ResolvedModel, seed: int = 42,
) -> EvaluatorSuite:
    Ridge, SGDClassifier, LabelEncoder, StandardScaler, make_pipeline, joblib = _imports()
    evaluator_key = fingerprint(
        [cache.path],
        {"seed": int(seed), "schema": 3, "kind": "independent_sae_reconstruction_evaluators"},
    )
    path = cache_dir / f"independent-evaluators-{evaluator_key}.joblib"
    if path.exists():
        return joblib.load(path)

    metadata = cache_metadata(bundle, cache)
    split = metadata["split"].astype(str).to_numpy()
    train_name = str(bundle.spec.split_map.get("train", "train"))
    sample_frames = cache.h_sample_frames
    sample_utt = frame_to_utterance(cache, sample_frames)
    train_frames = split[sample_utt] == train_name
    decoder_weight = resolved.state["sae.dec_weight"].detach().float().cpu().numpy()
    decoder_bias = resolved.state["sae.b_pre"].detach().float().cpu().numpy()
    # Fit in the same feature domain used by the interventions. Only unswapped
    # baseline reconstructions are used; swapped examples never enter fitting.
    h_frame = cache.dense(sample_frames) @ decoder_weight.T + decoder_bias

    phone_model = None
    phone_classes = np.asarray([], dtype="U1")
    phone_y = cache.phones[sample_frames].astype(str)
    valid = train_frames & (phone_y != "<unaligned>")
    if valid.sum() >= 20 and len(np.unique(phone_y[valid])) >= 2:
        le = LabelEncoder().fit(phone_y[valid])
        phone_classes = le.classes_
        phone_model = make_pipeline(
            StandardScaler(),
            SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1000, tol=1e-3,
                          class_weight="balanced", random_state=seed),
        )
        phone_model.fit(h_frame[valid], le.transform(phone_y[valid]))

    speaker_model = emotion_model = None
    speaker_classes = emotion_classes = np.asarray([], dtype="U1")
    train_utt = split == train_name
    # The reconstructed utterance mean is exact from the cached mean latent and
    # avoids decoding every frame merely to fit the speaker measurement probe.
    h_stats = cache.pooled_z.astype(np.float32) @ decoder_weight.T + decoder_bias
    evaluation_utterance_ids = np.asarray([], dtype=cache.utterance_ids.dtype)
    if "speaker_id" in metadata:
        y = metadata["speaker_id"].astype(str).to_numpy()
        # LibriSpeech train/dev/test speakers are disjoint. Build the independent
        # measurement probe on a stratified portion of test utterances and reserve
        # different utterances from the same test speakers for interventions.
        test_name = str(bundle.spec.split_map.get("test", "test"))
        test_utt = np.flatnonzero((split == test_name) & (y != "nan"))
        rng = np.random.default_rng(seed)
        speaker_fit: list[int] = []
        speaker_eval: list[int] = []
        for speaker in sorted(np.unique(y[test_utt]).tolist()):
            ids = test_utt[y[test_utt] == speaker]
            ids = ids[rng.permutation(len(ids))]
            if len(ids) < 2:
                continue
            n_eval = max(1, int(round(0.25 * len(ids))))
            n_eval = min(n_eval, len(ids) - 1)
            speaker_eval.extend(ids[:n_eval].tolist())
            speaker_fit.extend(ids[n_eval:].tolist())
        valid_ids = np.asarray(speaker_fit, dtype=int)
        if len(valid_ids) >= 10 and len(np.unique(y[valid_ids])) >= 2:
            le = LabelEncoder().fit(y[valid_ids]); speaker_classes = le.classes_
            speaker_model = make_pipeline(
                StandardScaler(),
                SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1500, tol=1e-3,
                              class_weight="balanced", random_state=seed),
            )
            speaker_model.fit(h_stats[valid_ids], le.transform(y[valid_ids]))
            evaluation_utterance_ids = cache.utterance_ids[np.asarray(speaker_eval, dtype=int)]
    if "emotion" in metadata:
        y = metadata["emotion"].astype(str).to_numpy()
        valid_u = train_utt & (y != "nan")
        if valid_u.sum() >= 10 and len(np.unique(y[valid_u])) >= 2:
            le = LabelEncoder().fit(y[valid_u]); emotion_classes = le.classes_
            emotion_model = make_pipeline(
                StandardScaler(),
                SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=1500, tol=1e-3,
                              class_weight="balanced", random_state=seed),
            )
            emotion_model.fit(h_stats[valid_u], le.transform(y[valid_u]))

    prosody_model = None
    y_pros = np.stack([cache.f0[sample_frames], cache.energy[sample_frames], cache.voicing[sample_frames]], 1)
    if train_frames.sum() >= 20 and np.any(np.std(y_pros[train_frames], 0) > 0):
        prosody_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        prosody_model.fit(h_frame[train_frames], y_pros[train_frames])

    suite = EvaluatorSuite(
        phone_model, speaker_model, emotion_model, prosody_model,
        phone_classes, speaker_classes, emotion_classes,
        evaluation_utterance_ids,
    )
    joblib.dump(suite, path)
    write_json(cache_dir / "independent-evaluators.json", {
        "phone_classes": phone_classes.tolist(), "speaker_classes": speaker_classes.tolist(),
        "emotion_classes": emotion_classes.tolist(), "prosody": prosody_model is not None,
        "training_source": "unswapped baseline SAE reconstructions",
        "intervention_examples_used_for_training": False,
        "speaker_statistic": "mean reconstructed SPEAR feature",
        "speaker_protocol": "test-speaker stratified utterance holdout",
        "evaluation_utterance_ids": evaluation_utterance_ids.tolist(),
        "cache": str(cache.path),
        "seed": int(seed),
    })
    return suite


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for value in np.unique(y_true):
        mask = y_true == value
        if mask.any(): scores.append(np.mean(y_pred[mask] == value))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate_frames(suite: EvaluatorSuite, h: np.ndarray, phone: np.ndarray,
                    f0: np.ndarray, energy: np.ndarray, voicing: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    if suite.phone is not None:
        known = np.isin(phone.astype(str), suite.phone_classes) & (phone != "<unaligned>")
        if known.any():
            pred = suite.phone_classes[suite.phone.predict(h[known])]
            out["phone_accuracy"] = float(np.mean(pred == phone[known]))
    if suite.prosody is not None and len(h):
        pred = suite.prosody.predict(h)
        target = np.stack([f0, energy, voicing], 1)
        out["prosody_rmse"] = float(np.sqrt(np.mean((pred-target) ** 2)))
        for i, name in enumerate(("f0", "energy", "voicing")):
            if np.std(target[:, i]) > 0 and np.std(pred[:, i]) > 0:
                out[f"{name}_correlation"] = float(np.corrcoef(target[:, i], pred[:, i])[0, 1])
    return out


def evaluate_utterances(suite: EvaluatorSuite, h_stats: np.ndarray,
                        speaker: np.ndarray | None = None,
                        emotion: np.ndarray | None = None) -> dict[str, float]:
    out: dict[str, float] = {}
    if suite.speaker is not None and speaker is not None:
        known = np.isin(speaker.astype(str), suite.speaker_classes)
        if known.any():
            pred = suite.speaker_classes[suite.speaker.predict(h_stats[known])]
            out["speaker_accuracy"] = float(np.mean(pred == speaker[known].astype(str)))
    if suite.emotion is not None and emotion is not None:
        known = np.isin(emotion.astype(str), suite.emotion_classes)
        if known.any():
            pred = suite.emotion_classes[suite.emotion.predict(h_stats[known])]
            out["emotion_accuracy"] = float(np.mean(pred == emotion[known].astype(str)))
            out["emotion_uar"] = macro_recall(emotion[known].astype(str), pred)
    return out
