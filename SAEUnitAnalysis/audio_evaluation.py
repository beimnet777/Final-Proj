"""Independent ASR and open-set speaker evaluation for rendered swaps."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .bundle import AnalysisBundle
from .causal import _two_way_cluster_draws
from .extraction import _read_audio
from .utils import AnalysisError, write_json


def _device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_wave(path: Path, target_rate: int = 16_000) -> np.ndarray:
    try:
        import soundfile as sf
        import librosa
    except ImportError as exc:
        raise AnalysisError("Audio evaluation requires soundfile and librosa.") from exc
    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if int(rate) != int(target_rate):
        audio = librosa.resample(audio, orig_sr=int(rate), target_sr=int(target_rate))
    return np.asarray(audio, dtype=np.float32)


def _normalise_transcript(text: str) -> str:
    text = re.sub(r"[^A-Z0-9' ]+", " ", str(text).upper())
    return " ".join(text.split())


class WhisperASR:
    def __init__(self, model_id: str, device: str) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise AnalysisError("ASR evaluation requires transformers.") from exc
        pipeline_device: Any = 0 if device.startswith("cuda") else device
        self.pipeline = pipeline(
            "automatic-speech-recognition", model=model_id, device=pipeline_device,
        )
        self.model_id = model_id

    def __call__(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        result = self.pipeline({"array": audio, "sampling_rate": int(sample_rate)})
        return _normalise_transcript(str(result["text"]))


class WavLMSpeakerVerifier:
    def __init__(self, model_id: str, device: str) -> None:
        try:
            from transformers import AutoFeatureExtractor, WavLMForXVector
        except ImportError as exc:
            raise AnalysisError("Speaker verification requires transformers with WavLM.") from exc
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.model_id = model_id

    @torch.inference_mode()
    def __call__(self, audio: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
        inputs = self.extractor(
            audio, sampling_rate=int(sample_rate), return_tensors="pt", padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        embedding = self.model(**inputs).embeddings[0]
        embedding = torch.nn.functional.normalize(embedding, dim=0)
        return embedding.float().cpu().numpy()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def _enrollment_utterances(
    bundle: AnalysisBundle,
    speaker: str,
    *,
    exclude: set[str],
    count: int,
) -> list[tuple[str, Path]]:
    rows = bundle.utterances[
        bundle.utterances["speaker_id"].astype(str) == str(speaker)
    ].sort_values("utterance_id", kind="stable")
    selected = []
    for _, row in rows.iterrows():
        utterance_id = str(row["utterance_id"])
        if utterance_id in exclude:
            continue
        selected.append((utterance_id, bundle.audio_path(row)))
        if len(selected) >= int(count):
            break
    if not selected:
        raise AnalysisError(f"No independent enrollment utterance remains for speaker {speaker}.")
    return selected


def _audio_contrasts(
    scores: pd.DataFrame,
    *,
    seed: int,
    repetitions: int = 1000,
) -> pd.DataFrame:
    metrics = [column for column in (
        "wer", "cer", "donor_similarity", "recipient_similarity",
        "donor_minus_recipient_similarity", "closer_to_donor",
    ) if column in scores.columns]
    baseline = scores[scores["mode"] == "sae_baseline"]
    rows = []
    rng = np.random.default_rng(seed)
    for mode, group in scores[scores["mode"] != "sae_baseline"].groupby("mode", sort=False):
        merged = group.merge(
            baseline[["pair", *metrics]], on="pair", suffixes=("", "__baseline"),
            validate="one_to_one",
        )
        for metric in metrics:
            valid = merged[metric].notna() & merged[f"{metric}__baseline"].notna()
            if not valid.any():
                continue
            effects = (
                merged.loc[valid, metric] - merged.loc[valid, f"{metric}__baseline"]
            ).to_numpy(dtype=float)
            recipients = merged.loc[valid, "recipient_speaker"].astype(str).to_numpy()
            donors = merged.loc[valid, "donor_speaker"].astype(str).to_numpy()
            draws = _two_way_cluster_draws(
                effects, recipients, donors, rng=rng, repetitions=repetitions,
            ) if len(np.unique(recipients)) > 1 and len(np.unique(donors)) > 1 else rng.choice(
                effects, size=(int(repetitions), len(effects)), replace=True,
            ).mean(axis=1)
            low, high = np.quantile(draws, [.025, .975])
            rows.append({
                "mode": str(mode), "metric": metric, "pairs": int(len(effects)),
                "baseline_mean": float(merged.loc[valid, f"{metric}__baseline"].mean()),
                "mode_mean": float(merged.loc[valid, metric].mean()),
                "paired_effect": float(effects.mean()),
                "ci95_low": float(low), "ci95_high": float(high),
                "interval_method": "paired_two_way_recipient_donor_speaker_bootstrap",
            })
    return pd.DataFrame(rows)


def reconstruction_gates(scores: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Predeclared reconstruction ceilings required before swap interpretation."""
    thresholds = {
        "oracle_mel_vocoder": {"max_wer_increase": .05, "max_recipient_similarity_drop": .10},
        "original_spear_bridge": {"max_wer_increase": .15, "max_recipient_similarity_drop": .20},
        "sae_baseline": {"max_wer_increase": .20, "max_recipient_similarity_drop": .25},
    }
    source = scores[scores["mode"] == "recipient_source"]
    rows: list[dict[str, Any]] = []
    for mode, limits in thresholds.items():
        current = scores[scores["mode"] == mode]
        merged = current.merge(source, on="pair", suffixes=("", "__source"), validate="one_to_one")
        checks: list[bool] = []
        row: dict[str, Any] = {"gate": mode, "pairs": int(len(merged))}
        if len(merged) and {"wer", "wer__source"} <= set(merged.columns):
            valid = merged["wer"].notna() & merged["wer__source"].notna()
            if valid.any():
                increase = (merged.loc[valid, "wer"] - merged.loc[valid, "wer__source"]).mean()
                row["wer_increase"] = float(increase)
                row["max_wer_increase"] = float(limits["max_wer_increase"])
                row["wer_pass"] = bool(increase <= limits["max_wer_increase"])
                checks.append(bool(row["wer_pass"]))
        if len(merged) and {
            "recipient_similarity", "recipient_similarity__source",
        } <= set(merged.columns):
            valid = merged["recipient_similarity"].notna() & merged["recipient_similarity__source"].notna()
            if valid.any():
                drop = (
                    merged.loc[valid, "recipient_similarity__source"]
                    - merged.loc[valid, "recipient_similarity"]
                ).mean()
                row["recipient_similarity_drop"] = float(drop)
                row["max_recipient_similarity_drop"] = float(
                    limits["max_recipient_similarity_drop"]
                )
                row["speaker_pass"] = bool(
                    drop <= limits["max_recipient_similarity_drop"]
                )
                checks.append(bool(row["speaker_pass"]))
        row["available_checks"] = int(len(checks))
        row["gate_pass"] = bool(checks and all(checks))
        rows.append(row)
    table = pd.DataFrame(rows)
    overall = bool(len(table) and table["gate_pass"].all())
    return table, overall


def _make_plots(output_dir: Path, scores: pd.DataFrame, contrasts: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    made = []
    if (
        {"wer", "donor_minus_recipient_similarity"} <= set(scores.columns)
        and scores["wer"].notna().any()
        and scores["donor_minus_recipient_similarity"].notna().any()
    ):
        summary = scores.groupby("mode", as_index=False).agg(
            wer=("wer", "mean"),
            donor_minus_recipient_similarity=("donor_minus_recipient_similarity", "mean"),
        ).dropna()
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
        ax.scatter(summary["wer"], summary["donor_minus_recipient_similarity"], s=65)
        for _, row in summary.iterrows():
            ax.annotate(str(row["mode"]).replace("_", " "), (row["wer"], row["donor_minus_recipient_similarity"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.axhline(0, color="black", lw=.8)
        ax.set(xlabel="word error rate (lower is better)",
               ylabel="donor minus recipient speaker cosine (higher is better)",
               title="Content–speaker conversion trade-off")
        fig.tight_layout()
        path = plots / "audio_conversion_tradeoff.png"
        fig.savefig(path, dpi=180); plt.close(fig); made.append(path)
    if len(contrasts):
        selected = contrasts[
            contrasts["mode"].isin([
                "P_from_donor", "L_from_donor", "matched_P_subset_from_donor",
                "matched_nonP_from_donor", "P_zero",
            ])
            & contrasts["metric"].isin(["wer", "donor_minus_recipient_similarity"])
        ].copy()
        if len(selected):
            selected["label"] = selected["mode"].str.replace("_", " ", regex=False)
            selected["label"] += " · " + selected["metric"].str.replace("_", " ", regex=False)
            selected = selected.iloc[::-1].reset_index(drop=True)
            y = np.arange(len(selected))
            value = selected["paired_effect"].to_numpy(float)
            fig, ax = plt.subplots(figsize=(9.5, max(4.6, .5 * len(selected))))
            ax.errorbar(
                value, y,
                xerr=np.vstack([value-selected["ci95_low"], selected["ci95_high"]-value]),
                fmt="o", capsize=3,
            )
            ax.axvline(0, color="black", lw=.8)
            ax.set(yticks=y, yticklabels=selected["label"],
                   xlabel="paired mode-minus-SAE-baseline effect",
                   title="Independent audio effects with 95% clustered intervals")
            fig.tight_layout()
            path = plots / "audio_conversion_paired_effects.png"
            fig.savefig(path, dpi=180); plt.close(fig); made.append(path)
    return made


def _evaluation_report(
    output_dir: Path,
    scores: pd.DataFrame,
    contrasts: pd.DataFrame,
    gates: pd.DataFrame,
    gates_passed: bool,
    plots: list[Path],
) -> Path:
    aggregations: dict[str, tuple[str, str]] = {"pairs": ("pair", "nunique")}
    for metric in (
        "wer", "cer", "donor_similarity", "recipient_similarity", "closer_to_donor",
    ):
        if metric in scores.columns and scores[metric].notna().any():
            aggregations[metric] = (metric, "mean")
    summary = scores.groupby("mode", as_index=False).agg(**aggregations)
    images = "".join(
        f"<img src='../plots/{html.escape(path.name)}' style='max-width:100%'>" for path in plots
    )
    verdict = "PASS" if gates_passed else "FAIL — do not interpret route-swap audio"
    verdict_color = "#d3f9d8" if gates_passed else "#ffe3e3"
    page = """<!doctype html><html><head><meta charset='utf-8'><title>Audio evaluation</title>
    <style>body{font-family:system-ui;margin:2rem;color:#162033}table{border-collapse:collapse;width:100%}
    th,td{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}.note{background:#eef4ff;padding:1rem;border-radius:8px}</style>
    </head><body><h1>Independent waveform evaluation</h1>
    <p class='note'>ASR and speaker embeddings come from pretrained models outside SAE training. Speaker scores use enrollment utterances excluded from each intervention pair. Route-swap claims remain conditional on the oracle, original-SPEAR and SAE reconstruction gates.</p>
    """ + f"<h2 style='background:{verdict_color};padding:1rem;border-radius:8px'>Reconstruction gates: {html.escape(verdict)}</h2>" \
        + gates.to_html(index=False, escape=True) \
        + "<p>Gate thresholds are fixed absolute WER increases of 0.05/0.15/0.20 and recipient-speaker cosine drops of 0.10/0.20/0.25 for oracle mel, original SPEAR and SAE reconstruction respectively. Available checks must all pass.</p>" \
        + "<h2>Mode summary</h2>" + summary.to_html(index=False, escape=True) + images
    if len(contrasts):
        page += "<h2>Paired effects</h2>" + contrasts.to_html(index=False, escape=True)
    page += "</body></html>"
    report = output_dir / "evaluation_report" / "index.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(page, encoding="utf-8")
    return report


def _attach_evaluation_link(audio_dir: Path) -> None:
    report = audio_dir / "report" / "index.html"
    if not report.exists():
        return
    marker_start = "<!-- independent-evaluation:start -->"
    marker_end = "<!-- independent-evaluation:end -->"
    section = (
        marker_start
        + "<section><h2>Independent numerical evaluation</h2>"
          "<p><a href='../evaluation_report/index.html'>Open the ASR and open-set "
          "speaker evaluation report</a>.</p></section>"
        + marker_end
    )
    text = report.read_text(encoding="utf-8")
    if marker_start in text and marker_end in text:
        before, rest = text.split(marker_start, 1)
        _, after = rest.split(marker_end, 1)
        text = before + section + after
    else:
        text = text.replace("</body>", section + "</body>")
    report.write_text(text, encoding="utf-8")


def evaluate_audio(
    audio_dir: Path,
    data_root: Path,
    *,
    device: str | None = None,
    asr_model_id: str | None = "openai/whisper-small.en",
    speaker_model_id: str | None = "microsoft/wavlm-base-plus-sv",
    enrollment_utterances: int = 3,
    seed: int = 42,
) -> Path:
    device = _device(device)
    audio_dir = audio_dir.resolve()
    manifest_path = audio_dir / "audio_manifest.csv"
    if not manifest_path.exists():
        raise AnalysisError(f"Missing rendered-audio manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    bundle = AnalysisBundle(data_root)
    asr = WhisperASR(asr_model_id, device) if asr_model_id else None
    speaker = WavLMSpeakerVerifier(speaker_model_id, device) if speaker_model_id else None
    embedding_cache: dict[str, np.ndarray] = {}

    def embedding(path: Path) -> np.ndarray:
        key = str(path.resolve())
        if key not in embedding_cache:
            if speaker is None:
                raise AnalysisError("Speaker embedding requested while speaker evaluation is disabled.")
            embedding_cache[key] = speaker(_load_wave(path))
        return embedding_cache[key]

    centroid_cache: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray, list[str]]] = {}

    def centroid(speaker_id: str, exclude: set[str]) -> tuple[np.ndarray, list[str]]:
        key = (str(speaker_id), tuple(sorted(exclude)))
        if key not in centroid_cache:
            references = _enrollment_utterances(
                bundle, speaker_id, exclude=exclude, count=enrollment_utterances,
            )
            vectors = [embedding(path) for _, path in references]
            value = np.mean(vectors, axis=0)
            value /= max(float(np.linalg.norm(value)), 1e-12)
            centroid_cache[key] = value, [utterance_id for utterance_id, _ in references]
        return centroid_cache[key]

    try:
        from jiwer import cer, wer
    except ImportError as exc:
        raise AnalysisError("Audio evaluation requires jiwer.") from exc
    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        path = Path(str(row["audio_path"]))
        audio = _load_wave(path)
        hypothesis = asr(audio) if asr is not None else ""
        reference = _normalise_transcript(str(row.get("transcript", "")))
        donor_enrollment: list[str] = []
        recipient_enrollment: list[str] = []
        donor_similarity = recipient_similarity = float("nan")
        if speaker is not None:
            exclude = {str(row["recipient"]), str(row["donor"])}
            donor_centroid, donor_enrollment = centroid(str(row["donor_speaker"]), exclude)
            recipient_centroid, recipient_enrollment = centroid(str(row["recipient_speaker"]), exclude)
            generated_embedding = embedding(path)
            donor_similarity = _cosine(generated_embedding, donor_centroid)
            recipient_similarity = _cosine(generated_embedding, recipient_centroid)
        rows.append({
            **row.to_dict(),
            "reference_transcript": reference,
            "asr_transcript": hypothesis,
            "wer": float(wer(reference, hypothesis)) if reference and asr is not None else float("nan"),
            "cer": float(cer(reference, hypothesis)) if reference and asr is not None else float("nan"),
            "donor_similarity": donor_similarity,
            "recipient_similarity": recipient_similarity,
            "donor_minus_recipient_similarity": donor_similarity - recipient_similarity,
            "closer_to_donor": (
                float(donor_similarity > recipient_similarity)
                if np.isfinite(donor_similarity) and np.isfinite(recipient_similarity)
                else float("nan")
            ),
            "donor_enrollment_utterances": "|".join(donor_enrollment),
            "recipient_enrollment_utterances": "|".join(recipient_enrollment),
        })
    scores = pd.DataFrame(rows)
    scores.to_csv(audio_dir / "audio_evaluation.csv", index=False)
    contrasts = _audio_contrasts(scores, seed=seed)
    contrasts.to_csv(audio_dir / "audio_evaluation_contrasts.csv", index=False)
    gates, gates_passed = reconstruction_gates(scores)
    gates.to_csv(audio_dir / "reconstruction_gates.csv", index=False)
    plots = _make_plots(audio_dir, scores, contrasts)
    report = _evaluation_report(audio_dir, scores, contrasts, gates, gates_passed, plots)
    _attach_evaluation_link(audio_dir)
    write_json(audio_dir / "evaluation_manifest.json", {
        "format": "spear_audio_swap_evaluation_v1",
        "asr_model_id": asr_model_id,
        "speaker_model_id": speaker_model_id,
        "speaker_protocol": "open_set_cosine_to_independent_enrollment_centroids",
        "enrollment_utterances": int(enrollment_utterances),
        "pair_sources_excluded_from_enrollment": True,
        "seed": int(seed),
        "rows": int(len(scores)),
        "pairs": int(scores["pair"].nunique()) if len(scores) else 0,
        "reconstruction_gates_passed": gates_passed,
        "route_swap_audio_interpretable": gates_passed,
        "report": str(report),
    })
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate rendered swaps with external models.")
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--asr-model-id", default="openai/whisper-small.en")
    parser.add_argument("--speaker-model-id", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-speaker", action="store_true")
    parser.add_argument("--enrollment-utterances", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        report = evaluate_audio(
            args.audio_dir, args.data, device=args.device,
            asr_model_id=None if args.skip_asr else args.asr_model_id,
            speaker_model_id=None if args.skip_speaker else args.speaker_model_id,
            enrollment_utterances=args.enrollment_utterances, seed=args.seed,
        )
    except AnalysisError as exc:
        raise SystemExit(f"[audio-evaluation] ERROR: {exc}") from exc
    print(f"[audio-evaluation] report: {report}")


if __name__ == "__main__":
    main()
