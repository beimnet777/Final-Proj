from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import AnalysisError, write_json


DEFAULT_CONDITIONS = (
    "sae_baseline",
    "P_from_donor",
    "matched_P_subset_from_donor",
    "matched_nonP_from_donor",
)


def _blind_code(seed: int, *parts: object) -> str:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:14]


def _copy_blind(source: Path, output_audio: Path, code: str) -> Path:
    destination = output_audio / f"clip_{code}.wav"
    if not destination.exists():
        shutil.copy2(source, destination)
    return destination


def prepare_study(
    audio_result: Path,
    output: Path,
    *,
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    seed: int = 42,
    max_pairs: int = 24,
) -> Path:
    audio_result = audio_result.resolve()
    manifest_path = audio_result / "audio_manifest.csv"
    if not manifest_path.exists():
        raise AnalysisError(f"Missing audio manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = {
        "pair", "mode", "audio_path", "recipient_speaker", "donor_speaker",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise AnalysisError(f"Audio manifest is missing columns: {sorted(missing)}")
    available = set(manifest["mode"].astype(str))
    required_modes = set(conditions) | {"recipient_source", "donor_source"}
    missing_modes = required_modes - available
    if missing_modes:
        raise AnalysisError(
            "Listening study requires rendered modes: " + ", ".join(sorted(missing_modes))
        )
    pairs = sorted(manifest["pair"].unique().tolist())[:max(0, int(max_pairs))]
    if not pairs:
        raise AnalysisError("No rendered pairs are available for the listening study.")

    output = output.resolve()
    output_audio = output / "audio"
    output_audio.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    public_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []

    for pair in pairs:
        pair_rows = manifest[manifest["pair"] == pair].set_index("mode")
        recipient_source = Path(str(pair_rows.loc["recipient_source", "audio_path"]))
        donor_source = Path(str(pair_rows.loc["donor_source", "audio_path"]))
        for condition in conditions:
            candidate_source = Path(str(pair_rows.loc[condition, "audio_path"]))
            for source in (recipient_source, donor_source, candidate_source):
                if not source.exists():
                    raise AnalysisError(f"Rendered audio is missing: {source}")

            naturalness_id = _blind_code(seed, "naturalness", pair, condition)
            candidate_code = _blind_code(seed, "candidate", pair, condition)
            candidate_path = _copy_blind(
                candidate_source, output_audio, candidate_code,
            ).relative_to(output)
            public_rows.append({
                "trial_id": naturalness_id,
                "task": "naturalness",
                "candidate_audio": str(candidate_path),
                "reference_A_audio": "",
                "reference_B_audio": "",
                "prompt": "Rate naturalness from 1 (very unnatural) to 5 (completely natural).",
            })
            key_rows.append({
                "trial_id": naturalness_id,
                "task": "naturalness",
                "pair": int(pair),
                "condition": condition,
                "donor_reference_slot": "",
                "recipient_speaker": str(pair_rows.iloc[0]["recipient_speaker"]),
                "donor_speaker": str(pair_rows.iloc[0]["donor_speaker"]),
                "original_candidate_audio": str(candidate_source.resolve()),
            })

            identity_id = _blind_code(seed, "identity", pair, condition)
            donor_is_a = bool(rng.integers(2))
            reference_a_source = donor_source if donor_is_a else recipient_source
            reference_b_source = recipient_source if donor_is_a else donor_source
            reference_a_code = _blind_code(seed, "reference_A", pair, condition, donor_is_a)
            reference_b_code = _blind_code(seed, "reference_B", pair, condition, donor_is_a)
            reference_a_path = _copy_blind(
                reference_a_source, output_audio, reference_a_code,
            ).relative_to(output)
            reference_b_path = _copy_blind(
                reference_b_source, output_audio, reference_b_code,
            ).relative_to(output)
            public_rows.append({
                "trial_id": identity_id,
                "task": "speaker_identity_abx",
                "candidate_audio": str(candidate_path),
                "reference_A_audio": str(reference_a_path),
                "reference_B_audio": str(reference_b_path),
                "prompt": "Which reference speaker sounds more like the candidate: A, B, or neither?",
            })
            key_rows.append({
                "trial_id": identity_id,
                "task": "speaker_identity_abx",
                "pair": int(pair),
                "condition": condition,
                "donor_reference_slot": "A" if donor_is_a else "B",
                "recipient_speaker": str(pair_rows.iloc[0]["recipient_speaker"]),
                "donor_speaker": str(pair_rows.iloc[0]["donor_speaker"]),
                "original_candidate_audio": str(candidate_source.resolve()),
            })

    public = pd.DataFrame(public_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    public.insert(0, "trial_order", np.arange(1, len(public) + 1))
    key = pd.DataFrame(key_rows)
    public.to_csv(output / "listening_trials.csv", index=False)
    key.to_csv(output / "PRIVATE_answer_key.csv", index=False)
    write_json(output / "study_manifest.json", {
        "format": "sae_blinded_listening_study_v1",
        "source_audio_result": str(audio_result),
        "seed": int(seed),
        "pairs": int(len(pairs)),
        "conditions": list(conditions),
        "naturalness_trials": int((public["task"] == "naturalness").sum()),
        "speaker_identity_trials": int((public["task"] == "speaker_identity_abx").sum()),
        "blinding": "neutral copied filenames; condition mapping stored only in PRIVATE_answer_key.csv",
        "data_collection": "not performed by this command",
        "ethics": "obtain the required institutional approval/consent before recruiting participants",
    })

    trial_cards = []
    for _, trial in public.iterrows():
        candidate = html.escape(str(trial["candidate_audio"]))
        if trial["task"] == "naturalness":
            response = "".join(
                f"<label><input type='radio' name='{trial.trial_id}' value='{score}'> {score}</label>"
                for score in range(1, 6)
            )
            references = ""
        else:
            reference_a = html.escape(str(trial["reference_A_audio"]))
            reference_b = html.escape(str(trial["reference_B_audio"]))
            references = (
                f"<div class='refs'><span>Reference A<audio controls preload='none' src='../{reference_a}'></audio></span>"
                f"<span>Reference B<audio controls preload='none' src='../{reference_b}'></audio></span></div>"
            )
            response = "".join(
                f"<label><input type='radio' name='{trial.trial_id}' value='{choice}'> {choice}</label>"
                for choice in ("A", "B", "neither")
            )
        trial_cards.append(
            f"<section class='trial' data-trial='{html.escape(str(trial.trial_id))}' "
            f"data-task='{html.escape(str(trial.task))}'><h3>Trial {int(trial.trial_order)}</h3>"
            f"<p>{html.escape(str(trial.prompt))}</p>{references}"
            f"<span>Candidate<audio controls preload='none' src='../{candidate}'></audio></span>"
            f"<div class='response'>{response}</div></section>"
        )
    page = """<!doctype html><html><head><meta charset='utf-8'><style>
body{font-family:Inter,system-ui,sans-serif;background:#f5f7fb;color:#172033;margin:0;padding:32px;max-width:1000px}
.trial{background:#fff;border-radius:12px;padding:18px;margin:16px 0;box-shadow:0 2px 10px #1d2b4b18}.refs{display:flex;gap:28px;flex-wrap:wrap}audio{display:block;margin:7px 0 12px}.response label{margin-right:18px}button{padding:10px 16px;font-weight:650}</style><title>Blinded listening study</title></head><body>
<h1>Blinded listening study</h1><p>Use headphones in a quiet room. Listen as often as needed. Do not inspect filenames or page source. Complete every trial before downloading responses.</p>
""" + "".join(trial_cards) + """
<button onclick='downloadResponses()'>Download responses</button>
<script>function downloadResponses(){const rows=[['trial_id','task','response']];document.querySelectorAll('.trial').forEach(t=>{const x=t.querySelector('input:checked');rows.push([t.dataset.trial,t.dataset.task,x?x.value:'']);});const csv=rows.map(r=>r.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(',')).join('\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='responses.csv';a.click();}</script></body></html>"""
    report = output / "report"
    report.mkdir(exist_ok=True)
    page_path = report / "index.html"
    page_path.write_text(page, encoding="utf-8")
    return page_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a blinded naturalness and speaker-identity listening-study pack."
    )
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    conditions = tuple(args.condition) if args.condition else DEFAULT_CONDITIONS
    output = args.output_dir or args.audio_dir / "listening_study"
    page = prepare_study(
        args.audio_dir, output, conditions=conditions,
        seed=args.seed, max_pairs=args.max_pairs,
    )
    print(page)


if __name__ == "__main__":
    main()
