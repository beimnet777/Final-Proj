from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

from .bundle import AnalysisBundle


_SPACE_RE = re.compile(r"\s+")


def _clean_transcript(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("{", "").replace("}", "")
    text = text.replace("[", "").replace("]", "")
    text = _SPACE_RE.sub(" ", text)
    return text


def _link_or_copy(src: Path, dst: Path, *, copy_audio: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_audio:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def prepare_corpus(
    bundle_root: Path,
    output: Path,
    *,
    split: str = "all",
    copy_audio: bool = False,
) -> None:
    bundle = AnalysisBundle(bundle_root)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if split == "all":
        rows = bundle.utterances.copy()
    else:
        rows = bundle.utterances[bundle.utterances["split"].astype(str) == split].copy()
    if rows.empty:
        raise SystemExit(f"No utterances selected from split={split!r}.")

    map_rows: list[dict[str, str]] = []
    for _, row in rows.sort_values("utterance_id").iterrows():
        utt_id = str(row["utterance_id"])
        transcript = _clean_transcript(str(row.get("transcript", "")))
        if not transcript:
            continue
        audio = bundle.audio_path(row)
        suffix = audio.suffix.lower() or ".flac"
        # MFA infers speaker identity from the corpus directory layout.  Keep a
        # speaker subdirectory when speaker_id is available; otherwise MFA sees
        # the whole corpus as one speaker and disables useful parallelism /
        # speaker adaptation.  The utterance map preserves the original ID, so
        # importing TextGrids remains stable.
        speaker = str(row.get("speaker_id", "")).strip()
        speaker_dir = Path("audio") / speaker if speaker else Path("audio")
        audio_rel = speaker_dir / f"{utt_id}{suffix}"
        lab_rel = speaker_dir / f"{utt_id}.lab"
        _link_or_copy(audio, output / audio_rel, copy_audio=copy_audio)
        (output / lab_rel).write_text(transcript + "\n", encoding="utf-8")
        map_rows.append(
            {
                "utterance_id": utt_id,
                "corpus_stem": utt_id,
                "audio_path": str(audio_rel),
                "lab_path": str(lab_rel),
                "split": str(row["split"]),
                "speaker_id": speaker,
                "transcript": transcript,
            }
        )

    if not map_rows:
        raise SystemExit("No non-empty transcripts were available for MFA.")

    with (output / "mfa_utterance_map.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utterance_id",
                "corpus_stem",
                "audio_path",
                "lab_path",
                "split",
                "speaker_id",
                "transcript",
            ],
        )
        writer.writeheader()
        writer.writerows(map_rows)

    print(f"Wrote MFA corpus: {output}")
    print(f"utterances={len(map_rows)}")
    print(f"map={output / 'mfa_utterance_map.csv'}")
    print("Run MFA on this directory, then import the TextGrids with:")
    print("python -m SAEUnitAnalysis.import_mfa_alignments --bundle BUNDLE --mfa-output MFA_OUTPUT --utterance-map MAP")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an MFA corpus from an existing LibriSpeech SAEUnitAnalysis "
            "bundle. The output contains one audio file and one .lab transcript "
            "per utterance, plus mfa_utterance_map.csv."
        )
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--split",
        default="all",
        help="Manifest split to export, or 'all'. Typical values: train, val, test.",
    )
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="Copy audio into the MFA corpus instead of symlinking.",
    )
    args = parser.parse_args()
    prepare_corpus(args.bundle, args.output, split=args.split, copy_audio=args.copy_audio)


if __name__ == "__main__":
    main()
