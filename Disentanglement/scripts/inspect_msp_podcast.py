#!/usr/bin/env python3
"""Inspect an MSP-Podcast release layout and label file.

This is intentionally read-only.  Run it on the cluster where the /rds dataset
path is mounted, for example:

    python scripts/inspect_msp_podcast.py /rds/project/.../MSP-Podcast-2.0
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


LABEL_CANDIDATES = (
    "labels_consensus.csv",
    "label_consensus.csv",
    "processed_labels.csv",
)
AUDIO_DIR_CANDIDATES = (
    "Audios",
    "Audio",
    "audios",
    "audio",
    "Wav",
    "Wavs",
    "wav",
    "wavs",
)
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a")


def _find_label_file(root: Path, override: str | None) -> Path:
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"label file not found: {path}")
        return path

    for name in LABEL_CANDIDATES:
        hits = list(root.rglob(name))
        if hits:
            return hits[0]

    csv_hits = list(root.rglob("*.csv"))
    if len(csv_hits) == 1:
        return csv_hits[0]
    raise FileNotFoundError(
        "Could not find an obvious label CSV. Use --label_csv explicitly. "
        f"CSV files found: {len(csv_hits)}"
    )


def _find_audio_root(root: Path, override: str | None) -> Path:
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"audio directory not found: {path}")
        return path

    for name in AUDIO_DIR_CANDIDATES:
        path = root / name
        if path.exists():
            return path
    return root


def _first_present(columns: list[str], names: tuple[str, ...]) -> str | None:
    lower_to_real = {c.lower(): c for c in columns}
    for name in names:
        hit = lower_to_real.get(name.lower())
        if hit is not None:
            return hit
    return None


def _resolve_audio(audio_root: Path, filename: str) -> Path | None:
    raw = Path(filename)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(audio_root / raw)
        candidates.append(audio_root / raw.name)
        if raw.suffix:
            candidates.append(audio_root / raw.with_suffix(raw.suffix.lower()).name)
        else:
            for suffix in AUDIO_SUFFIXES:
                candidates.append(audio_root / f"{filename}{suffix}")
                candidates.append(audio_root / f"{raw.name}{suffix}")
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="MSP-Podcast release root")
    parser.add_argument("--label_csv", default=None, help="Optional explicit label CSV path")
    parser.add_argument("--audio_root", default=None, help="Optional explicit audio directory")
    parser.add_argument("--check_audio", type=int, default=2000,
                        help="Number of rows to test for audio path resolution")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"MSP root not found: {root}")

    label_csv = _find_label_file(root, args.label_csv)
    audio_root = _find_audio_root(root, args.audio_root)

    with label_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    file_col = _first_present(columns, ("FileName", "Filename", "file", "path", "AudioFile"))
    split_col = _first_present(columns, ("Split_Set", "Split", "partition", "set"))
    emo_col = _first_present(columns, ("EmoClass", "Emotion", "emotion", "PrimaryEmotion"))
    speaker_col = _first_present(columns, ("SpeakerID", "SpkrID", "speaker_id", "Speaker", "speaker"))
    vad_cols = [
        c for c in columns
        if c.lower() in {"valence", "activation", "arousal", "dominance", "val", "act", "aro", "dom"}
    ]

    print("=== MSP-Podcast Inspection ===")
    print(f"root          : {root}")
    print(f"label_csv     : {label_csv}")
    print(f"audio_root    : {audio_root}")
    print(f"rows          : {len(rows)}")
    print(f"columns       : {', '.join(columns)}")
    print(f"file_col      : {file_col}")
    print(f"split_col     : {split_col}")
    print(f"emotion_col   : {emo_col}")
    print(f"speaker_col   : {speaker_col}")
    print(f"vad_cols      : {', '.join(vad_cols) if vad_cols else '<none detected>'}")

    if split_col:
        print("\nrows by split:")
        for split, count in sorted(Counter(r.get(split_col, "") for r in rows).items()):
            print(f"  {split or '<blank>'}: {count}")

    if emo_col:
        print("\nrows by emotion:")
        for emo, count in sorted(Counter(r.get(emo_col, "") for r in rows).items()):
            print(f"  {emo or '<blank>'}: {count}")

    if speaker_col:
        by_split: dict[str, set[str]] = defaultdict(set)
        missing = 0
        for row in rows:
            speaker = row.get(speaker_col, "")
            if not speaker:
                missing += 1
                continue
            split = row.get(split_col, "<all>") if split_col else "<all>"
            by_split[split].add(speaker)
        print("\nspeaker coverage:")
        print(f"  rows_missing_speaker: {missing}")
        for split, speakers in sorted(by_split.items()):
            print(f"  {split}: {len(speakers)} speakers")

    if file_col and args.check_audio > 0:
        checked = min(args.check_audio, len(rows))
        found = 0
        missing_examples = []
        for row in rows[:checked]:
            resolved = _resolve_audio(audio_root, row.get(file_col, ""))
            if resolved is not None:
                found += 1
            elif len(missing_examples) < 5:
                missing_examples.append(row.get(file_col, ""))
        print("\naudio path check:")
        print(f"  checked: {checked}")
        print(f"  found  : {found}")
        print(f"  missing: {checked - found}")
        if missing_examples:
            print("  missing examples:")
            for example in missing_examples:
                print(f"    {example}")


if __name__ == "__main__":
    main()
