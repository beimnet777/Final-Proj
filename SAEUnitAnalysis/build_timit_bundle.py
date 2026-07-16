from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


AUDIO_SUFFIXES = (".WAV", ".wav", ".FLAC", ".flac")


def _find_timit_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "TRAIN").exists() or (path / "TEST").exists():
        return path
    for parent in [path, *path.parents]:
        if (parent / "TRAIN").exists() or (parent / "TEST").exists():
            return parent
    raise SystemExit(
        f"Could not find a TIMIT root containing TRAIN/ or TEST/ from: {path}"
    )


def _read_transcript(txt_path: Path) -> str:
    if not txt_path.exists():
        return ""
    text = txt_path.read_text(errors="ignore").strip()
    if not text:
        return ""
    parts = text.split(maxsplit=2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        return parts[2].strip()
    return text


def _find_audio(phn_path: Path) -> Path | None:
    stem = phn_path.with_suffix("")
    for suffix in AUDIO_SUFFIXES:
        candidate = stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _safe_rel(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def _link_or_copy_audio(src: Path, dst: Path, *, copy_audio: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_audio:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def build_bundle(
    timit_root: Path,
    output: Path,
    *,
    sample_rate: int = 16000,
    copy_audio: bool = False,
    max_utterances: int = 0,
    validation_fraction: float = 0.10,
    seed: int = 42,
) -> None:
    timit_root = _find_timit_root(timit_root)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    utterance_rows: list[dict[str, str]] = []
    alignment_rows: list[dict[str, str]] = []

    phn_files = sorted(
        list(timit_root.glob("TRAIN/*/*/*.PHN"))
        + list(timit_root.glob("TRAIN/*/*/*.phn"))
        + list(timit_root.glob("TEST/*/*/*.PHN"))
        + list(timit_root.glob("TEST/*/*/*.phn"))
    )
    if max_utterances > 0:
        phn_files = phn_files[:max_utterances]
    if not phn_files:
        raise SystemExit(f"No .PHN files found under {timit_root}")

    for phn_path in phn_files:
        audio_path = _find_audio(phn_path)
        if audio_path is None:
            raise SystemExit(f"No matching audio file found for {phn_path}")

        rel = _safe_rel(phn_path, timit_root)
        parts = rel.parts
        if len(parts) < 4:
            raise SystemExit(f"Unexpected TIMIT path layout: {phn_path}")
        split_raw, dialect, speaker = parts[0], parts[1], parts[2]
        split = "train" if split_raw.upper() == "TRAIN" else "test"
        if split == "test":
            split = "test"
        utt_stem = phn_path.stem
        utterance_id = "_".join(
            [split_raw.lower(), dialect.lower(), speaker.lower(), utt_stem.lower()]
        )

        audio_rel = Path("audio") / _safe_rel(audio_path, timit_root)
        _link_or_copy_audio(audio_path, output / audio_rel, copy_audio=copy_audio)

        txt_path = phn_path.with_suffix(".TXT")
        if not txt_path.exists():
            txt_path = phn_path.with_suffix(".txt")

        sex = speaker[:1].upper() if speaker else ""
        if sex not in {"F", "M"}:
            sex = ""

        utterance_rows.append(
            {
                "utterance_id": utterance_id,
                "audio_path": str(audio_rel),
                "split": split,
                "transcript": _read_transcript(txt_path),
                "speaker_id": speaker.lower(),
                "sex": sex,
                "dialect_region": dialect.upper(),
            }
        )

        with phn_path.open("r", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                start, end, phone = parts[0], parts[1], parts[2]
                alignment_rows.append(
                    {
                        "utterance_id": utterance_id,
                        "start_sec": f"{int(start) / sample_rate:.6f}",
                        "end_sec": f"{int(end) / sample_rate:.6f}",
                        "phone": phone,
                    }
                )

    # Reserve speaker-disjoint validation speakers from the official TRAIN set.
    # Never alias validation to the official TEST set.
    train_speakers = sorted({row["speaker_id"] for row in utterance_rows if row["split"] == "train"})
    validation_speakers: set[str] = set()
    if len(train_speakers) >= 2 and validation_fraction > 0:
        n_validation = min(
            len(train_speakers) - 1,
            max(1, int(round(len(train_speakers) * float(validation_fraction)))),
        )
        validation_speakers = set(random.Random(seed).sample(train_speakers, n_validation))
        for row in utterance_rows:
            if row["split"] == "train" and row["speaker_id"] in validation_speakers:
                row["split"] = "val"

    with (output / "utterances.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utterance_id",
                "audio_path",
                "split",
                "transcript",
                "speaker_id",
                "sex",
                "dialect_region",
            ],
        )
        writer.writeheader()
        writer.writerows(utterance_rows)

    with (output / "alignments.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["utterance_id", "start_sec", "end_sec", "phone"]
        )
        writer.writeheader()
        writer.writerows(alignment_rows)

    dataset = {
        "schema_version": 1,
        "sample_rate": sample_rate,
        "manifest": "utterances.csv",
        "alignments": "alignments.csv",
        "splits": {"train": "train", "validation": "val", "test": "test"},
        "factors": [
            {
                "name": "phone",
                "family": "linguistic",
                "level": "frame",
                "type": "categorical",
                "source": "alignment",
            },
            {
                "name": "speaker_id",
                "family": "paralinguistic",
                "level": "utterance",
                "type": "categorical",
                "source": "speaker_id",
            },
        ],
    }
    (output / "dataset.yaml").write_text(json.dumps(dataset, indent=2) + "\n")
    print(f"Wrote TIMIT SAE bundle: {output}")
    print(f"utterances={len(utterance_rows)} alignments={len(alignment_rows)}")
    print(f"dataset={output / 'dataset.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an SAEUnitAnalysis bundle from raw TIMIT TRAIN/TEST folders."
    )
    parser.add_argument("--timit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-rate", default=16000, type=int)
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="Copy audio into the bundle instead of creating symlinks.",
    )
    parser.add_argument(
        "--max-utterances",
        default=0,
        type=int,
        help="Optional debugging limit. 0 means all utterances.",
    )
    parser.add_argument("--validation-fraction", default=0.10, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    build_bundle(
        args.timit_root,
        args.output,
        sample_rate=args.sample_rate,
        copy_audio=args.copy_audio,
        max_utterances=args.max_utterances,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
