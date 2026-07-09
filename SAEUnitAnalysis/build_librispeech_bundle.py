from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


DEFAULT_SPLITS = ("train-clean-100", "dev-clean", "test-clean")
AUDIO_SUFFIXES = (".flac", ".FLAC", ".wav", ".WAV")


def _find_librispeech_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "LibriSpeech").exists():
        path = path / "LibriSpeech"
    if any((path / split).exists() for split in DEFAULT_SPLITS):
        return path
    for parent in [path, *path.parents]:
        candidate = parent / "LibriSpeech"
        if any((candidate / split).exists() for split in DEFAULT_SPLITS):
            return candidate
        if any((parent / split).exists() for split in DEFAULT_SPLITS):
            return parent
    raise SystemExit(
        f"Could not find a LibriSpeech root containing one of {DEFAULT_SPLITS} from: {path}"
    )


def _load_transcripts(split_dir: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for path in split_dir.rglob("*.trans.txt"):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 1:
                    transcripts[parts[0]] = ""
                else:
                    transcripts[parts[0]] = parts[1]
    return transcripts


def _find_audio(stem: Path) -> Path | None:
    for suffix in AUDIO_SUFFIXES:
        path = stem.with_suffix(suffix)
        if path.exists():
            return path
    return None


def _link_or_copy_audio(src: Path, dst: Path, *, copy_audio: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_audio:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _limit(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    chosen = rows[:]
    rng.shuffle(chosen)
    return sorted(chosen[:limit], key=lambda row: row["utterance_id"])


def build_bundle(
    librispeech_root: Path,
    output: Path,
    *,
    sample_rate: int = 16000,
    train_split: str = "train-clean-100",
    validation_split: str = "dev-clean",
    test_split: str = "test-clean",
    max_train: int = 2000,
    max_validation: int = 500,
    max_test: int = 500,
    seed: int = 42,
    copy_audio: bool = False,
) -> None:
    librispeech_root = _find_librispeech_root(librispeech_root)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    split_plan = [
        (train_split, "train", max_train),
        (validation_split, "val", max_validation),
        (test_split, "test", max_test),
    ]
    all_rows: list[dict[str, str]] = []
    split_counts: dict[str, int] = {}

    for split_name, logical_split, limit in split_plan:
        split_dir = librispeech_root / split_name
        if not split_dir.exists():
            raise SystemExit(f"LibriSpeech split not found: {split_dir}")
        transcripts = _load_transcripts(split_dir)
        rows: list[dict[str, str]] = []
        for utt_id, transcript in sorted(transcripts.items()):
            parts = utt_id.split("-")
            if len(parts) < 3:
                continue
            speaker_id, chapter_id = parts[0], parts[1]
            audio = _find_audio(split_dir / speaker_id / chapter_id / utt_id)
            if audio is None:
                raise SystemExit(f"No audio found for utterance {utt_id} under {split_dir}")
            rel_audio = Path("audio") / split_name / speaker_id / chapter_id / audio.name
            _link_or_copy_audio(audio, output / rel_audio, copy_audio=copy_audio)
            rows.append(
                {
                    "utterance_id": utt_id,
                    "audio_path": str(rel_audio),
                    "split": logical_split,
                    "transcript": transcript,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "source_split": split_name,
                }
            )
        rows = _limit(rows, limit, seed)
        split_counts[logical_split] = len(rows)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No LibriSpeech utterances were collected.")

    with (output / "utterances.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utterance_id",
                "audio_path",
                "split",
                "transcript",
                "speaker_id",
                "chapter_id",
                "source_split",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    dataset = {
        "schema_version": 1,
        "sample_rate": sample_rate,
        "manifest": "utterances.csv",
        "splits": {"train": "train", "validation": "val", "test": "test"},
        "batch_size": 4,
        "notes": (
            "LibriSpeech in-domain SAE analysis bundle. No phone alignments are "
            "included; phone-specific analyses are intentionally unavailable."
        ),
        "factors": [
            {
                "name": "speaker_id",
                "family": "paralinguistic",
                "level": "utterance",
                "type": "categorical",
                "source": "speaker_id",
            },
            {
                "name": "chapter_id",
                "family": "metadata",
                "level": "utterance",
                "type": "categorical",
                "source": "chapter_id",
            },
            {
                "name": "energy",
                "family": "paralinguistic",
                "level": "frame",
                "type": "continuous",
                "source": "computed:energy",
            },
            {
                "name": "voicing",
                "family": "paralinguistic",
                "level": "frame",
                "type": "continuous",
                "source": "computed:voicing",
            },
            {
                "name": "speaking_rate",
                "family": "paralinguistic",
                "level": "utterance",
                "type": "continuous",
                "source": "computed:speaking_rate",
            },
        ],
    }
    (output / "dataset.yaml").write_text(json.dumps(dataset, indent=2) + "\n")
    print(f"Wrote LibriSpeech SAE bundle: {output}")
    print(f"utterances={len(all_rows)} splits={split_counts}")
    print("phone_alignments=absent intentionally")
    print(f"dataset={output / 'dataset.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an in-domain SAEUnitAnalysis bundle from LibriSpeech audio "
            "and transcripts. This does not create phone alignments."
        )
    )
    parser.add_argument("--librispeech-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-rate", default=16000, type=int)
    parser.add_argument("--train-split", default="train-clean-100")
    parser.add_argument("--validation-split", default="dev-clean")
    parser.add_argument("--test-split", default="test-clean")
    parser.add_argument("--max-train", default=2000, type=int)
    parser.add_argument("--max-validation", default=500, type=int)
    parser.add_argument("--max-test", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="Copy audio into the bundle instead of creating symlinks.",
    )
    args = parser.parse_args()
    build_bundle(
        args.librispeech_root,
        args.output,
        sample_rate=args.sample_rate,
        train_split=args.train_split,
        validation_split=args.validation_split,
        test_split=args.test_split,
        max_train=args.max_train,
        max_validation=args.max_validation,
        max_test=args.max_test,
        seed=args.seed,
        copy_audio=args.copy_audio,
    )


if __name__ == "__main__":
    main()
