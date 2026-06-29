"""Prepare and verify local-data archives used by the Colab notebook.

Licensed MSP material is only copied into the requested archive; it is never
downloaded or placed in the repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_metadata(root: Path, *, dataset: str, profile: str, files: list[Path], extra: dict) -> None:
    payload = {
        "schema_version": 1, "dataset": dataset, "profile": profile,
        "files": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                   "sha256": _sha(p)} for p in sorted(files)],
        **extra,
    }
    (root / "bundle_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def _archive(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as tf:
        for path in sorted(root.rglob("*")):
            tf.add(path, arcname=path.relative_to(root), recursive=False)


def _balanced_msp(rows: list[dict], seed: int, speakers: int = 50,
                  cap: int = 40) -> list[dict]:
    rng = random.Random(seed)
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for row in rows: by_spk[row["speaker_idx"]].append(row)
    ranked = sorted(by_spk, key=lambda s: (
        len({r["emotion"] for r in by_spk[s]}), len(by_spk[s]), s), reverse=True)[:speakers]
    selected = []
    for new_idx, speaker in enumerate(ranked):
        buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in by_spk[speaker]: buckets[(row["split"], row["emotion"])].append(row)
        for values in buckets.values(): rng.shuffle(values)
        keys = sorted(buckets); i = 0; kept = []
        while len(kept) < min(cap, len(by_spk[speaker])):
            progressed = False
            for key in keys:
                if i < len(buckets[key]):
                    copy = dict(buckets[key][i]); copy["speaker_idx"] = str(new_idx)
                    kept.append(copy); progressed = True
                    if len(kept) >= cap: break
            if not progressed: break
            i += 1
        selected.extend(kept)
    return selected


def prepare_msp(a) -> None:
    with a.manifest.open(newline="") as f:
        reader = csv.DictReader(f); fields = reader.fieldnames or []; rows = list(reader)
    if a.profile == "pilot": rows = _balanced_msp(rows, a.seed)
    with tempfile.TemporaryDirectory(prefix="msp-colab-") as td:
        root = Path(td)
        with (root / "manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        copied = [root / "manifest.csv"]
        for row in rows:
            src = a.audio_root / row["wav"]
            if not src.exists(): raise FileNotFoundError(src)
            dst = root / row["wav"]; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst); copied.append(dst)
        wanted = {Path(r["FileName"]).with_suffix(".txt").name for r in rows}
        tdir = root / "Transcripts"; tdir.mkdir()
        if a.transcripts.suffix.lower() == ".zip":
            with zipfile.ZipFile(a.transcripts) as zf:
                members = {Path(n).name: n for n in zf.namelist() if n.endswith(".txt")}
                for name in sorted(wanted):
                    if name not in members: raise FileNotFoundError(f"transcript {name} missing")
                    dst = tdir / name; dst.write_bytes(zf.read(members[name])); copied.append(dst)
        else:
            indexed = {p.name: p for p in a.transcripts.rglob("*.txt")}
            for name in sorted(wanted):
                if name not in indexed: raise FileNotFoundError(f"transcript {name} missing")
                dst = tdir / name; shutil.copy2(indexed[name], dst); copied.append(dst)
        _write_metadata(root, dataset="msp", profile=a.profile, files=copied,
                        extra={"rows": len(rows), "seed": a.seed})
        _archive(root, a.output)


def _libri_files(root: Path, profile: str, seed: int) -> list[Path]:
    files = sorted((root / "train-clean-100").rglob("*.flac"))
    if profile == "full": return files
    rng = random.Random(seed)
    by_speaker: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in files:
        by_speaker[path.parent.parent.name][path.parent.name].append(path)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    chosen: list[Path] = []
    for speaker in speakers[:32]:
        chapters = list(by_speaker[speaker].values()); rng.shuffle(chapters)
        for chapter in chapters:
            chosen.extend(chapter)
            if sum(p.parent.parent.name == speaker for p in chosen) >= 64: break
    return chosen


def prepare_librispeech(a) -> None:
    source = a.librispeech_root / "LibriSpeech" if (a.librispeech_root / "LibriSpeech").exists() else a.librispeech_root
    selected = _libri_files(source, a.profile, a.seed)
    if not selected: raise FileNotFoundError(f"no FLAC files under {source / 'train-clean-100'}")
    with tempfile.TemporaryDirectory(prefix="libri-colab-") as td:
        root = Path(td); copied = []
        for audio in selected:
            rel = audio.relative_to(source); dst = root / "LibriSpeech" / rel
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(audio, dst); copied.append(dst)
            transcript = next(audio.parent.glob("*.trans.txt"), None)
            if transcript is not None:
                tdst = dst.parent / transcript.name
                if not tdst.exists(): shutil.copy2(transcript, tdst); copied.append(tdst)
        if a.profile == "full":
            for split in ("dev-clean", "test-clean"):
                if (source / split).exists():
                    shutil.copytree(source / split, root / "LibriSpeech" / split)
                    copied.extend(p for p in (root / "LibriSpeech" / split).rglob("*") if p.is_file())
        if a.lexicon:
            shutil.copy2(a.lexicon, root / "librispeech-lexicon.txt")
            copied.append(root / "librispeech-lexicon.txt")
        _write_metadata(root, dataset="librispeech", profile=a.profile, files=copied,
                        extra={"utterances": len(selected), "seed": a.seed})
        _archive(root, a.output)


def verify(a) -> None:
    a.extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.archive, "r:gz") as tf:
        root = a.extract_to.resolve()
        for member in tf.getmembers():
            target = (root / member.name).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
        tf.extractall(a.extract_to)
    manifest = json.loads((a.extract_to / "bundle_manifest.json").read_text())
    for item in manifest["files"]:
        path = a.extract_to / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
            raise ValueError(f"bundle verification failed for {item['path']}")
    print(json.dumps({"verified": len(manifest["files"]), "dataset": manifest["dataset"],
                      "profile": manifest["profile"]}, indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("prepare-msp"); m.set_defaults(func=prepare_msp)
    m.add_argument("--manifest", type=Path, required=True); m.add_argument("--audio-root", type=Path, required=True)
    m.add_argument("--transcripts", type=Path, required=True); m.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    m.add_argument("--seed", type=int, default=42); m.add_argument("--output", type=Path, required=True)
    l = sub.add_parser("prepare-librispeech"); l.set_defaults(func=prepare_librispeech)
    l.add_argument("--librispeech-root", type=Path, required=True); l.add_argument("--lexicon", type=Path)
    l.add_argument("--profile", choices=("pilot", "full"), default="pilot"); l.add_argument("--seed", type=int, default=42)
    l.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify"); v.set_defaults(func=verify)
    v.add_argument("--archive", type=Path, required=True); v.add_argument("--extract-to", type=Path, required=True)
    a = p.parse_args(argv); a.func(a)


if __name__ == "__main__": main()
