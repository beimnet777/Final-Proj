#!/usr/bin/env python3
"""Download VoxCeleb1 wav files from HuggingFace and arrange them for the SID pipeline.

Source: https://huggingface.co/datasets/ProgramComputer/voxceleb

Usage (run on a login node, NOT as a SLURM job):
    python sid/download_data.py --out_dir /rds/user/${USER}/hpc-work/data/VoxCeleb1
    python sid/download_data.py --split dev  --out_dir /rds/user/bbg25/hpc-work/data/VoxCeleb1
    python sid/download_data.py --split test --out_dir /rds/user/bbg25/hpc-work/data/VoxCeleb1

After completion the directory looks exactly as sid_data.py expects:

    VoxCeleb1/
        dev/
            wav/
                id00001/
                    1zcIwhmdeo4/
                        00001.wav
                        …
                id00002/ …
        test/
            wav/
                id00001/ …
        vox1_meta.csv

The script uses HF's resume_download so it is safe to interrupt and re-run.
Only wav archives are downloaded (txt transcript zips are skipped).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID   = "ProgramComputer/voxceleb"
REPO_TYPE = "dataset"

# Only the wav zips + metadata — transcripts not needed for SID.
SPLIT_FILES = {
    "dev":  ["vox1/vox1_dev_wav.zip",  "vox1/vox1_meta.csv"],
    "test": ["vox1/vox1_test_wav.zip", "vox1/vox1_meta.csv"],
}


def _download(filename: str, cache_dir: Path, token: str | None) -> Path:
    print(f"\n[download] {filename}")
    local = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
    )
    print(f"  → {local}")
    return Path(local)


def _extract(zip_path: Path, out_dir: Path, split: str) -> None:
    """Extract zip into out_dir/split/.

    The VoxCeleb1 zips contain entries like wav/{speaker}/{video}/{utt}.wav
    with no split prefix, so we extract into out_dir/split/ to get:
        out_dir/dev/wav/...  or  out_dir/test/wav/...
    """
    dest = out_dir / split
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n[extract] {zip_path.name}  →  {dest}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for i, member in enumerate(members, 1):
            target = dest / member.filename
            if target.exists() and not member.is_dir():
                continue   # resume: skip already-extracted files
            zf.extract(member, dest)
            if i % 10_000 == 0:
                print(f"  {i}/{len(members)} files extracted …")
    print(f"  done ({len(members)} entries)")


def _verify(out_dir: Path, splits: list[str]) -> None:
    """Quick sanity check that the expected layout is in place."""
    ok = True
    for split in splits:
        wav_root = out_dir / split / "wav"
        if not wav_root.is_dir():
            print(f"[WARN] expected {wav_root} — not found")
            ok = False
            continue
        n_spk = sum(1 for p in wav_root.iterdir() if p.is_dir())
        if n_spk == 0:
            print(f"[WARN] {wav_root} is empty")
            ok = False
        else:
            print(f"[OK]   {wav_root}  ({n_spk} speaker dirs)")
    if ok:
        print("\nVoxCeleb1 layout looks correct. Ready to submit SID training jobs.")
    else:
        print("\nSome checks failed — inspect the warnings above.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split",   choices=["dev", "test", "all"], default="all",
                   help="Which split(s) to download (default: all).")
    p.add_argument("--out_dir", default="/rds/user/bbg25/hpc-work/data/VoxCeleb1",
                   help="Destination VoxCeleb1 root (passed as --voxceleb1_root to sid_run.py).")
    p.add_argument("--token",   default=None,
                   help="HuggingFace token — not usually required for this public dataset.")
    p.add_argument("--no_extract", action="store_true",
                   help="Download zip files only, do not extract.")
    p.add_argument("--no_download", action="store_true",
                   help="Skip download; extract zips already present in out_dir/.hf_cache. "
                        "Use this on compute nodes that lack internet after running "
                        "--no_extract on a login node first.")
    args = p.parse_args()

    out_dir  = Path(args.out_dir).expanduser().resolve()
    # Keep downloaded zips in a sub-cache so they don't clutter the wav layout.
    cache_dir = out_dir / ".hf_cache"
    out_dir.mkdir(parents=True,   exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = ["dev", "test"] if args.split == "all" else [args.split]

    # Collect unique filenames (vox1_meta.csv appears in both splits).
    filenames: list[str] = []
    seen: set[str] = set()
    for split in splits:
        for fn in SPLIT_FILES[split]:
            if fn not in seen:
                filenames.append(fn)
                seen.add(fn)

    downloaded: list[Path] = []
    if args.no_download:
        # Zips must already be present in cache_dir from a prior --no_extract run.
        for fn in filenames:
            p_ = cache_dir / fn
            if not p_.exists():
                raise FileNotFoundError(
                    f"Expected cached zip not found: {p_}\n"
                    f"Run first on a login node: "
                    f"python sid/download_data.py --no_extract --out_dir {out_dir}"
                )
            downloaded.append(p_)
        print(f"Using {len(downloaded)} cached zip(s) from {cache_dir}")
    else:
        for fn in filenames:
            path = _download(fn, cache_dir, args.token)
            downloaded.append(path)

    if not args.no_extract:
        for path in downloaded:
            if path.suffix == ".zip":
                # Infer split from filename: vox1_dev_wav.zip → "dev"
                split = "dev" if "_dev_" in path.name else "test"
                _extract(path, out_dir, split)
        # Copy metadata to root for convenience.
        meta_src = cache_dir / "vox1" / "vox1_meta.csv"
        if meta_src.exists():
            import shutil
            shutil.copy2(meta_src, out_dir / "vox1_meta.csv")

    _verify(out_dir, splits)
    print(f"\nOutput directory: {out_dir}")


if __name__ == "__main__":
    main()
