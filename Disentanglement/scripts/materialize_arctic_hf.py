"""Materialize the HF MikhailT/cmu-arctic snapshot into the festvox layout
that Disentanglement.data.parallel_datasets.ARCTICIndex expects:

    <out_root>/cmu_us_<spk>_arctic/
        wav/<file>.wav                 # 16 kHz mono int16
        etc/txt.done.data              # ( <file> "<transcript>" )

Reads parquet shards directly with pyarrow + soundfile (no `datasets` library
round-trip needed, and works whether or not the HF cache exists).

Idempotent: per-speaker, skips work if the wav dir already contains the
expected number of files.

Usage:
    python Disentanglement/scripts/materialize_arctic_hf.py \
        --snapshot Probing/data/CMU_ARCTIC_hf \
        --out      Probing/data/CMU_ARCTIC \
        --speakers awb,bdl,clb,jmk,ksp,rms,slt
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

TARGET_SR = 16_000


def materialize_speaker(parquet_path: Path, spk: str, out_root: Path) -> tuple[int, int]:
    spk_dir = out_root / f"cmu_us_{spk}_arctic"
    wav_dir = spk_dir / "wav"
    etc_dir = spk_dir / "etc"
    wav_dir.mkdir(parents=True, exist_ok=True)
    etc_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(parquet_path)
    files = table.column("file").to_pylist()
    texts = table.column("text").to_pylist()
    audios = table.column("audio").to_pylist()

    expected = len(files)
    existing = sum(1 for f in files if (wav_dir / f"{f}.wav").exists())
    if existing == expected and (etc_dir / "txt.done.data").exists():
        return (expected, 0)

    written = 0
    txt_lines = []
    for fname, text, aud in zip(files, texts, audios):
        # HF Audio column: {"bytes": <encoded bytes or None>, "path": <str or None>}
        # The bytes are the original wav file contents.
        wav_out = wav_dir / f"{fname}.wav"
        txt_lines.append(f'( {fname} "{text}" )')
        if wav_out.exists():
            continue
        if aud is None or aud.get("bytes") is None:
            raise RuntimeError(f"{spk}/{fname}: parquet row has no audio bytes")
        arr, sr = sf.read(io.BytesIO(aud["bytes"]))
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        if sr != TARGET_SR:
            import librosa
            arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        # Write as int16 PCM to match festvox releases.
        arr = np.clip(arr, -1.0, 1.0)
        sf.write(str(wav_out), arr.astype(np.float32), sr, subtype="PCM_16")
        written += 1

    (etc_dir / "txt.done.data").write_text("\n".join(txt_lines) + "\n")
    return (expected, written)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, type=Path,
                   help="path to CMU_ARCTIC_hf snapshot (contains data/*.parquet)")
    p.add_argument("--out", required=True, type=Path,
                   help="output root; festvox-style cmu_us_<spk>_arctic/ dirs go here")
    p.add_argument("--speakers", default="awb,bdl,clb,jmk,ksp,rms,slt",
                   help="comma-separated speaker tags to materialize (default: standard 7)")
    args = p.parse_args()

    data_dir = args.snapshot / "data"
    if not data_dir.is_dir():
        sys.exit(f"ERROR: no data/ dir under {args.snapshot}")

    args.out.mkdir(parents=True, exist_ok=True)
    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]

    print(f"snapshot : {args.snapshot}")
    print(f"out_root : {args.out}")
    print(f"speakers : {speakers}\n")

    total_expected = 0
    total_written = 0
    for spk in speakers:
        matches = sorted(data_dir.glob(f"{spk}-*.parquet"))
        if not matches:
            print(f"  [warn] {spk}: no parquet shard found, skipping")
            continue
        if len(matches) > 1:
            print(f"  [warn] {spk}: {len(matches)} shards, using all in order")
        n_exp = 0
        n_wrote = 0
        for shard in matches:
            e, w = materialize_speaker(shard, spk, args.out)
            n_exp += e
            n_wrote += w
        print(f"  [done] {spk}: {n_exp} utts ({n_wrote} newly written)")
        total_expected += n_exp
        total_written += n_wrote

    print(f"\nTotal: {total_expected} utterances across {len(speakers)} speakers "
          f"({total_written} newly written this run).")
    print(f"Layout: {args.out}/cmu_us_<spk>_arctic/{{wav,etc/txt.done.data}}")


if __name__ == "__main__":
    main()
