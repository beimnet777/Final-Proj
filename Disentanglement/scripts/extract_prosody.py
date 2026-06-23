"""Offline per-frame F0 + log-energy extraction with pyworld.

Walks a corpus tree, finds every audio file, computes per-frame log-F0 and
log-energy, caches as <utt_id>.prosody.npy alongside the audio.

Idempotent: skips any file whose .prosody.npy already exists.

Targets v1 dual-invariance corpora: LibriSpeech, CMU_ARCTIC, ESD, VCTK.
Frame period matches WORLD default (5 ms) so SR 16 kHz audio yields one frame
per 80 samples — the prosody-head consumer should resample to its own frame
rate downstream rather than fight WORLD's grid here.

Usage:
    python Disentanglement/scripts/extract_prosody.py --data_root <DATA_ROOT>
        [--corpora LibriSpeech CMU_ARCTIC ESD VCTK]
        [--workers 8]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import pyworld as pw
except ImportError:
    print("pyworld not installed — pip install pyworld", file=sys.stderr); sys.exit(2)

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
_FRAME_PERIOD_MS = 5.0


def _extract_one(audio_path: Path) -> tuple[Path, str]:
    out_path = audio_path.with_suffix(".prosody.npy")
    if out_path.exists():
        return out_path, "skip"
    try:
        wav, sr = sf.read(str(audio_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        if len(wav) < 2048:
            return out_path, "too_short"
        x = wav.astype(np.float64)
        f0, t = pw.dio(x, sr, frame_period=_FRAME_PERIOD_MS)
        f0 = pw.stonemask(x, f0, t, sr)                                # (F,)
        # Per-frame log-energy over a hop matching pyworld's frame period.
        hop = int(sr * _FRAME_PERIOD_MS / 1000.0)                      # 80 @ 16 kHz
        n_frames = len(f0)
        # Pad signal to cover n_frames hops
        needed = n_frames * hop + hop
        pad = max(0, needed - len(x))
        xp = np.pad(x, (0, pad), mode="constant")
        frames = np.lib.stride_tricks.sliding_window_view(xp, hop)[:n_frames * hop:hop]
        energy = np.maximum((frames ** 2).mean(axis=-1), 1e-10)
        # Mask: 1 where voiced (f0 > 0), 0 otherwise.  Consumer can decide how
        # to treat unvoiced frames; we store log_f0 with -1e3 sentinel there.
        voiced = (f0 > 0).astype(np.float32)
        log_f0 = np.where(voiced > 0, np.log(np.maximum(f0, 1e-3)), -1e3).astype(np.float32)
        log_energy = np.log(energy).astype(np.float32)
        np.save(out_path, np.stack([log_f0, log_energy, voiced], axis=-1))   # (F, 3)
        return out_path, "ok"
    except Exception as ex:
        return out_path, f"err:{type(ex).__name__}:{ex}"


def _gather_audio(data_root: Path, corpora: list[str]) -> list[Path]:
    paths: list[Path] = []
    for corpus in corpora:
        croot = data_root / corpus
        if not croot.exists():
            print(f"  [warn] {corpus}: not found at {croot}", file=sys.stderr)
            continue
        for p in croot.rglob("*"):
            if p.suffix.lower() in _AUDIO_EXTS:
                paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=Path)
    ap.add_argument("--corpora", nargs="+",
                    default=["LibriSpeech", "CMU_ARCTIC", "ESD", "VCTK"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.data_root.exists():
        print(f"data_root not found: {args.data_root}", file=sys.stderr); return 2

    audio = _gather_audio(args.data_root, args.corpora)
    print(f"discovered {len(audio)} audio files across {args.corpora}")

    ok = skip = err = tooshort = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_extract_one, p) for p in audio]
        for i, fut in enumerate(as_completed(futs), 1):
            _, status = fut.result()
            if status == "ok":         ok += 1
            elif status == "skip":     skip += 1
            elif status == "too_short": tooshort += 1
            else:                       err += 1
            if i % 500 == 0:
                print(f"  [{i}/{len(audio)}]  ok={ok}  skip={skip}  short={tooshort}  err={err}")
    print(f"done: ok={ok}  skip={skip}  short={tooshort}  err={err}  total={len(audio)}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
