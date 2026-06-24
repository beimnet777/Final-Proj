"""ARCTIC SID probe data — random utterance split per speaker.

Used for the matched-distribution diagnostic: probe v1's z_L against the same
18-speaker pool the invariance objective was trained on. If invariance worked,
the probe should be near chance (1/18 = 5.6%).

Yields the same 5-tuple as Stage2Dataset (with dummy phone tensors so the
existing `collate_stage2` and `_train_sid_probe` paths work unmodified).
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

_SPK_RE = re.compile(r"cmu_us_([a-z0-9]+)_arctic")


def _enumerate_arctic(root: Path) -> Tuple[List[Tuple[Path, int]], Dict[str, int]]:
    """Walk `<root>/cmu_us_<spk>_arctic/wav/arctic_*.wav` for all 18 speakers."""
    by_spk: Dict[str, List[Path]] = {}
    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir():
            continue
        m = _SPK_RE.match(spk_dir.name)
        if not m:
            continue
        wavs = sorted((spk_dir / "wav").glob("arctic_*.wav"))
        if wavs:
            by_spk[m.group(1)] = wavs
    speakers = sorted(by_spk.keys())
    spk_to_idx = {s: i for i, s in enumerate(speakers)}
    items = [(wav, spk_to_idx[s]) for s in speakers for wav in by_spk[s]]
    return items, spk_to_idx


def _read_audio(path: Path, target_sr: int) -> np.ndarray:
    arr, sr = sf.read(str(path))
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    if sr != target_sr:
        import librosa
        arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return arr.astype(np.float32)


class ArcticSIDDataset(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], sample_rate: int) -> None:
        self.items = items
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, spk_idx = self.items[idx]
        arr = _read_audio(path, self.sample_rate)
        audio = torch.from_numpy(arr)
        # Empty phone-id tensor preserves the 4-tuple shape expected by
        # collate_stage2 / _train_sid_probe (which uses _ for phones anyway).
        return audio, arr.shape[0], torch.zeros(0, dtype=torch.long), int(spk_idx)


def make_arctic_sid_dataloaders(
    arctic_root: Path,
    sample_rate: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[int, DataLoader, DataLoader, DataLoader]:
    """Random per-speaker 80/10/10 utterance split. Returns (num_speakers, train_dl, val_dl, test_dl)."""
    from .collate import collate_stage2

    items, spk_to_idx = _enumerate_arctic(Path(arctic_root))
    num_speakers = len(spk_to_idx)
    if num_speakers == 0:
        raise RuntimeError(f"No ARCTIC speakers found under {arctic_root}")

    rng = random.Random(seed)
    by_idx: Dict[int, List[Tuple[Path, int]]] = {}
    for it in items:
        by_idx.setdefault(it[1], []).append(it)
    train_items: List[Tuple[Path, int]] = []
    val_items:   List[Tuple[Path, int]] = []
    test_items:  List[Tuple[Path, int]] = []
    for spk_idx in sorted(by_idx.keys()):
        spk_items = by_idx[spk_idx][:]
        rng.shuffle(spk_items)
        n = len(spk_items)
        n_train = int(round(n * train_frac))
        n_val   = int(round(n * val_frac))
        train_items.extend(spk_items[:n_train])
        val_items.extend(spk_items[n_train:n_train + n_val])
        test_items.extend(spk_items[n_train + n_val:])

    train_ds = ArcticSIDDataset(train_items, sample_rate)
    val_ds   = ArcticSIDDataset(val_items,   sample_rate)
    test_ds  = ArcticSIDDataset(test_items,  sample_rate)
    print(f"[arctic_sid] {num_speakers} speakers  "
          f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  seed={seed}")

    pin = torch.cuda.is_available()
    kw  = dict(num_workers=num_workers, pin_memory=pin)
    return (
        num_speakers,
        DataLoader(train_ds, batch_size=batch_size,      shuffle=True,  collate_fn=collate_stage2, **kw),
        DataLoader(val_ds,   batch_size=eval_batch_size, shuffle=False, collate_fn=collate_stage2, **kw),
        DataLoader(test_ds,  batch_size=eval_batch_size, shuffle=False, collate_fn=collate_stage2, **kw),
    )
