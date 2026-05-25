"""VoxCeleb1 data loading for the Speaker Identification task.

Splits
------
- train : dev/ minus the last val_split fraction of each speaker's utterances
- val   : last val_split fraction of dev/ utterances (for early stopping)
- test  : test/ (official VoxCeleb1 test partition)

Speaker → class index mapping is built from the sorted list of all speaker IDs
found in dev/, so it is deterministic and does not require an external metadata
file.

A record is (wav_path, speaker_class_index).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from sid_config import SIDConfig

Record = Tuple[Path, int]


# -------------------------------------------------------------- Scanning ---


def _build_speaker_map(dev_root: Path) -> Dict[str, int]:
    """Return {speaker_id: class_index} sorted alphabetically."""
    speakers = sorted(p.name for p in (dev_root / "wav").iterdir() if p.is_dir())
    return {spk: idx for idx, spk in enumerate(speakers)}


def _collect_records(split_root: Path, speaker_map: Dict[str, int]) -> List[Record]:
    """Walk split_root/wav/{spk}/{video}/*.wav and return (path, label) list."""
    records: List[Record] = []
    wav_dir = split_root / "wav"
    for spk_dir in sorted(wav_dir.iterdir()):
        if not spk_dir.is_dir() or spk_dir.name not in speaker_map:
            continue
        label = speaker_map[spk_dir.name]
        for video_dir in sorted(spk_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for wav_file in sorted(video_dir.glob("*.wav")):
                records.append((wav_file, label))
    return records


# -------------------------------------------------------------- Dataset ---


class VoxCeleb1Dataset(Dataset):
    """Utterance-level speaker dataset returning (waveform, num_samples, label)."""

    def __init__(
        self,
        records: List[Record],
        sample_rate: int = 16_000,
        max_duration_s: float = 0.0,
    ) -> None:
        self.records = records
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_s * sample_rate) if max_duration_s > 0 else 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        wav_path, label = self.records[idx]
        wav, sr = torchaudio.load(str(wav_path))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(0)   # stereo → mono if needed: (T,)
        if self.max_samples > 0 and wav.shape[0] > self.max_samples:
            wav = wav[:self.max_samples]
        return wav, wav.shape[0], label


# ------------------------------------------------------------ Collation ---


def _collate(batch):
    """Zero-pad waveforms to the longest in the batch."""
    waveforms, lengths, labels = zip(*batch)
    max_len = max(lengths)
    padded = torch.zeros(len(waveforms), max_len)
    for i, (wav, l) in enumerate(zip(waveforms, lengths)):
        padded[i, :l] = wav
    return (
        padded,
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels,  dtype=torch.long),
    )


# --------------------------------------------------------- DataLoaders ---


def make_sid_dataloaders(cfg: SIDConfig):
    """Build train / val / test DataLoaders for VoxCeleb1 SID.

    Returns (train_dl, val_dl, test_dl, speaker_map).
    speaker_map is {speaker_id: class_index} and is saved for reference.
    """
    root     = Path(cfg.voxceleb1_root)
    dev_root = root / "dev"
    tst_root = root / "test"

    speaker_map = _build_speaker_map(dev_root)
    cfg.num_classes = len(speaker_map)   # update in case it differs from default

    dev_records  = _collect_records(dev_root, speaker_map)
    test_records = _collect_records(tst_root, speaker_map)

    # Per-speaker val split: hold out last val_split fraction of each speaker's
    # dev utterances so every speaker appears in both train and val.
    from collections import defaultdict
    by_speaker: Dict[int, List[Record]] = defaultdict(list)
    for rec in dev_records:
        by_speaker[rec[1]].append(rec)

    train_records: List[Record] = []
    val_records:   List[Record] = []
    for spk_recs in by_speaker.values():
        n_val = max(1, int(len(spk_recs) * cfg.val_split))
        val_records.extend(spk_recs[-n_val:])
        train_records.extend(spk_recs[:-n_val])

    train_ds = VoxCeleb1Dataset(train_records, cfg.sample_rate, cfg.max_duration_s)
    val_ds   = VoxCeleb1Dataset(val_records,   cfg.sample_rate, cfg.max_duration_s)
    test_ds  = VoxCeleb1Dataset(test_records,  cfg.sample_rate, cfg.max_duration_s)

    print(
        f"[SID]  speakers={cfg.num_classes}"
        f"  train={len(train_ds)}"
        f"  val={len(val_ds)}"
        f"  test={len(test_ds)}"
    )

    _loader = lambda ds, bs, shuffle: DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=cfg.num_workers, collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    )
    train_dl = _loader(train_ds, cfg.batch_size,       True)
    val_dl   = _loader(val_ds,   cfg.eval_batch_size,  False)
    test_dl  = _loader(test_ds,  cfg.eval_batch_size,  False)
    return train_dl, val_dl, test_dl, speaker_map
