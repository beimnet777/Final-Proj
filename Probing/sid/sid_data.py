"""VoxCeleb1 data loading for the Speaker Identification task.

Splits (SUPERB-compliant via veri_test_class.txt)
--------------------------------------------------
All utterances come from dev/wav/.  The manifest assigns each file to:
  index 1 → train   (138,361 utts, 1,251 speakers)
  index 2 → val     (  6,904 utts, 1,251 speakers)  — used for early stopping
  index 3 → test    (  8,251 utts, 1,251 speakers)  — final evaluation

Training applies a random 8-second crop (SUPERB: max_timestep=128,000).
Val and test are evaluated on full utterances (no truncation).

Speaker label: int(speaker_id[2:]) - 10001  →  id10001=0 … id11251=1250
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from sid_config import SIDConfig

Record = Tuple[Path, int]


# --------------------------------------------------------- Meta-file parse ---


def _parse_meta(
    meta_path: Path,
    vox_root: Path,
    max_examples: int = 0,
) -> Tuple[List[Record], List[Record], List[Record]]:
    """Parse veri_test_class.txt → (train, val, test) record lists.

    All paths live under vox_root/dev/wav/.
    Label = int(speaker_id[2:]) - 10001  (matches SUPERB exactly).
    """
    train, val, test = [], [], []
    dev_wav = vox_root / "dev" / "wav"

    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, rel = line.split(None, 1)
            idx = int(idx)
            full_path = dev_wav / rel
            speaker_id = rel.split("/")[0]          # e.g. "id10003"
            label = int(speaker_id[2:]) - 10001     # id10001 → 0
            rec = (full_path, label)
            if idx == 1:
                train.append(rec)
            elif idx == 2:
                val.append(rec)
            else:
                test.append(rec)

    if max_examples > 0:
        train = train[:max_examples]
        val   = val[:max_examples]
        test  = test[:max_examples]

    return train, val, test


# -------------------------------------------------------------- Dataset ---


class VoxCeleb1Dataset(Dataset):
    """Returns (waveform, num_samples, label) tuples.

    random_crop=True  : randomly crop to max_samples during __getitem__ (training).
    random_crop=False : return the full waveform, ignoring max_samples (val/test).
    """

    def __init__(
        self,
        records: List[Record],
        sample_rate: int = 16_000,
        max_samples: int = 0,
        random_crop: bool = False,
    ) -> None:
        self.records     = records
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        wav_path, label = self.records[idx]
        wav, sr = torchaudio.load(str(wav_path))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(0)   # stereo → mono: (T,)

        if self.random_crop and self.max_samples > 0 and wav.shape[0] > self.max_samples:
            start = random.randint(0, wav.shape[0] - self.max_samples)
            wav = wav[start : start + self.max_samples]

        return wav, wav.shape[0], label


# ------------------------------------------------------------ Collation ---


def _collate(batch):
    """Zero-pad waveforms to the longest in the batch."""
    waveforms, lengths, labels = zip(*batch)
    max_len = max(lengths)
    padded  = torch.zeros(len(waveforms), max_len)
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

    Follows the SUPERB split exactly via veri_test_class.txt:
      - train: random 8-second crop (SUPERB max_timestep=128,000)
      - val  : full utterances, no truncation
      - test : full utterances, no truncation

    Returns (train_dl, val_dl, test_dl).
    """
    root = Path(cfg.voxceleb1_root)

    train_records, val_records, test_records = _parse_meta(
        cfg.meta_data, root, max_examples=cfg.max_examples
    )
    cfg.num_classes = 1251   # fixed by SUPERB spec

    max_train_samples = int(cfg.train_max_duration_s * cfg.sample_rate)  # 128,000

    train_ds = VoxCeleb1Dataset(
        train_records, cfg.sample_rate,
        max_samples=max_train_samples, random_crop=True,
    )
    val_ds = VoxCeleb1Dataset(
        val_records, cfg.sample_rate,
        max_samples=0, random_crop=False,
    )
    test_ds = VoxCeleb1Dataset(
        test_records, cfg.sample_rate,
        max_samples=0, random_crop=False,
    )

    print(
        f"[SID]  speakers={cfg.num_classes}"
        f"  train={len(train_ds)}"
        f"  val={len(val_ds)}"
        f"  test={len(test_ds)}"
        f"  train_max={cfg.train_max_duration_s}s  val/test=full"
    )

    pin = torch.cuda.is_available()
    def _loader(ds, bs, shuffle):
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=cfg.num_workers, collate_fn=_collate,
            pin_memory=pin,
        )

    train_dl = _loader(train_ds, cfg.batch_size,      True)
    val_dl   = _loader(val_ds,   cfg.eval_batch_size, False)
    test_dl  = _loader(test_ds,  cfg.eval_batch_size, False)
    return train_dl, val_dl, test_dl
