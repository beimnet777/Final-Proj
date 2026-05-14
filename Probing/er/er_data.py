"""IEMOCAP data loading for the Emotion Recognition task.

IEMOCAP directory layout expected on disk
------------------------------------------
IEMOCAP_full_release/
    Session{1-5}/
        sentences/
            wav/
                {dialog_id}/          e.g. Ses01F_impro01/
                    {utt_id}.wav      e.g. Ses01F_impro01_F000.wav
        dialog/
            EmoEvaluation/
                {dialog_id}.txt       annotation file, one utterance per line

Annotation line format:
    [start - end]\\tutt_id\\temotion\\t[valence, arousal, dominance]

Only the 4 standard SUPERB classes are kept:
    neutral (neu), happy/excited (hap/exc → 1), sad (sad), angry (ang).
Utterances labelled with any other emotion code are discarded.

Cross-validation protocol
--------------------------
Five sessions → five folds.  For fold k:
    test  = Session k
    train = Sessions 1–5 minus Session k, minus last 10 % (used as val)
    val   = last 10 % of the non-test utterances (for early stopping)

This matches the 5-fold leave-one-session-out convention used in SUPERB.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from er_config import EMOTION_MAP, ERConfig

# A record is (path_to_wav, integer_class_label).
Record = Tuple[Path, int]


# -------------------------------------------------------------- Parsing ---


def _parse_session_labels(session_dir: Path) -> List[Record]:
    """Return all (wav_path, label) records for one IEMOCAP session.

    Utterances whose emotion code is not in EMOTION_MAP are silently skipped.
    """
    emo_dir = session_dir / "dialog" / "EmoEvaluation"
    wav_root = session_dir / "sentences" / "wav"
    records: List[Record] = []

    for label_file in sorted(emo_dir.glob("*.txt")):
        with label_file.open() as f:
            for line in f:
                line = line.strip()
                # Skip blank lines, comment blocks, and per-evaluator lines.
                if not line or line.startswith("//") or line.startswith("C-"):
                    continue
                # Expected: [start - end]\tutt_id\temotion\t[v, a, d]
                m = re.match(r"\[[\d\. -]+\]\s+(\S+)\s+(\w+)", line)
                if not m:
                    continue
                utt_id, emotion = m.group(1), m.group(2)
                if emotion not in EMOTION_MAP:
                    continue
                label = EMOTION_MAP[emotion]
                # Wav lives one directory below the dialog name.
                dialog_id = "_".join(utt_id.split("_")[:-1])
                wav_path = wav_root / dialog_id / f"{utt_id}.wav"
                if wav_path.exists():
                    records.append((wav_path, label))
    return records


# -------------------------------------------------------------- Dataset ---


class IEMOCAPDataset(Dataset):
    """Utterance-level emotion dataset returning (waveform, num_samples, label)."""

    def __init__(self, records: List[Record], sample_rate: int = 16_000) -> None:
        self.records = records
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        wav_path, label = self.records[idx]
        wav, sr = torchaudio.load(str(wav_path))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(0)   # (channels, T) → (T,)  — handles mono & stereo
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
        torch.tensor(labels, dtype=torch.long),
    )


# --------------------------------------------------------- DataLoaders ---


def make_er_dataloaders(cfg: ERConfig):
    """Build train / val / test DataLoaders for one cross-validation fold.

    Returns (train_dl, val_dl, test_dl).
    """
    root = Path(cfg.iemocap_root)
    test_session = cfg.test_fold

    train_records: List[Record] = []
    test_records:  List[Record] = []

    for session_num in range(1, 6):
        session_dir = root / f"Session{session_num}"
        records = _parse_session_labels(session_dir)
        if session_num == test_session:
            test_records.extend(records)
        else:
            train_records.extend(records)

    # Hold out last 10 % of non-test utterances as validation.
    val_size = max(1, int(len(train_records) * 0.10))
    val_records   = train_records[-val_size:]
    train_records = train_records[:-val_size]

    train_ds = IEMOCAPDataset(train_records, cfg.sample_rate)
    val_ds   = IEMOCAPDataset(val_records,   cfg.sample_rate)
    test_ds  = IEMOCAPDataset(test_records,  cfg.sample_rate)

    print(
        f"[fold {cfg.test_fold}]  "
        f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
    )

    _loader = lambda ds, bs, shuffle: DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=cfg.num_workers, collate_fn=_collate,
        pin_memory=True,
    )
    train_dl = _loader(train_ds, cfg.batch_size,       True)
    val_dl   = _loader(val_ds,   cfg.eval_batch_size,  False)
    test_dl  = _loader(test_ds,  cfg.eval_batch_size,  False)
    return train_dl, val_dl, test_dl
