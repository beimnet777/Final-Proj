"""IEMOCAP emotion dataloaders for the disentanglement auxiliary task."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


EMOTION_MAP: dict[str, int] = {
    "neu": 0,
    "hap": 1,
    "exc": 1,
    "sad": 2,
    "ang": 3,
}
EMOTION_NAMES = ("neutral", "happy_excited", "sad", "angry")

Record = Tuple[Path, int]


def _parse_session_labels(session_dir: Path) -> List[Record]:
    emo_dir = session_dir / "dialog" / "EmoEvaluation"
    wav_root = session_dir / "sentences" / "wav"
    records: List[Record] = []
    for label_file in sorted(emo_dir.glob("*.txt")):
        with label_file.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("//") or line.startswith("C-"):
                    continue
                match = re.match(r"\[[\d\. -]+\]\s+(\S+)\s+(\w+)", line)
                if not match:
                    continue
                utt_id, emotion = match.group(1), match.group(2)
                if emotion not in EMOTION_MAP:
                    continue
                dialog_id = "_".join(utt_id.split("_")[:-1])
                wav_path = wav_root / dialog_id / f"{utt_id}.wav"
                if wav_path.exists():
                    records.append((wav_path, EMOTION_MAP[emotion]))
    return records


class IEMOCAPEmotionDataset(Dataset):
    """Utterance-level emotion dataset returning (waveform, samples, label)."""

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
        wav = wav.mean(0)
        return wav, int(wav.shape[0]), int(label)


def collate_iemocap_emotion(batch):
    audios, lengths, labels = zip(*batch)
    return (
        pad_sequence(list(audios), batch_first=True, padding_value=0.0),
        torch.tensor(list(lengths), dtype=torch.long),
        torch.tensor(list(labels), dtype=torch.long),
    )


def _class_counts(records: List[Record]) -> str:
    counts = Counter(label for _path, label in records)
    return " ".join(f"{EMOTION_NAMES[i]}={counts.get(i, 0)}" for i in range(len(EMOTION_NAMES)))


def make_iemocap_emotion_dataloaders(cfg):
    """Build train/val/test IEMOCAP loaders for one leave-session-out fold."""

    root = Path(getattr(cfg, "iemocap_root"))
    if not root.exists():
        raise FileNotFoundError(
            f"IEMOCAP root not found: {root}. Set --iemocap_root or IEMOCAP_ROOT "
            "to the extracted IEMOCAP_full_release directory."
        )
    fold = int(getattr(cfg, "iemocap_fold", 5))
    if fold < 1 or fold > 5:
        raise ValueError(f"iemocap_fold must be in [1, 5], got {fold}")

    train_records: List[Record] = []
    test_records: List[Record] = []
    for session_num in range(1, 6):
        session_dir = root / f"Session{session_num}"
        if not session_dir.exists():
            raise FileNotFoundError(f"IEMOCAP session directory not found: {session_dir}")
        records = _parse_session_labels(session_dir)
        if session_num == fold:
            test_records.extend(records)
        else:
            train_records.extend(records)

    if not train_records or not test_records:
        raise RuntimeError(
            f"No IEMOCAP records found under {root}; expected Session*/dialog/EmoEvaluation "
            "and Session*/sentences/wav layout."
        )

    full_train_ds = IEMOCAPEmotionDataset(train_records, getattr(cfg, "sample_rate", 16_000))
    val_frac = float(getattr(cfg, "iemocap_val_fraction", 0.20))
    val_size = max(1, int(len(full_train_ds) * val_frac))
    train_size = len(full_train_ds) - val_size
    generator = torch.Generator().manual_seed(int(getattr(cfg, "seed", 42)))
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=generator)
    test_ds = IEMOCAPEmotionDataset(test_records, getattr(cfg, "sample_rate", 16_000))

    cfg.emotion_num_classes = len(EMOTION_NAMES)
    print(
        f"[iemocap] fold={fold} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"classes={','.join(EMOTION_NAMES)}"
    )
    print(f"[iemocap] train pool counts: {_class_counts(train_records)}")
    print(f"[iemocap] heldout counts  : {_class_counts(test_records)}")

    pin = torch.cuda.is_available()
    common = dict(num_workers=getattr(cfg, "num_workers", 0), pin_memory=pin,
                  collate_fn=collate_iemocap_emotion)
    train_generator = torch.Generator().manual_seed(int(getattr(cfg, "seed", 42)) + 13)
    return (
        DataLoader(train_ds, batch_size=getattr(cfg, "iemocap_batch_size", 8),
                   shuffle=True, generator=train_generator, **common),
        DataLoader(val_ds, batch_size=getattr(cfg, "iemocap_eval_batch_size", 16),
                   shuffle=False, **common),
        DataLoader(test_ds, batch_size=getattr(cfg, "iemocap_eval_batch_size", 16),
                   shuffle=False, **common),
    )
