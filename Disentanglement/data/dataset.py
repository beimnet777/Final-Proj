"""LibriSpeech dataset for SAE reconstruction training.

Each sample yields (waveform, n_samples).
No phone labels or speaker IDs — reconstruction only.
"""

from __future__ import annotations

import io
import itertools
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio

from .collate import collate_fn


def _decode_audio(audio_dict: dict, target_sr: int) -> np.ndarray:
    arr, sr = sf.read(io.BytesIO(audio_dict["bytes"]))
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return arr.astype(np.float32)


def _stream_examples(split_name: str, cache_dir: str, n: Optional[int]) -> List[dict]:
    ds = load_dataset(
        "librispeech_asr", "clean",
        split=split_name,
        streaming=True,
        cache_dir=cache_dir,
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    if n is not None:
        return list(itertools.islice(ds, n))
    return list(ds)


class DISDataset(Dataset):
    """Returns (waveform_tensor, n_samples) pairs."""

    def __init__(self, examples: List[dict], sample_rate: int) -> None:
        self.examples    = examples
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ex  = self.examples[idx]
        arr = _decode_audio(ex["audio"], self.sample_rate)
        return torch.from_numpy(arr), arr.shape[0]


def make_dis_dataloaders(cfg):
    """Build train / val DataLoaders.

    Returns
    -------
    train_dl : DataLoader
    val_dl   : DataLoader
    """
    n_train = cfg.max_train_examples if cfg.max_train_examples > 0 else None
    n_val   = cfg.max_val_examples   if cfg.max_val_examples   > 0 else None

    print("[dis_data] loading train-clean-100 …")
    trn_ex = _stream_examples("train.100", str(cfg.librispeech_cache_dir), n_train)
    print("[dis_data] loading dev-clean …")
    val_ex = _stream_examples("validation", str(cfg.librispeech_cache_dir), n_val)

    train_ds = DISDataset(trn_ex, cfg.sample_rate)
    val_ds   = DISDataset(val_ex, cfg.sample_rate)

    print(f"[dis_data]  train={len(train_ds)}  val={len(val_ds)}")

    pin = torch.cuda.is_available()
    loader_kw = dict(collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=pin)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,      shuffle=True,  **loader_kw)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.eval_batch_size, shuffle=False, **loader_kw)
    return train_dl, val_dl
