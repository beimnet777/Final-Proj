"""LibriSpeech 10h slice + character tokenizer + CTC-friendly collate function.

Pipeline:
    We stream librispeech_asr train.100 with a shuffle buffer, materialise only
    the ~`cfg.train_hours` examples we need, then split off `cfg.val_split` for
    validation. test.clean is loaded normally (it's already a small split).

    Streaming matters: without it, `load_dataset("librispeech_asr", "clean", ...)`
    downloads the *entire* "clean" config -- train.100 + train.360 (~28 GB) --
    even if you only request split="train.100". With streaming we pull ~2 shards
    (~950 MB) instead.
"""

from __future__ import annotations

import io
import itertools
import random
from typing import Iterable, List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Audio

from config import Config
from typing import Optional


class CharTokenizer:
    """Maps characters to integer ids and back.

    Index layout (matches config.Config):
        0           -> CTC blank symbol
        1..len(V)   -> visible characters from cfg.vocab
    Characters not in cfg.vocab are silently dropped on encode.
    """

    def __init__(self, vocab: str, blank_id: int = 0):
        self.blank_id = blank_id
        self.char_to_id = {c: i + 1 for i, c in enumerate(vocab)}
        self.id_to_char = {i + 1: c for i, c in enumerate(vocab)}

    def encode(self, text: str) -> torch.Tensor:
        text = text.lower()
        ids = [self.char_to_id[c] for c in text if c in self.char_to_id]
        return torch.tensor(ids, dtype=torch.long)  # (T_text,)

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.id_to_char.get(int(i), "") for i in ids)


# -- LibriSpeech loaders -----------------------------------------------------

# LibriSpeech utterances average ~12.5s. We use this to estimate how many
# examples to stream without decoding each file to measure its duration.
_LIBRISPEECH_AVG_SEC = 12.5


def _decode_audio(audio_dict: dict, target_sr: int) -> np.ndarray:
    """Decode raw audio bytes from a HF parquet audio dict to a float32 array.

    HF stores audio as {"bytes": b"...", "path": "file.flac"} in the parquet.
    We decode with soundfile (no FFmpeg / torchcodec needed) and resample with
    librosa only when the native rate differs from target_sr (LibriSpeech is
    already 16 kHz so the resample branch is rarely hit).
    """
    arr, sr = sf.read(io.BytesIO(audio_dict["bytes"]))
    if arr.ndim > 1:
        arr = arr.mean(axis=1)          # stereo → mono
    if sr != target_sr:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return arr.astype(np.float32)


def _stream_examples(split_name: str, cfg: Config, n: Optional[int] = None) -> List[dict]:
    """Stream `n` examples from `split_name` and return as a plain list of dicts.

    Using streaming=True means HF downloads shards on demand and stops once we
    have enough — so for a 10h slice we pull ~2 train.100 shards (~950 MB)
    instead of all 48 shards in the "clean" config (~28 GB).

    We intentionally do NOT call cast_column(Audio(...)) here so we never
    trigger datasets' audio backend (torchcodec / soundfile). Audio bytes stay
    raw; _decode_audio handles decoding in LibriSpeechCharDataset.__getitem__.
    """
    ds = load_dataset(
        "librispeech_asr", "clean",
        split=split_name,
        streaming=True,
        cache_dir=str(cfg.data_cache_dir),
    )
    # Disable auto-decoding so datasets returns raw {"bytes": ..., "path": ...}
    # instead of calling torchcodec. Our _decode_audio handles it from there.
    ds = ds.cast_column("audio", Audio(decode=False))
    # No streaming shuffle: buffer-based shuffle pre-fetches across shards,
    # which causes train.360 to be downloaded. We shuffle after materialising.
    if n is not None:
        return list(itertools.islice(ds, n))
    return list(ds)


def build_datasets(cfg: Config):
    """Return (train_examples, val_examples, test_hf).

    train_examples and val_examples are plain Python lists of dicts:
        {"audio": {"bytes": b"...", "path": "..."}, "text": str, ...}

    test_hf is a regular HF Dataset (test.clean is small enough to load fully).

    LibriSpeechCharDataset accepts both types transparently since both support
    integer indexing and have the same audio/text key structure.
    """
    n_total = int(cfg.train_hours * 3600 / _LIBRISPEECH_AVG_SEC)
    examples = _stream_examples("train.100", cfg, n=None) # stream a slice of the training set, then split into train/val

    # Second shuffle so val examples aren't drawn from a biased tail of the
    # streaming buffer.
    rng = random.Random(cfg.seed)
    rng.shuffle(examples)

    n_val = max(1, int(len(examples) * cfg.val_split))
    val_examples   = examples[:n_val]
    train_examples = examples[n_val:]

    # test.clean (~350 MB, single shard) — load fully, already cached.
    test_examples = _stream_examples("test", cfg, n=None) # the n_total is set to None, so it will load the entire test set

    # Length-bucket the test set: sort by encoded audio byte size as a proxy
    # for duration (FLAC compression ratio is stable on speech, so longer
    # utterances → larger byte arrays). With sorted-by-length batches, each
    # batch pads to a length close to its members' true length, instead of
    # to one outlier. Test order doesn't affect metrics so this is safe.
    test_examples.sort(key=lambda ex: len(ex["audio"]["bytes"]))

    return train_examples, val_examples, test_examples


# -- Torch Dataset wrapper ---------------------------------------------------


class LibriSpeechCharDataset(Dataset):
    """Wraps a list-of-dicts or HF Dataset, decoding audio on access."""

    def __init__(self, ds, tokenizer: CharTokenizer, sample_rate: int):
        self.ds = ds
        self.tok = tokenizer
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        ex = self.ds[idx]
        # ex["audio"] is {"bytes": b"...", "path": "..."} — raw parquet bytes.
        arr = _decode_audio(ex["audio"], self.sample_rate)        # (T_audio,)
        audio = torch.from_numpy(arr)
        text = ex["text"].lower()
        target = self.tok.encode(text)                            # (T_text,)
        return audio, target, text


# -- Collate (variable-length audio + variable-length targets) ---------------


def collate_fn(batch):
    """Pad audios to max length, pack targets for CTC.

    Returns
    -------
    audios          : (B, T_audio_max) float, zero-padded
    audio_lengths   : (B,)             long,  true sample counts (unpadded)
    targets         : (B, T_text_max)  long,  zero-padded
    target_lengths  : (B,)             long,  unpadded character counts
    texts           : list[str]                original transcripts (for eval)
    """
    audios, targets, texts = zip(*batch)
    audio_lengths = torch.tensor([a.size(0) for a in audios], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)

    audios = pad_sequence(audios, batch_first=True, padding_value=0.0)  # (B, T_audio_max)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)  # (B, T_text_max)

    return audios, audio_lengths, targets, target_lengths, list(texts)


def make_dataloaders(cfg: Config):
    """Return (tokenizer, train_dl, val_dl, test_dl)."""
    tokenizer = CharTokenizer(cfg.vocab, blank_id=cfg.blank_id)
    train_data, val_data, test_data = build_datasets(cfg)

    train_ds = LibriSpeechCharDataset(train_data, tokenizer, cfg.sample_rate)
    val_ds   = LibriSpeechCharDataset(val_data,   tokenizer, cfg.sample_rate)
    test_ds  = LibriSpeechCharDataset(test_data,  tokenizer, cfg.sample_rate)

    pin = torch.cuda.is_available()  # pin_memory only works on CUDA
    common = dict(collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=pin)
    # Eval uses a bigger batch (no grad → less memory) to cut the number of
    # SPEAR forwards. shuffle=False on val/test preserves the length-sorted
    # order on test so padding stays minimal.
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,      shuffle=True,  **common)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.eval_batch_size, shuffle=False, **common)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.eval_batch_size, shuffle=False, **common)

    return tokenizer, train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Cached-features dataset (used when encoder outputs are pre-extracted)
# ---------------------------------------------------------------------------

class CachedFeaturesDataset(Dataset):
    """Wraps a list of {"feat": (T_i, D), "target": (T_text,), "text": str}
    dicts written by cache_features.py.  No audio decoding or encoder forward
    pass required — __getitem__ just returns the pre-computed tensor."""

    def __init__(self, records: list):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        r = self.records[idx]
        # feat is stored fp16; cast to fp32 here so the probe and loss stay in
        # fp32 without needing AMP.
        return r["feat"].float(), r["target"], r["text"]


def _cached_collate_fn(batch):
    """Collate for CachedFeaturesDataset.

    Returns the same 5-tuple as collate_fn so training/eval loops are
    identical:
        feats        : (B, T_max, D) float32, zero-padded
        frame_lens   : (B,)          long, true frame counts
        targets      : (B, T_text_max) long, zero-padded
        target_lens  : (B,)          long
        texts        : list[str]
    """
    feats, targets, texts = zip(*batch)
    frame_lens   = torch.tensor([f.size(0) for f in feats],   dtype=torch.long)
    target_lens  = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    feats   = pad_sequence(feats,   batch_first=True, padding_value=0.0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return feats, frame_lens, targets, target_lens, list(texts)


def make_cached_dataloaders(cfg: "Config", cache_dir):
    """Return (tokenizer, train_dl, val_dl, test_dl) reading from cached .pt files.

    The returned dataloaders produce the same 5-tuple as the live dataloaders
    but feats is already the extracted final-layer tensor — no encoder needed.
    """
    from pathlib import Path as _Path
    cache_dir = _Path(cache_dir)

    tokenizer = CharTokenizer(cfg.vocab, blank_id=cfg.blank_id)

    def _load(split: str) -> CachedFeaturesDataset:
        path = cache_dir / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Cache file not found: {path}\n"
                f"Run  python cache_features.py --cache_dir {cache_dir}  first."
            )
        data = torch.load(path, map_location="cpu", weights_only=False)
        return CachedFeaturesDataset(data["records"])

    train_ds = _load("train")
    val_ds   = _load("val")
    test_ds  = _load("test")

    pin = torch.cuda.is_available()
    common = dict(collate_fn=_cached_collate_fn, num_workers=cfg.num_workers,
                  pin_memory=pin)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,      shuffle=True,  **common)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.eval_batch_size, shuffle=False, **common)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.eval_batch_size, shuffle=False, **common)
    return tokenizer, train_dl, val_dl, test_dl


if __name__ == "__main__":
    # Smoke test: build dataloaders, fetch one batch, print shapes.
    cfg = Config()
    tokenizer, train_dl, val_dl, test_dl = make_dataloaders(cfg)
    print(f"train batches : {len(train_dl)}")
    print(f"val   batches : {len(val_dl)}")
    print(f"test  batches : {len(test_dl)}")

    audios, audio_lens, targets, target_lens, texts = next(iter(train_dl))
    print(f"audios  : {tuple(audios.shape)}        # (B, T_audio_max)")
    print(f"audio_lens : {audio_lens.tolist()}")
    print(f"targets : {tuple(targets.shape)}       # (B, T_text_max)")
    print(f"target_lens : {target_lens.tolist()}")
    print(f"first text     : {texts[0]!r}")
    decoded = tokenizer.decode(targets[0][: target_lens[0]].tolist())
    print(f"first decoded  : {decoded!r}")
