"""LibriSpeech phone tokenizer and DataLoader factory for Phone Recognition.

Text → phone pipeline  (SUPERB-compliant)
------------------------------------------
1. Lowercase the transcript and split into words.
2. Look up each word in the official LibriSpeech lexicon
   (librispeech-lexicon.txt from openslr.org/11).
   - Strip stress digits from vowel phones (AH1 → ah).
   - Apply a small remap table for non-TIMIT-39 phones.
   - OOV words: run g2p_en if available, else emit SPN token.
3. The resulting flat list of phone strings is integer-encoded with
   PhoneTokenizer.encode() and passed to CTC.

LibriSpeech loading  (SUPERB-compliant)
----------------------------------------
train-clean-100  → training
dev-clean        → validation (early stopping)
test-clean       → test
Audio is decoded lazily in __getitem__ using soundfile.
"""

from __future__ import annotations

import io
import itertools
from typing import Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Audio

from pr_config import (
    PRConfig, ARPABET_39, SPN_TOKEN, _CMU_REMAP,
)


# ====================================================================
# LibriSpeech lexicon loading (lazy, cached at module level)
# ====================================================================

_LEXICON: Optional[dict] = None
_G2P = None  # g2p_en instance or False if unavailable


def _get_lexicon(path) -> dict:
    """Load (and cache) the official LibriSpeech lexicon from disk.

    File format (openslr.org/resources/11/librispeech-lexicon.txt)::

        WORD\tPH1 PH2 PH3 ...
        WORD(2)\tPH1 PH2 ...   <- alternate pronunciation, skip if base seen
    """
    global _LEXICON
    if _LEXICON is None:
        lexicon: dict = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                word = parts[0].upper()
                base = word.split("(")[0]   # strip alternate-pron suffix
                if base not in lexicon:      # keep first (most common) pron
                    lexicon[base] = parts[1:]
        _LEXICON = lexicon
        print(f"[pr_data] loaded LibriSpeech lexicon: {len(_LEXICON):,} entries")
    return _LEXICON


def _get_g2p():
    """Return a g2p_en.G2p instance, or False if the package is unavailable."""
    global _G2P
    if _G2P is None:
        try:
            from g2p_en import G2p
            _G2P = G2p()
            print("[pr_data] g2p_en loaded for OOV words")
        except ImportError:
            _G2P = False
            print("[pr_data] g2p_en not found — OOV words will map to SPN")
    return _G2P


# ====================================================================
# PhoneTokenizer
# ====================================================================

class PhoneTokenizer:
    """Converts phone strings to integer ids and back.

    Index layout:
        0           CTC blank
        1 … 39      ARPAbet 39 phones (ARPABET_39 order)
        40          SPN (spoken noise / OOV)
    """

    BLANK_ID = 0

    def __init__(self) -> None:
        phones = ARPABET_39 + [SPN_TOKEN]
        self.phone_to_id: dict[str, int] = {p: i + 1 for i, p in enumerate(phones)}
        self.id_to_phone: dict[int, str] = {v: k for k, v in self.phone_to_id.items()}
        self.vocab_size: int = 1 + len(phones)   # blank + phones

    # ------------------------------------------------------------------
    def encode(self, phones: List[str]) -> torch.Tensor:
        """Map a list of phone strings to a 1-D LongTensor of ids."""
        ids = [self.phone_to_id.get(p, self.phone_to_id[SPN_TOKEN]) for p in phones]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Iterable[int]) -> str:
        """Map integer ids → space-separated phone string (for PER computation)."""
        return " ".join(self.id_to_phone.get(int(i), SPN_TOKEN) for i in ids)


# ====================================================================
# Text → phone conversion
# ====================================================================

def _normalise_cmu_phone(raw: str) -> str:
    """Strip stress digit and apply remap table to get an ARPAbet-39 phone."""
    p = raw.lower().rstrip("012")      # "AH1" → "ah"
    return _CMU_REMAP.get(p, p)


def text_to_phones(text: str, lexicon: dict) -> List[str]:
    """Convert a transcript to a flat list of ARPAbet-39 phone strings.

    Uses the official LibriSpeech lexicon.  OOV words are handled by g2p_en
    (required); no word is ever skipped or mapped to SPN.
    """
    phones: List[str] = []
    g2p = _get_g2p()
    if g2p is False:
        raise RuntimeError(
            "g2p_en is required but not installed. "
            "Run: pip install g2p_en"
        )
    for word in text.upper().split():
        word = word.strip("'-.,!?;:")
        if not word:
            continue
        if word in lexicon:
            phones.extend(_normalise_cmu_phone(p) for p in lexicon[word])
        else:
            raw = g2p(word)
            converted = [
                _normalise_cmu_phone(p) for p in raw
                if p.strip() and not p.isspace()
            ]
            valid = [p for p in converted if p in set(ARPABET_39)]
            phones.extend(valid if valid else [SPN_TOKEN])
    return phones


# ====================================================================
# Audio decoding
# ====================================================================

def _decode_audio(audio_dict: dict, target_sr: int) -> np.ndarray:
    arr, sr = sf.read(io.BytesIO(audio_dict["bytes"]))
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return arr.astype(np.float32)


# ====================================================================
# LibriSpeech streaming helpers
# ====================================================================

def _stream_examples(split_name: str, cfg: PRConfig,
                     n: Optional[int] = None) -> List[dict]:
    ds = load_dataset(
        "librispeech_asr", "clean",
        split=split_name,
        streaming=True,
        cache_dir=str(cfg.data_cache_dir),
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    if n is not None:
        return list(itertools.islice(ds, n))
    return list(ds)


# ====================================================================
# Dataset
# ====================================================================

class LibriSpeechPhoneDataset(Dataset):
    """Returns (waveform, phone_ids, transcript_text) tuples."""

    def __init__(self, examples: list, tokenizer: PhoneTokenizer,
                 sample_rate: int, lexicon: dict) -> None:
        self.examples    = examples
        self.tokenizer   = tokenizer
        self.sample_rate = sample_rate
        self.lexicon     = lexicon

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        ex = self.examples[idx]
        arr = _decode_audio(ex["audio"], self.sample_rate)
        audio = torch.from_numpy(arr)
        text  = ex["text"].lower()
        phones = text_to_phones(text, self.lexicon)
        target = self.tokenizer.encode(phones)
        return audio, target, text


# ====================================================================
# Collate
# ====================================================================

def collate_fn(batch):
    """Zero-pad audio; pack targets for CTC.

    Returns
    -------
    audios         : (B, T_max)      float
    audio_lengths  : (B,)            long   — true sample counts
    targets        : (B, P_max)      long   — zero-padded phone ids
    target_lengths : (B,)            long   — true phone sequence lengths
    texts          : list[str]               original transcripts
    """
    audios, targets, texts = zip(*batch)
    audio_lengths  = torch.tensor([a.size(0) for a in audios],  dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    audios  = pad_sequence(audios,  batch_first=True, padding_value=0.0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return audios, audio_lengths, targets, target_lengths, list(texts)


# ====================================================================
# DataLoader factory
# ====================================================================

def make_pr_dataloaders(cfg: PRConfig):
    """Build train / val / test DataLoaders and a PhoneTokenizer.

    Follows the SUPERB split exactly:
      train  → train-clean-100  (full, no holdout)
      val    → dev-clean        (official SUPERB validation split)
      test   → test-clean

    Returns (tokenizer, train_dl, val_dl, test_dl).
    """
    tokenizer = PhoneTokenizer()
    cfg.vocab_size = tokenizer.vocab_size   # 41

    lexicon = _get_lexicon(cfg.librispeech_lexicon)
    n_cap = cfg.max_examples if cfg.max_examples > 0 else None

    print("[pr_data] loading train-clean-100 …")
    trn_examples = _stream_examples("train.100", cfg, n=n_cap)

    print("[pr_data] loading dev-clean …")
    val_examples = _stream_examples("validation", cfg, n=n_cap)

    print("[pr_data] loading test-clean …")
    tst_examples = _stream_examples("test", cfg, n=n_cap)
    tst_examples.sort(key=lambda ex: len(ex["audio"]["bytes"]))

    train_ds = LibriSpeechPhoneDataset(trn_examples, tokenizer, cfg.sample_rate, lexicon)
    val_ds   = LibriSpeechPhoneDataset(val_examples,  tokenizer, cfg.sample_rate, lexicon)
    test_ds  = LibriSpeechPhoneDataset(tst_examples,  tokenizer, cfg.sample_rate, lexicon)

    print(
        f"[pr_data]  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
        f"  vocab_size={tokenizer.vocab_size}"
    )

    pin = torch.cuda.is_available()
    kw  = dict(collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=pin)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,      shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.eval_batch_size, shuffle=False, **kw)
    test_dl  = DataLoader(test_ds,  batch_size=cfg.eval_batch_size, shuffle=False, **kw)
    return tokenizer, train_dl, val_dl, test_dl
