"""LibriSpeech phone tokenizer and DataLoader factory for Phone Recognition.

Text → phone pipeline  (SUPERB-compliant, matches phoneme.txt exactly)
-----------------------------------------------------------------------
1. Uppercase the transcript and split into words.
2. Look up each word in the official LibriSpeech lexicon
   (librispeech-lexicon.txt from openslr.org/11).
   - Keep raw CMU ARPAbet phones WITH stress digits (AH0, AH1, AA2, …).
   - OOV words: run g2p_en if available, else emit SPN token.
3. The flat list of phone strings is integer-encoded with PhoneTokenizer.encode(),
   which appends <eos> (index 1), and passed to CTC.

Tokenizer index layout (matches SUPERB CharacterTextEncoder + phoneme.txt):
  0  <pad>  — CTC blank
  1  <eos>  — appended by encode()
  2  <unk>  — unknown phone
  3  SIL,  4  SPN,  5  AA0  …  73  ZH
  vocab_size = 74

LibriSpeech loading  (SUPERB-compliant)
----------------------------------------
train-clean-100  → training
dev-clean        → validation (early stopping)
test-clean       → test
"""

from __future__ import annotations

import io
import itertools
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Audio

sys.path.insert(0, str(Path(__file__).parent.parent))
from pr_config import PRConfig, SUPERB_PHONES
from reproducibility import dataloader_seed_kwargs


# ====================================================================
# LibriSpeech lexicon loading (lazy, cached at module level)
# ====================================================================

_LEXICON: Optional[dict] = None
_G2P = None  # g2p_en instance or False if unavailable


def _get_lexicon(path) -> dict:
    """Load (and cache) the official LibriSpeech lexicon from disk.

    Returns UPPERCASE word → list of raw CMU phone strings (with stress digits).
    Alternate pronunciations (WORD(2), WORD(3)) are discarded; first entry kept.
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
                base = word.split("(")[0]
                if base not in lexicon:
                    lexicon[base] = parts[1:]   # raw CMU phones e.g. ["AH0", "AA1"]
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
    """Maps CMU ARPAbet phone strings (stress-marked) to integer ids.

    Index layout — matches SUPERB's CharacterTextEncoder + phoneme.txt:
        0           <pad>  — CTC blank
        1           <eos>  — appended to every encoded sequence
        2           <unk>  — unknown phone
        3 … 73      71 SUPERB phones (SIL, SPN, AA0 … ZH)
    """

    BLANK_ID = 0
    EOS_ID   = 1
    UNK_ID   = 2

    def __init__(self) -> None:
        all_tokens = ["<pad>", "<eos>", "<unk>"] + SUPERB_PHONES  # 74 total
        self.phone_to_id: dict[str, int] = {p: i for i, p in enumerate(all_tokens)}
        self.id_to_phone: dict[int, str] = {i: p for i, p in enumerate(all_tokens)}
        self.vocab_size: int = len(all_tokens)   # 74

    def encode(self, phones: List[str]) -> torch.Tensor:
        """Map phone list → 1-D LongTensor; appends EOS (index 1)."""
        ids = [self.phone_to_id.get(p, self.UNK_ID) for p in phones]
        ids.append(self.EOS_ID)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Iterable[int]) -> str:
        """Map integer ids → space-separated phone string. Stops at EOS."""
        phones = []
        for i in ids:
            if int(i) == self.EOS_ID:
                break
            tok = self.id_to_phone.get(int(i), "<unk>")
            if tok not in ("<pad>", "<eos>", "<unk>"):
                phones.append(tok)
        return " ".join(phones)


# ====================================================================
# Text → phone conversion
# ====================================================================

_PHONE_VOCAB: Optional[set] = None

def _phone_vocab_set() -> set:
    global _PHONE_VOCAB
    if _PHONE_VOCAB is None:
        _PHONE_VOCAB = set(SUPERB_PHONES)
    return _PHONE_VOCAB


def text_to_phones(text: str, lexicon: dict) -> List[str]:
    """Convert a transcript to a flat list of raw CMU ARPAbet phone strings.

    Stress digits are kept (AH0, AH1, AA2, …) to match SUPERB's phoneme.txt.
    OOV words fall back to g2p_en; phones not in SUPERB's vocab map to SPN.
    """
    vocab = _phone_vocab_set()
    phones: List[str] = []
    g2p = _get_g2p()

    for word in text.upper().split():
        word = word.strip("'-.,!?;:")
        if not word:
            continue
        if word in lexicon:
            for p in lexicon[word]:
                phones.append(p if p in vocab else "SPN")
        elif g2p is not False:
            raw = g2p(word)
            for p in raw:
                p = p.strip()
                if not p or p.isspace():
                    continue
                phones.append(p if p in vocab else "SPN")
        else:
            phones.append("SPN")
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
        text  = ex["text"]
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
    cfg.vocab_size = tokenizer.vocab_size   # 74

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
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, **kw,
        **dataloader_seed_kwargs(cfg.seed, stream=0),
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.eval_batch_size, shuffle=False, **kw,
        **dataloader_seed_kwargs(cfg.seed, stream=1),
    )
    test_dl = DataLoader(
        test_ds, batch_size=cfg.eval_batch_size, shuffle=False, **kw,
        **dataloader_seed_kwargs(cfg.seed, stream=2),
    )
    return tokenizer, train_dl, val_dl, test_dl
