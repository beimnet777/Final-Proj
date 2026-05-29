"""LibriSpeech dataset for disentanglement training.

Each sample yields (waveform, phone_ids, speaker_idx, text).

Phone labels are produced with the same lexicon + g2p pipeline used in the
PR probing task (see Probing/pr/pr_data.py), so no new downloads are needed.

Speaker IDs are remapped to 0-indexed local labels from the training set.
Validation examples with unseen speakers are silently dropped.
"""

from __future__ import annotations

import io
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio

from .collate import collate_fn

# ---------------------------------------------------------------------------
# ARPAbet phone set (copied from PR probing for self-containment)
# ---------------------------------------------------------------------------

ARPABET_39 = [
    "aa", "ae", "ah", "ao", "aw", "ay",
    "b",  "ch", "d",  "dh",
    "eh", "er", "ey",
    "f",  "g",  "hh",
    "ih", "iy", "jh",
    "k",  "l",  "m",  "n",  "ng",
    "ow", "oy", "p",  "r",
    "s",  "sh", "t",  "th",
    "uh", "uw", "v",  "w",  "y",  "z",  "zh",
]
SPN_TOKEN = "spn"

_CMU_REMAP = {
    "ax": "ah", "axr": "er", "ix": "ih", "ux": "uw",
    "el": "l",  "em": "m",  "en": "n",  "nx": "ng",
}

# Module-level caches
_LEXICON: Optional[Dict] = None
_G2P = None


def _get_lexicon(path) -> Dict:
    global _LEXICON
    if _LEXICON is None:
        lexicon: Dict = {}
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
                    lexicon[base] = parts[1:]
        _LEXICON = lexicon
        print(f"[dis_data] loaded LibriSpeech lexicon: {len(_LEXICON):,} entries")
    return _LEXICON


def _get_g2p():
    global _G2P
    if _G2P is None:
        try:
            from g2p_en import G2p
            _G2P = G2p()
            print("[dis_data] g2p_en loaded for OOV words")
        except ImportError:
            _G2P = False
            print("[dis_data] g2p_en not found — OOV words map to SPN")
    return _G2P


def _normalise_cmu_phone(raw: str) -> str:
    p = raw.lower().rstrip("012")
    return _CMU_REMAP.get(p, p)


def text_to_phones(text: str, lexicon: Dict) -> List[str]:
    phones: List[str] = []
    g2p = _get_g2p()
    for word in text.upper().split():
        word = word.strip("'-.,!?;:")
        if not word:
            continue
        if word in lexicon:
            phones.extend(_normalise_cmu_phone(p) for p in lexicon[word])
        elif g2p:
            raw = g2p(word)
            converted = [
                _normalise_cmu_phone(p) for p in raw
                if p.strip() and not p.isspace()
            ]
            valid = [p for p in converted if p in set(ARPABET_39)]
            phones.extend(valid if valid else [SPN_TOKEN])
        else:
            phones.append(SPN_TOKEN)
    return phones


# ---------------------------------------------------------------------------
# PhoneTokenizer
# ---------------------------------------------------------------------------

class PhoneTokenizer:
    """CTC token set: 0=blank, 1-39=ARPAbet, 40=SPN."""

    BLANK_ID = 0

    def __init__(self) -> None:
        phones = ARPABET_39 + [SPN_TOKEN]
        self.phone_to_id: Dict[str, int] = {p: i + 1 for i, p in enumerate(phones)}
        self.id_to_phone: Dict[int, str] = {v: k for k, v in self.phone_to_id.items()}
        self.vocab_size: int = 1 + len(phones)   # 41

    def encode(self, phones: List[str]) -> torch.Tensor:
        ids = [self.phone_to_id.get(p, self.phone_to_id[SPN_TOKEN]) for p in phones]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        return " ".join(self.id_to_phone.get(int(i), SPN_TOKEN) for i in ids)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _decode_audio(audio_dict: dict, target_sr: int, max_duration_s: float) -> np.ndarray:
    arr, sr = sf.read(io.BytesIO(audio_dict["bytes"]))
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    max_samples = int(max_duration_s * target_sr)
    if arr.shape[0] > max_samples:
        arr = arr[:max_samples]
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DISDataset(Dataset):
    """Returns (waveform, phone_ids, speaker_idx, text)."""

    def __init__(
        self,
        examples: List[dict],
        tokenizer: PhoneTokenizer,
        speaker_to_idx: Dict[int, int],
        lexicon: Dict,
        sample_rate: int,
        max_duration_s: float,
    ) -> None:
        # Keep only examples whose speaker is in the training speaker map.
        self.examples = [
            ex for ex in examples if ex["speaker_id"] in speaker_to_idx
        ]
        dropped = len(examples) - len(self.examples)
        if dropped:
            print(f"[dis_data] dropped {dropped} examples with unseen speakers")
        self.tokenizer = tokenizer
        self.speaker_to_idx = speaker_to_idx
        self.lexicon = lexicon
        self.sample_rate = sample_rate
        self.max_duration_s = max_duration_s

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        ex = self.examples[idx]
        arr = _decode_audio(ex["audio"], self.sample_rate, self.max_duration_s)
        audio = torch.from_numpy(arr)
        text = ex["text"].lower()
        phones = text_to_phones(text, self.lexicon)
        phone_ids = self.tokenizer.encode(phones)
        speaker_idx = self.speaker_to_idx[ex["speaker_id"]]
        return audio, phone_ids, speaker_idx, text


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dis_dataloaders(cfg):
    """Build train / val DataLoaders plus tokenizer and speaker_to_idx.

    Returns
    -------
    tokenizer      : PhoneTokenizer
    speaker_to_idx : Dict[int, int]  raw_speaker_id → local 0-indexed label
    train_dl       : DataLoader
    val_dl         : DataLoader
    """
    tokenizer = PhoneTokenizer()
    cfg.vocab_size = tokenizer.vocab_size   # 41

    lexicon = _get_lexicon(cfg.lexicon_path)
    n_train = cfg.max_train_examples if cfg.max_train_examples > 0 else None
    n_val   = cfg.max_val_examples   if cfg.max_val_examples   > 0 else None

    print("[dis_data] loading train-clean-100 …")
    trn_ex = _stream_examples("train.100", str(cfg.librispeech_cache_dir), n_train)

    print("[dis_data] loading dev-clean …")
    val_ex = _stream_examples("validation", str(cfg.librispeech_cache_dir), n_val)

    # Build speaker map from training examples only.
    train_speakers = sorted(set(ex["speaker_id"] for ex in trn_ex))
    speaker_to_idx = {spk: i for i, spk in enumerate(train_speakers)}
    cfg.num_speakers = len(train_speakers)
    print(f"[dis_data] {len(train_speakers)} speakers in training subset")

    kw = dict(sample_rate=cfg.sample_rate, max_duration_s=cfg.max_duration_s)
    train_ds = DISDataset(trn_ex, tokenizer, speaker_to_idx, lexicon, **kw)
    val_ds   = DISDataset(val_ex,  tokenizer, speaker_to_idx, lexicon, **kw)

    print(
        f"[dis_data]  train={len(train_ds)}  val={len(val_ds)}"
        f"  vocab={tokenizer.vocab_size}  speakers={cfg.num_speakers}"
    )

    pin = torch.cuda.is_available()
    loader_kw = dict(collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=pin)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,      shuffle=True,  **loader_kw)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.eval_batch_size, shuffle=False, **loader_kw)
    return tokenizer, speaker_to_idx, train_dl, val_dl
