"""LibriSpeech datasets for the disentanglement system.

Stage 1 : audio only  → make_stage1_dataloaders(cfg)
Stage 2 : audio + phone labels + speaker IDs  → make_stage2_dataloaders(cfg)
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

from .collate import collate_stage1, collate_stage2


# ---------------------------------------------------------------- ARPAbet

ARPABET_39 = [
    "aa","ae","ah","ao","aw","ay",
    "b","ch","d","dh",
    "eh","er","ey",
    "f","g","hh",
    "ih","iy","jh",
    "k","l","m","n","ng",
    "ow","oy","p","r",
    "s","sh","t","th",
    "uh","uw","v","w","y","z","zh",
]
SPN_TOKEN = "spn"

_CMU_REMAP = {
    "ax":"ah","axr":"er","ix":"ih","ux":"uw",
    "el":"l","em":"m","en":"n","nx":"ng",
}

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
        print(f"[dis_data] loaded lexicon: {len(_LEXICON):,} entries")
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
            print("[dis_data] g2p_en not found — OOV words → SPN")
    return _G2P


def _normalise_phone(raw: str) -> str:
    return _CMU_REMAP.get(raw.lower().rstrip("012"), raw.lower().rstrip("012"))


def text_to_phones(text: str, lexicon: Dict) -> List[str]:
    phones: List[str] = []
    g2p = _get_g2p()
    for word in text.upper().split():
        word = word.strip("'-.,!?;:")
        if not word:
            continue
        if word in lexicon:
            phones.extend(_normalise_phone(p) for p in lexicon[word])
        elif g2p:
            raw = g2p(word)
            converted = [_normalise_phone(p) for p in raw if p.strip() and not p.isspace()]
            valid = [p for p in converted if p in set(ARPABET_39)]
            phones.extend(valid if valid else [SPN_TOKEN])
        else:
            phones.append(SPN_TOKEN)
    return phones


# ---------------------------------------------------------------- PhoneTokenizer

class PhoneTokenizer:
    """CTC token set: 0=blank, 1-39=ARPAbet, 40=SPN."""
    BLANK_ID = 0

    def __init__(self) -> None:
        phones = ARPABET_39 + [SPN_TOKEN]
        self.phone_to_id = {p: i + 1 for i, p in enumerate(phones)}
        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}
        self.vocab_size   = 1 + len(phones)   # 41

    def encode(self, phones: List[str]) -> torch.Tensor:
        return torch.tensor(
            [self.phone_to_id.get(p, self.phone_to_id[SPN_TOKEN]) for p in phones],
            dtype=torch.long,
        )


# ---------------------------------------------------------------- audio helper

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
    return list(itertools.islice(ds, n) if n is not None else ds)


# ---------------------------------------------------------------- Stage 1 dataset (audio only)

class Stage1Dataset(Dataset):
    def __init__(self, examples: List[dict], sample_rate: int) -> None:
        self.examples    = examples
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        arr = _decode_audio(self.examples[idx]["audio"], self.sample_rate)
        return torch.from_numpy(arr), arr.shape[0]


# ---------------------------------------------------------------- Stage 2 dataset (audio + labels)

class Stage2Dataset(Dataset):
    def __init__(
        self,
        examples: List[dict],
        tokenizer: PhoneTokenizer,
        speaker_to_idx: Dict[int, int],
        lexicon: Dict,
        sample_rate: int,
    ) -> None:
        self.examples       = [ex for ex in examples if ex["speaker_id"] in speaker_to_idx]
        dropped = len(examples) - len(self.examples)
        if dropped:
            print(f"[dis_data] dropped {dropped} examples with unseen speakers")
        self.tokenizer      = tokenizer
        self.speaker_to_idx = speaker_to_idx
        self.lexicon        = lexicon
        self.sample_rate    = sample_rate

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex          = self.examples[idx]
        arr         = _decode_audio(ex["audio"], self.sample_rate)
        audio       = torch.from_numpy(arr)
        phones      = text_to_phones(ex["text"].lower(), self.lexicon)
        phone_ids   = self.tokenizer.encode(phones)
        speaker_idx = self.speaker_to_idx[ex["speaker_id"]]
        return audio, arr.shape[0], phone_ids, speaker_idx


# ---------------------------------------------------------------- DataLoader factories

def make_stage1_dataloaders(cfg):
    """Audio-only DataLoaders for stage 1 reconstruction."""
    n_train = cfg.max_train_examples if cfg.max_train_examples > 0 else None
    n_val   = cfg.max_val_examples   if cfg.max_val_examples   > 0 else None

    print("[dis_data] loading train-clean-100 …")
    trn_ex = _stream_examples("train.100", str(cfg.librispeech_cache_dir), n_train)
    print("[dis_data] loading dev-clean …")
    val_ex = _stream_examples("validation", str(cfg.librispeech_cache_dir), n_val)

    train_ds = Stage1Dataset(trn_ex, cfg.sample_rate)
    val_ds   = Stage1Dataset(val_ex, cfg.sample_rate)
    print(f"[dis_data]  train={len(train_ds)}  val={len(val_ds)}")

    pin = torch.cuda.is_available()
    kw  = dict(num_workers=cfg.num_workers, pin_memory=pin)
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size,      shuffle=True,  collate_fn=collate_stage1, **kw),
        DataLoader(val_ds,   batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_stage1, **kw),
    )


def make_stage2_dataloaders(cfg):
    """Audio + phone labels + speaker IDs for stage 2.

    Shuffles all examples with a fixed seed, builds speaker_to_idx from ALL examples
    (so val speakers are always known), then splits first n_val as validation.
    This guarantees every speaker appears in both train and val.
    """
    import random as _random

    tokenizer = PhoneTokenizer()
    cfg.vocab_size = tokenizer.vocab_size

    lexicon = _get_lexicon(cfg.lexicon_path)

    print("[dis_data] loading train-clean-100 …")
    n_load = cfg.max_train_examples if cfg.max_train_examples > 0 else None
    all_ex = _stream_examples("train.100", str(cfg.librispeech_cache_dir), n_load)

    # Shuffle with fixed seed so split is reproducible
    rng = _random.Random(42)
    rng.shuffle(all_ex)

    # Build speaker map from ALL examples — val speakers are always known
    all_speakers  = sorted(set(ex["speaker_id"] for ex in all_ex))
    speaker_to_idx = {spk: i for i, spk in enumerate(all_speakers)}
    cfg.num_speakers = len(all_speakers)
    print(f"[dis_data]  {cfg.num_speakers} speakers")

    # Split: first n_val → val, rest → train
    n_val = cfg.max_val_examples if cfg.max_val_examples > 0 else 0
    if n_val > 0 and n_val < len(all_ex):
        val_ex, trn_ex = all_ex[:n_val], all_ex[n_val:]
    else:
        trn_ex, val_ex = all_ex, []

    kw_ds = dict(tokenizer=tokenizer, speaker_to_idx=speaker_to_idx,
                 lexicon=lexicon, sample_rate=cfg.sample_rate)
    train_ds = Stage2Dataset(trn_ex, **kw_ds)
    val_ds   = Stage2Dataset(val_ex, **kw_ds)
    print(f"[dis_data]  train={len(train_ds)}  val={len(val_ds)}  "
          f"vocab={cfg.vocab_size}  speakers={cfg.num_speakers}")

    pin = torch.cuda.is_available()
    kw  = dict(num_workers=cfg.num_workers, pin_memory=pin)
    return (
        tokenizer,
        DataLoader(train_ds, batch_size=cfg.batch_size,      shuffle=True,  collate_fn=collate_stage2, **kw),
        DataLoader(val_ds,   batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_stage2, **kw),
    )
