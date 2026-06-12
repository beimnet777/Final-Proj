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


# ---------------------------------------------------------------- SUPERB phones
# 74-token CTC vocab, IDENTICAL to Probing/pr (SUPERB PR): 3 special
# (<pad>=blank/<eos>/<unk>) + 71 stress-marked CMU phones. Training PR now uses
# the same vocab as the SUPERB probe, so train/test phone sets always match.

SUPERB_PHONES = [
    "SIL", "SPN",
    "AA0", "AA1", "AA2",
    "AE0", "AE1", "AE2",
    "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2",
    "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2",
    "ER0", "ER1", "ER2",
    "EY0", "EY1", "EY2",
    "F", "G", "HH",
    "IH0", "IH1", "IH2",
    "IY0", "IY1", "IY2",
    "JH", "K", "L", "M", "N", "NG",
    "OW0", "OW1", "OW2",
    "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH",
    "UH0", "UH1", "UH2",
    "UW0", "UW1", "UW2",
    "V", "W", "Y", "Z", "ZH",
]  # 71 entries → indices 3..73

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


_PHONE_VOCAB = set(SUPERB_PHONES)


def text_to_phones(text: str, lexicon: Dict) -> List[str]:
    """Transcript → raw CMU ARPAbet phones (stress digits kept), SUPERB scheme.
    Mirrors Probing/pr/pr_data.text_to_phones so training and probing agree."""
    phones: List[str] = []
    g2p = _get_g2p()
    for word in text.upper().split():
        word = word.strip("'-.,!?;:")
        if not word:
            continue
        if word in lexicon:
            phones.extend(p for p in lexicon[word] if p in _PHONE_VOCAB)
        elif g2p:
            converted = [p for p in g2p(word) if p in _PHONE_VOCAB]
            phones.extend(converted if converted else ["SPN"])
        else:
            phones.append("SPN")
    return phones


# ---------------------------------------------------------------- PhoneTokenizer

class PhoneTokenizer:
    """SUPERB 74-token CTC set: 0=<pad>(blank), 1=<eos>, 2=<unk>, 3..73=phones."""
    BLANK_ID = 0
    UNK_ID   = 2

    def __init__(self) -> None:
        all_tokens = ["<pad>", "<eos>", "<unk>"] + SUPERB_PHONES   # 74 total
        self.phone_to_id = {p: i for i, p in enumerate(all_tokens)}
        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}
        self.vocab_size  = len(all_tokens)   # 74

    def encode(self, phones: List[str]) -> torch.Tensor:
        return torch.tensor(
            [self.phone_to_id.get(p, self.UNK_ID) for p in phones],
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

    # Closed-set SID needs the same speakers in every split, so split by
    # UTTERANCE: first n_test → test, next n_val → val, rest → train. (PR probing
    # uses dev/test-clean instead — see make_pr_eval_dataloaders.)
    n_val  = cfg.max_val_examples if cfg.max_val_examples > 0 else 0
    n_test = getattr(cfg, "max_test_examples", n_val)
    tst_ex = all_ex[:n_test]
    val_ex = all_ex[n_test:n_test + n_val]
    trn_ex = all_ex[n_test + n_val:] if (n_test + n_val) < len(all_ex) else all_ex

    kw_ds = dict(tokenizer=tokenizer, speaker_to_idx=speaker_to_idx,
                 lexicon=lexicon, sample_rate=cfg.sample_rate)
    train_ds = Stage2Dataset(trn_ex, **kw_ds)
    val_ds   = Stage2Dataset(val_ex, **kw_ds)
    test_ds  = Stage2Dataset(tst_ex, **kw_ds)
    print(f"[dis_data]  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  "
          f"vocab={cfg.vocab_size}  speakers={cfg.num_speakers}")

    pin = torch.cuda.is_available()
    kw  = dict(num_workers=cfg.num_workers, pin_memory=pin)
    return (
        tokenizer,
        DataLoader(train_ds, batch_size=cfg.batch_size,      shuffle=True,  collate_fn=collate_stage2, **kw),
        DataLoader(val_ds,   batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_stage2, **kw),
        DataLoader(test_ds,  batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_stage2, **kw),
    )
