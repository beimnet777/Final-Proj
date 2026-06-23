"""Pair datasets for dual-invariance training.

Provides PairAlphaDataset (same content, paralinguistic varies — for L_inv_L)
and PairBetaDataset (different content, same speaker+session — for L_inv_P).

Pair α sources:
    * CMU ARCTIC — same prompt read by multiple speakers (natural parallel data)
    * LibriSpeech with on-the-fly speaker perturbation (frame-aligned synthetic)

Pair β sources:
    * LibriSpeech within-chapter utterance pairs (same speaker, same session)

Returns torch.Tensor waveforms at cfg.sample_rate; collation handled by
`collate_pairs`.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------- audio I/O
def _read_audio(path: str | Path, target_sr: int) -> np.ndarray:
    arr, sr = sf.read(str(path))
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    if sr != target_sr:
        import librosa
        arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    return arr.astype(np.float32)


# ---------------------------------------------------------------- ARCTIC index
class ARCTICIndex:
    """Index of CMU ARCTIC: maps prompt-id -> [(speaker, wav_path), ...].

    Default layout (`festvox.org` release):
        <root>/cmu_us_<spk>_arctic/wav/arctic_aXXXX.wav
        <root>/cmu_us_<spk>_arctic/etc/txt.done.data  (prompt transcripts)

    `prompt_id` is the basename without extension (e.g. "arctic_a0023"), so
    cross-speaker pairs share the same id.
    """

    _SPK_RE = re.compile(r"cmu_us_([a-z0-9]+)_arctic")

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        # prompt_id -> {speaker: wav_path}
        self.prompts: Dict[str, Dict[str, Path]] = {}
        if not self.root.exists():
            self.speakers: List[str] = []
            return
        speakers = set()
        for spk_dir in sorted(self.root.iterdir()):
            if not spk_dir.is_dir():
                continue
            m = self._SPK_RE.match(spk_dir.name)
            if not m:
                continue
            spk = m.group(1)
            wav_dir = spk_dir / "wav"
            if not wav_dir.is_dir():
                continue
            speakers.add(spk)
            for wav in sorted(wav_dir.glob("arctic_*.wav")):
                pid = wav.stem
                self.prompts.setdefault(pid, {})[spk] = wav
        self.speakers = sorted(speakers)

    def __len__(self) -> int:
        return len(self.prompts)

    def pair_pool(self) -> List[Tuple[str, str, str, Path, Path]]:
        """All cross-speaker pairs as (prompt_id, spk_a, spk_b, path_a, path_b)."""
        pool: List[Tuple[str, str, str, Path, Path]] = []
        for pid, by_spk in self.prompts.items():
            spks = sorted(by_spk.keys())
            for i in range(len(spks)):
                for j in range(i + 1, len(spks)):
                    sa, sb = spks[i], spks[j]
                    pool.append((pid, sa, sb, by_spk[sa], by_spk[sb]))
        return pool


# ---------------------------------------------------------------- LibriSpeech chapter index
class LibrispeechChapterIndex:
    """Group existing LibriSpeech examples by (speaker_id, chapter_id).

    LibriSpeech utterance id format: "<speaker>-<chapter>-<utterance>".
    `examples` is the list produced by `_local_examples()` (with "id" field).
    """

    def __init__(self, examples: List[dict]) -> None:
        # (speaker, chapter) -> [example_idx, ...]
        self.by_chapter: Dict[Tuple[int, int], List[int]] = {}
        for i, ex in enumerate(examples):
            uid = ex.get("id", "")
            parts = uid.split("-")
            if len(parts) < 3:
                continue
            try:
                spk, chap = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            self.by_chapter.setdefault((spk, chap), []).append(i)
        self.examples = examples
        self.keys = [k for k, v in self.by_chapter.items() if len(v) >= 2]

    def __len__(self) -> int:
        return len(self.keys)

    def sample_pair(self, rng: random.Random) -> Tuple[dict, dict]:
        key = rng.choice(self.keys)
        idxs = self.by_chapter[key]
        a, b = rng.sample(idxs, 2)
        return self.examples[a], self.examples[b]


# ---------------------------------------------------------------- Pair α dataset
class PairAlphaDataset(Dataset):
    """Yields (wav_a, wav_b) pair-α examples with weighted source mix.

    Sources implemented:
        - "arctic":     ARCTICIndex.pair_pool() — natural cross-speaker pairs
        - "perturb":    LibriSpeech utt + perturb_speaker(utt) — synthetic
                        frame-aligned pair

    `source_weights` is a dict like {"arctic": 0.6, "perturb": 0.4}; values are
    normalised internally and used per-__getitem__ to pick a source.
    """

    def __init__(
        self,
        arctic_index: Optional[ARCTICIndex],
        libri_examples: Optional[List[dict]],
        sample_rate: int,
        source_weights: Dict[str, float],
        perturb_kwargs: Optional[dict] = None,
        rng_seed: int = 0,
        epoch_size: int = 100_000,
    ) -> None:
        self.sample_rate = sample_rate
        self.epoch_size  = epoch_size
        self.rng = random.Random(rng_seed)
        self.perturb_kwargs = perturb_kwargs or {}
        self.arctic_pool = arctic_index.pair_pool() if arctic_index is not None else []
        self.libri_examples = libri_examples or []
        # Filter source_weights to sources that actually have data
        clean: Dict[str, float] = {}
        if self.arctic_pool and source_weights.get("arctic", 0) > 0:
            clean["arctic"] = float(source_weights["arctic"])
        if self.libri_examples and source_weights.get("perturb", 0) > 0:
            clean["perturb"] = float(source_weights["perturb"])
        if not clean:
            raise ValueError("PairAlphaDataset: no source has data")
        s = sum(clean.values())
        self.sources = list(clean.keys())
        self.weights = [v / s for v in clean.values()]

    def __len__(self) -> int:
        return self.epoch_size

    def _sample_arctic(self) -> Tuple[np.ndarray, np.ndarray]:
        pid, sa, sb, pa, pb = self.rng.choice(self.arctic_pool)
        return _read_audio(pa, self.sample_rate), _read_audio(pb, self.sample_rate)

    def _sample_perturb(self) -> Tuple[np.ndarray, np.ndarray]:
        from .perturb import perturb_speaker
        ex = self.rng.choice(self.libri_examples)
        arr = _read_audio(ex["audio"]["path"], self.sample_rate)
        pert = perturb_speaker(arr, self.sample_rate, **self.perturb_kwargs)
        return arr, pert

    def __getitem__(self, idx: int):
        src = self.rng.choices(self.sources, weights=self.weights, k=1)[0]
        if src == "arctic":
            a, b = self._sample_arctic()
        else:
            a, b = self._sample_perturb()
        return (torch.from_numpy(a), a.shape[0],
                torch.from_numpy(b), b.shape[0],
                src)


# ---------------------------------------------------------------- Pair β dataset
class PairBetaDataset(Dataset):
    """Yields (wav_a, wav_b) pair-β examples (same speaker+session, different content).

    Source implemented:
        - "libri": LibrispeechChapterIndex — within-chapter pairs
    """

    def __init__(
        self,
        libri_chapter_index: LibrispeechChapterIndex,
        sample_rate: int,
        rng_seed: int = 0,
        epoch_size: int = 100_000,
    ) -> None:
        if len(libri_chapter_index) == 0:
            raise ValueError("PairBetaDataset: chapter index empty")
        self.idx = libri_chapter_index
        self.sample_rate = sample_rate
        self.epoch_size  = epoch_size
        self.rng = random.Random(rng_seed)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int):
        ex_a, ex_b = self.idx.sample_pair(self.rng)
        a = _read_audio(ex_a["audio"]["path"], self.sample_rate)
        b = _read_audio(ex_b["audio"]["path"], self.sample_rate)
        return (torch.from_numpy(a), a.shape[0],
                torch.from_numpy(b), b.shape[0],
                "libri")


# ---------------------------------------------------------------- collate
def collate_pairs(batch):
    """Pad audio_a and audio_b independently; return tensors and length vectors.

    Batch item: (wav_a, len_a, wav_b, len_b, source_tag)
    Returns: dict with keys
        audio_a (B, T_a), len_a (B,), audio_b (B, T_b), len_b (B,), sources (list[str])
    """
    from torch.nn.utils.rnn import pad_sequence
    wa, la, wb, lb, src = zip(*batch)
    return {
        "audio_a":  pad_sequence(list(wa), batch_first=True, padding_value=0.0),
        "len_a":    torch.tensor(list(la), dtype=torch.long),
        "audio_b":  pad_sequence(list(wb), batch_first=True, padding_value=0.0),
        "len_b":    torch.tensor(list(lb), dtype=torch.long),
        "sources":  list(src),
    }
