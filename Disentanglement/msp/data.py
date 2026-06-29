"""MSP-Podcast 2.0 stage-2 dataloaders — one dataset carrying content + speaker +
emotion, so every batch trains PR (CTC), SID, prosody and emotion together.

This replaces the LibriSpeech + IEMOCAP-every-8 arrangement.  PR targets come from
the HUMAN transcript via the SAME data.dataset.text_to_phones() pipeline used for
LibriSpeech (lexicon + g2p -> SUPERB ARPABET), NOT from the ForceAligned phones
tier (that tier mixes ARPABET and IPA across files).  CTC is alignment-free, so the
transcript is all we need.

Batch contract (a dict, since this is multi-task):
    audios          (B, T)        padded waveforms
    audio_lengths   (B,)
    targets         (B, P)        padded phone ids (CTC)
    target_lengths  (B,)
    speaker_ids     (B,)          closed-set index 0..num_speakers-1  (-1 = unseen)
    emotion         (B,)          0=neutral 1=happy 2=sad 3=angry  (IEMOCAP order)
    avd             (B, 3)        arousal, valence, dominance (1..7)  [optional head]
    pert_audios     (B, T)        speaker-perturbed copy — TRAIN only, invariance on

Splits (from data/msp_subset/manifest.csv): train / val / test are the SAME
speakers (closed-set; PR+SID+emotion all scorable).  test_unseen, if present, is a
secondary emotion-generalization set (speaker_id=-1) and is returned as a 4th loader.
"""
from __future__ import annotations

import csv
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
try:
    from training_runtime import StatefulRandomSampler
except ImportError:  # pragma: no cover
    from Disentanglement.training_runtime import StatefulRandomSampler

from data.dataset import PhoneTokenizer, text_to_phones, _get_lexicon

EMOTION_NAMES = ("neutral", "happy", "sad", "angry")   # idx 0..3 (matches manifest)

# Non-verbal / uncertainty markers to drop before phonemizing:  [inaudible],
# [crosstalk], [guess 00:04:09], (laughing), (singing), ... .  text_to_phones()
# then handles per-word punctuation/casing itself.
_MARKER_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_WORD_RE   = re.compile(r"[a-z]", re.I)


def _clean_transcript(t: str) -> str:
    return _MARKER_RE.sub(" ", t).strip()


def _load_audio(path: Path, sr: int) -> np.ndarray:
    arr, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if file_sr != sr:
        import librosa
        arr = librosa.resample(arr, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _read_manifest(manifest: Path) -> List[dict]:
    with open(manifest, newline="") as f:
        return list(csv.DictReader(f))


def _preload_transcripts(file_names, transcripts: Path) -> Dict[str, str]:
    """Read only the needed transcripts once (zip random-access or a dir), so
    DataLoader workers inherit them copy-on-write — no per-item zip handle."""
    want = {fn.replace(".wav", ".txt") for fn in file_names}
    out: Dict[str, str] = {}
    if transcripts.suffix == ".zip":
        with zipfile.ZipFile(transcripts) as z:
            members = {Path(n).name: n for n in z.namelist() if n.endswith(".txt")}
            for base in want:
                m = members.get(base)
                if m is not None:
                    out[base] = z.read(m).decode("utf-8", "ignore")
    else:
        for base in want:
            p = transcripts / base
            if not p.exists():                       # maybe nested under Transcripts/
                p = transcripts / "Transcripts" / base
            if p.exists():
                out[base] = p.read_text(encoding="utf-8", errors="ignore")
    return out


class MSPDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer: PhoneTokenizer, lexicon: Dict,
                 transcripts: Dict[str, str], audio_root: Path, sample_rate: int,
                 perturb: bool = False, perturb_kwargs: Optional[dict] = None) -> None:
        self.tokenizer = tokenizer
        self.lexicon = lexicon
        self.transcripts = transcripts
        self.audio_root = Path(audio_root)
        self.sample_rate = sample_rate
        self.perturb = perturb
        self.perturb_kwargs = perturb_kwargs or {}
        # Drop rows with no usable transcript (would yield empty CTC targets).
        kept, dropped = [], 0
        for r in rows:
            txt = _clean_transcript(transcripts.get(r["FileName"].replace(".wav", ".txt"), ""))
            if _WORD_RE.search(txt):
                kept.append(r)
            else:
                dropped += 1
        self.rows = kept
        if dropped:
            print(f"[msp_data] dropped {dropped} rows with empty/non-verbal transcript")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        arr = _load_audio(self.audio_root / r["wav"], self.sample_rate)
        txt = _clean_transcript(self.transcripts.get(r["FileName"].replace(".wav", ".txt"), ""))
        phone_ids = self.tokenizer.encode(text_to_phones(txt.lower(), self.lexicon))
        item = {
            "audio": torch.from_numpy(arr),
            "n": arr.shape[0],
            "phones": phone_ids,
            "speaker": int(r["speaker_idx"]),
            "emotion": int(r["emotion_idx"]),
            "avd": torch.tensor([float(r["Act"] or 0), float(r["Val"] or 0),
                                 float(r["Dom"] or 0)], dtype=torch.float32),
        }
        if self.perturb:
            from data.perturb import perturb_speaker
            item["pert"] = torch.from_numpy(
                perturb_speaker(arr, self.sample_rate, **self.perturb_kwargs))
        return item


def collate_msp(batch: List[dict]) -> dict:
    has_pert = "pert" in batch[0]
    out = {
        "audios":         pad_sequence([b["audio"] for b in batch], batch_first=True),
        "audio_lengths":  torch.tensor([b["n"] for b in batch], dtype=torch.long),
        "targets":        pad_sequence([b["phones"] for b in batch], batch_first=True,
                                       padding_value=0),
        "target_lengths": torch.tensor([b["phones"].size(0) for b in batch], dtype=torch.long),
        "speaker_ids":    torch.tensor([b["speaker"] for b in batch], dtype=torch.long),
        "emotion":        torch.tensor([b["emotion"] for b in batch], dtype=torch.long),
        "avd":            torch.stack([b["avd"] for b in batch]),
    }
    if has_pert:
        out["pert_audios"] = pad_sequence([b["pert"] for b in batch], batch_first=True)
    return out


def make_msp_dataloaders(cfg):
    """Returns (tokenizer, train_dl, val_dl, test_dl[, test_unseen_dl]).

    Reads cfg.msp_manifest (dir or manifest.csv) + cfg.msp_audio_root + the human
    transcripts (cfg.msp_transcripts, a dir or Transcripts.zip).  Sets cfg.vocab_size,
    cfg.num_speakers, cfg.emotion_num_classes.
    """
    tokenizer = PhoneTokenizer()
    cfg.vocab_size = tokenizer.vocab_size
    lexicon = _get_lexicon(cfg.lexicon_path)

    manifest = Path(cfg.msp_manifest)
    if manifest.is_dir():
        manifest = manifest / "manifest.csv"
    rows = _read_manifest(manifest)

    by_split: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)

    # Closed-set SID label space from the SEEN speakers (train/val/test share them).
    seen = sorted({int(r["speaker_idx"]) for r in rows if int(r["speaker_idx"]) >= 0})
    cfg.num_speakers = len(seen)
    cfg.emotion_num_classes = getattr(cfg, "emotion_num_classes", 4)
    print(f"[msp_data] manifest={manifest}")
    print(f"[msp_data] speakers={cfg.num_speakers}  vocab={cfg.vocab_size}  "
          f"emotions={EMOTION_NAMES}")
    for s in ("train", "val", "test", "test_unseen"):
        if by_split.get(s):
            cc = Counter(int(r["emotion_idx"]) for r in by_split[s])
            dist = {EMOTION_NAMES[k]: cc[k] for k in sorted(cc)}
            print(f"[msp_data]  {s:11s} {len(by_split[s]):6d}  {dist}")

    transcripts = Path(getattr(cfg, "msp_transcripts",
                               "/rds/project/rds-xyBFuSj0hm0/dataset/"
                               "MSP-Podcast-2.0/Transcripts.zip"))
    text = _preload_transcripts((r["FileName"] for r in rows), transcripts)
    print(f"[msp_data] preloaded {len(text)} transcripts from {transcripts.name}")

    inv_on = getattr(cfg, "invariance", False)
    pk = dict(f0_range=(getattr(cfg, "inv_f0_low", 0.7), getattr(cfg, "inv_f0_high", 1.5)),
              formant_range=(getattr(cfg, "inv_formant_low", 0.85),
                             getattr(cfg, "inv_formant_high", 1.3)))
    audio_root = Path(cfg.msp_audio_root)
    common = dict(tokenizer=tokenizer, lexicon=lexicon, transcripts=text,
                  audio_root=audio_root, sample_rate=cfg.sample_rate)

    train_ds = MSPDataset(by_split["train"], perturb=inv_on, perturb_kwargs=pk, **common)
    val_ds   = MSPDataset(by_split["val"], **common)
    test_ds  = MSPDataset(by_split["test"], **common)
    if inv_on:
        print(f"[msp_data] invariance ON — train yields perturbed pairs "
              f"(f0×{pk['f0_range']}, formant×{pk['formant_range']})")

    pin = torch.cuda.is_available()
    kw = dict(num_workers=cfg.num_workers, pin_memory=pin, collate_fn=collate_msp)
    loaders = [
        tokenizer,
        DataLoader(train_ds, batch_size=cfg.batch_size,
                   sampler=StatefulRandomSampler(train_ds, cfg.seed), drop_last=True, **kw),
        DataLoader(val_ds,   batch_size=cfg.eval_batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=cfg.eval_batch_size, shuffle=False, **kw),
    ]
    if by_split.get("test_unseen"):
        tu_ds = MSPDataset(by_split["test_unseen"], **common)
        loaders.append(DataLoader(tu_ds, batch_size=cfg.eval_batch_size,
                                  shuffle=False, **kw))
    return tuple(loaders)
