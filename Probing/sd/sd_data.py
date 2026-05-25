"""Dataset parsers and DataLoader factory for Spoof Detection.

Label convention throughout:  1 = bonafide,  0 = spoof

Parsers
-------
parse_asv19_protocol  — ASVspoof 2019 LA train/dev/eval .txt files
parse_asv21_keys      — ASVspoof 2021 LA / DF eval keys file
parse_itw             — In-The-Wild (CSV or directory layout)
parse_generic_keys    — generic  "utt_id label" .txt  (DFEval24, FamousFigs, LD)

Each parser returns List[Record] = List[Tuple[Path, int]].
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from sd_config import SDConfig

Record = Tuple[Path, int]   # (audio_path, label)  label: 1=bonafide 0=spoof


# ====================================================================
# Audio loading helper
# ====================================================================

def _load_audio(path: Path, target_sr: int) -> torch.Tensor:
    """Load audio file → 1-D float32 tensor at target_sr."""
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.mean(0)   # stereo → mono if needed


# ====================================================================
# Protocol / keys parsers
# ====================================================================

def parse_asv19_protocol(protocol_file: Path, audio_dir: Path) -> List[Record]:
    """ASVspoof 2019 LA  protocol format.

    Columns (space-separated): speaker  utt_id  -  attack_type  label
    label ∈ {bonafide, spoof}
    Audio: audio_dir/{utt_id}.flac
    """
    records: List[Record] = []
    with open(protocol_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utt_id    = parts[1]
            label_str = parts[4]
            label     = 1 if label_str == "bonafide" else 0
            wav_path  = audio_dir / f"{utt_id}.flac"
            if wav_path.exists():
                records.append((wav_path, label))
    return records


def parse_asv21_keys(keys_file: Path, audio_dir: Path) -> List[Record]:
    """ASVspoof 2021 LA / DF eval keys file.

    Typical format: utt_id  -  -  attack_type  label
    OR just:        utt_id  label
    The label field is always the last token on the line.
    Audio may be directly in audio_dir or in audio_dir/flac/.
    """
    records: List[Record] = []
    with open(keys_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            utt_id    = parts[0]
            label_str = parts[-1]
            if label_str not in ("bonafide", "spoof"):
                continue
            label = 1 if label_str == "bonafide" else 0
            # Search common locations
            for candidate in [
                audio_dir / f"{utt_id}.flac",
                audio_dir / "flac" / f"{utt_id}.flac",
                audio_dir / f"{utt_id}.wav",
                audio_dir / "wav"  / f"{utt_id}.wav",
            ]:
                if candidate.exists():
                    records.append((candidate, label))
                    break
    return records


def parse_itw(itw_root: Path) -> List[Record]:
    """In-The-Wild dataset.

    Supports two layouts:
    1. CSV-based: itw_root/meta.csv (or any *.csv) with columns
       file/filename/path + label/class
    2. Directory-based: itw_root/{bonafide,bona-fide,real}/… and
       itw_root/{spoof,fake}/…
    """
    records: List[Record] = []

    # ── CSV layout ───────────────────────────────────────────────────────────
    csv_candidates = sorted(
        list(itw_root.glob("meta*.csv")) + list(itw_root.glob("*.csv"))
    )
    if csv_candidates:
        with open(csv_candidates[0], newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname     = (row.get("file") or row.get("filename")
                             or row.get("path") or "").strip()
                label_str = (row.get("label") or row.get("class")
                             or row.get("type") or "").strip().lower()
                if not fname or not label_str:
                    continue
                label    = 1 if ("bona" in label_str or "real" in label_str
                                 or "genuine" in label_str) else 0
                wav_path = itw_root / fname
                if not wav_path.exists():
                    wav_path = itw_root / "audio" / fname
                if wav_path.exists():
                    records.append((wav_path, label))
        if records:
            return records

    # ── Directory layout ─────────────────────────────────────────────────────
    dir_map = {
        "bonafide": 1, "bona-fide": 1, "real": 1, "genuine": 1,
        "spoof": 0, "fake": 0, "deepfake": 0,
    }
    for subdir_name, label in dir_map.items():
        subdir = itw_root / subdir_name
        if not subdir.is_dir():
            continue
        for ext in ("*.flac", "*.wav", "*.mp3"):
            for wav_path in sorted(subdir.rglob(ext)):
                records.append((wav_path, label))
    return records


def parse_generic_keys(keys_file: Path, audio_dir: Path) -> List[Record]:
    """Generic two-column keys file: utt_id  label

    label ∈ {bonafide, spoof} or {genuine, fake} or {1, 0}.
    """
    records: List[Record] = []
    with open(keys_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt_id    = parts[0]
            label_str = parts[-1].lower()
            if label_str in ("bonafide", "genuine", "real", "1", "true"):
                label = 1
            elif label_str in ("spoof", "fake", "deepfake", "0", "false"):
                label = 0
            else:
                continue
            for candidate in [
                audio_dir / f"{utt_id}.flac",
                audio_dir / f"{utt_id}.wav",
                audio_dir / "flac" / f"{utt_id}.flac",
                audio_dir / "wav"  / f"{utt_id}.wav",
            ]:
                if candidate.exists():
                    records.append((candidate, label))
                    break
    return records


# ====================================================================
# Dataset
# ====================================================================

class SpoofDataset(Dataset):
    """Returns (waveform, num_samples, label) for each utterance."""

    def __init__(self, records: List[Record], sample_rate: int) -> None:
        self.records     = records
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        path, label = self.records[idx]
        wav = _load_audio(path, self.sample_rate)   # (T,)
        return wav, wav.shape[0], label


# ====================================================================
# Collate
# ====================================================================

def _collate(batch):
    waveforms, lengths, labels = zip(*batch)
    max_len = max(lengths)
    padded  = torch.zeros(len(waveforms), max_len)
    for i, (wav, l) in enumerate(zip(waveforms, lengths)):
        padded[i, :l] = wav
    return (
        padded,
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels,  dtype=torch.long),
    )


# ====================================================================
# ASVspoof 2019 LA DataLoader factory (train + val)
# ====================================================================

def make_sd_train_dataloaders(cfg: SDConfig):
    """Build train and val DataLoaders from ASVspoof 2019 LA.

    Returns (train_dl, val_dl).
    """
    root      = Path(cfg.asv19_la_root)
    proto_dir = root / "ASVspoof2019_LA_cm_protocols"

    # locate protocol files tolerantly (name varies slightly across releases)
    def _find_proto(keyword: str) -> Path:
        candidates = sorted(proto_dir.glob(f"*{keyword}*"))
        if not candidates:
            raise FileNotFoundError(
                f"No protocol file matching '*{keyword}*' in {proto_dir}"
            )
        return candidates[0]

    train_proto = _find_proto("train")
    dev_proto   = _find_proto("dev")

    train_audio = root / "ASVspoof2019_LA_train" / "flac"
    dev_audio   = root / "ASVspoof2019_LA_dev"   / "flac"

    train_recs = parse_asv19_protocol(train_proto, train_audio)
    val_recs   = parse_asv19_protocol(dev_proto,   dev_audio)

    print(
        f"[sd_data] ASV19 LA  train={len(train_recs)}  val={len(val_recs)}"
        f"  bonafide_train={(sum(l for _,l in train_recs))}"
    )

    train_ds = SpoofDataset(train_recs, cfg.sample_rate)
    val_ds   = SpoofDataset(val_recs,   cfg.sample_rate)

    pin = torch.cuda.is_available()
    kw  = dict(collate_fn=_collate, num_workers=cfg.num_workers, pin_memory=pin)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,      shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.eval_batch_size, shuffle=False, **kw)
    return train_dl, val_dl


# ====================================================================
# Test DataLoader builders (one per eval dataset)
# ====================================================================

def _make_test_dl(records: List[Record], cfg: SDConfig) -> DataLoader:
    ds = SpoofDataset(records, cfg.sample_rate)
    return DataLoader(
        ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    )


def make_test_dataloaders(cfg: SDConfig) -> dict:
    """Build a {name: DataLoader} dict for every configured test dataset.

    Datasets whose root is None or whose path does not exist are skipped.
    The ASVspoof 2019 eval set is always included if asv19_la_root exists.
    """
    test_dls: dict = {}
    root = Path(cfg.asv19_la_root)

    # ── ASV19 eval ───────────────────────────────────────────────────────────
    proto_dir = root / "ASVspoof2019_LA_cm_protocols"
    eval_audio = root / "ASVspoof2019_LA_eval" / "flac"
    eval_protos = sorted(proto_dir.glob("*eval*"))
    if eval_protos and eval_audio.is_dir():
        recs = parse_asv19_protocol(eval_protos[0], eval_audio)
        if recs:
            test_dls["asv19_eval"] = _make_test_dl(recs, cfg)
            print(f"[sd_data] asv19_eval      : {len(recs)} utterances")

    # ── ASV21 LA ─────────────────────────────────────────────────────────────
    if cfg.asv21_la_root and cfg.asv21_la_keys:
        r = Path(cfg.asv21_la_root); k = Path(cfg.asv21_la_keys)
        if r.exists() and k.exists():
            recs = parse_asv21_keys(k, r)
            if recs:
                test_dls["asv21_la"] = _make_test_dl(recs, cfg)
                print(f"[sd_data] asv21_la        : {len(recs)} utterances")

    # ── ASV21 DF ─────────────────────────────────────────────────────────────
    if cfg.asv21_df_root and cfg.asv21_df_keys:
        r = Path(cfg.asv21_df_root); k = Path(cfg.asv21_df_keys)
        if r.exists() and k.exists():
            recs = parse_asv21_keys(k, r)
            if recs:
                test_dls["asv21_df"] = _make_test_dl(recs, cfg)
                print(f"[sd_data] asv21_df        : {len(recs)} utterances")

    # ── ITW ──────────────────────────────────────────────────────────────────
    if cfg.itw_root and Path(cfg.itw_root).exists():
        recs = parse_itw(Path(cfg.itw_root))
        if recs:
            test_dls["itw"] = _make_test_dl(recs, cfg)
            print(f"[sd_data] itw             : {len(recs)} utterances")

    # ── DFEval 2024 ──────────────────────────────────────────────────────────
    if cfg.dfeval24_root and cfg.dfeval24_keys:
        r = Path(cfg.dfeval24_root); k = Path(cfg.dfeval24_keys)
        if r.exists() and k.exists():
            recs = parse_generic_keys(k, r)
            if recs:
                test_dls["dfeval24"] = _make_test_dl(recs, cfg)
                print(f"[sd_data] dfeval24        : {len(recs)} utterances")

    # ── Famous Figures ───────────────────────────────────────────────────────
    if cfg.famous_figures_root and cfg.famous_figures_keys:
        r = Path(cfg.famous_figures_root); k = Path(cfg.famous_figures_keys)
        if r.exists() and k.exists():
            recs = parse_generic_keys(k, r)
            if recs:
                test_dls["famous_figures"] = _make_test_dl(recs, cfg)
                print(f"[sd_data] famous_figures  : {len(recs)} utterances")

    # ── ASVSpoofLD ───────────────────────────────────────────────────────────
    if cfg.asvspoofld_root and cfg.asvspoofld_keys:
        r = Path(cfg.asvspoofld_root); k = Path(cfg.asvspoofld_keys)
        if r.exists() and k.exists():
            recs = parse_generic_keys(k, r)
            if recs:
                test_dls["asvspoofld"] = _make_test_dl(recs, cfg)
                print(f"[sd_data] asvspoofld      : {len(recs)} utterances")

    print(f"[sd_data] {len(test_dls)} test set(s) configured.")
    return test_dls
