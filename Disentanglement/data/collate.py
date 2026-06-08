"""Collate functions for stage 1 (audio only) and stage 2 (audio + labels)."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_stage1(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch item: (waveform, n_samples)

    Returns: audios (B, T_max), audio_lengths (B,)
    """
    audios, lengths = zip(*batch)
    return (
        pad_sequence(list(audios), batch_first=True, padding_value=0.0),
        torch.tensor(list(lengths), dtype=torch.long),
    )


def collate_stage2(batch: List[Tuple]) -> Tuple:
    """Batch item: (waveform, n_samples, phone_ids, speaker_idx)

    Returns: audios (B, T_max), audio_lengths (B,),
             targets (B, P_max), target_lengths (B,), speaker_ids (B,)
    """
    audios, lengths, phone_ids, speaker_idxs = zip(*batch)
    return (
        pad_sequence(list(audios),    batch_first=True, padding_value=0.0),
        torch.tensor(list(lengths),       dtype=torch.long),
        pad_sequence(list(phone_ids), batch_first=True, padding_value=0),
        torch.tensor([p.size(0) for p in phone_ids], dtype=torch.long),
        torch.tensor(list(speaker_idxs),  dtype=torch.long),
    )
