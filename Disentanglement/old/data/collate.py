"""Collate function for variable-length DIS batches."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(
    batch: List[Tuple],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Pad audio and phone targets; stack speaker IDs.

    Batch item: (waveform, phone_ids, speaker_idx, text)

    Returns
    -------
    audios          (B, T_max)    float32, zero-padded
    audio_lengths   (B,)          long, true sample counts
    targets         (B, P_max)    long, zero-padded phone id sequences
    target_lengths  (B,)          long, true phone sequence lengths
    speaker_ids     (B,)          long, local 0-indexed speaker labels
    texts           list[str]     original transcripts (for debug / PER eval)
    """
    audios, phone_ids, speaker_idxs, texts = zip(*batch)

    audio_lengths  = torch.tensor([a.size(0) for a in audios],     dtype=torch.long)
    target_lengths = torch.tensor([p.size(0) for p in phone_ids],  dtype=torch.long)
    speaker_ids    = torch.tensor(list(speaker_idxs),              dtype=torch.long)

    audios   = pad_sequence(audios,    batch_first=True, padding_value=0.0)
    targets  = pad_sequence(phone_ids, batch_first=True, padding_value=0)

    return audios, audio_lengths, targets, target_lengths, speaker_ids, list(texts)
