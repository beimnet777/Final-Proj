"""Collate function for variable-length audio batches."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length waveforms into a batch.

    Batch item: (waveform, n_samples)

    Returns
    -------
    audios         (B, T_max)  float32, zero-padded
    audio_lengths  (B,)        long, true sample counts
    """
    audios, lengths = zip(*batch)
    audio_lengths = torch.tensor(list(lengths), dtype=torch.long)
    audios = pad_sequence(list(audios), batch_first=True, padding_value=0.0)
    return audios, audio_lengths
