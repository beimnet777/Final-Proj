"""Frozen SPEAR encoder — returns a fixed uniform average over all transformer layers.

No trainable parameters.  The layer mean gives the SAE a richer target than the
final layer alone while keeping h_t completely fixed (no moving target).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SpearEncoder(nn.Module):
    """Frozen SPEAR-Large backbone.

    Returns h_t = (1/L) Σ_l hidden_state_l  — uniform mean over all L layers.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._layernorm = bool(getattr(cfg, "spear_layernorm", False))
        self._spear = AutoModel.from_pretrained(
            cfg.spear_model_id, trust_remote_code=True
        )
        for p in self._spear.parameters():
            p.requires_grad_(False)
        self._spear.eval()

        self._last_input_T: int = 0
        self._last_output_T: int = 0

    def train(self, mode: bool = True):
        super().train(mode)
        self._spear.eval()
        return self

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        audio         : (B, T_samples)  raw 16 kHz waveform, zero-padded
        audio_lengths : (B,)            true sample counts

        Returns
        -------
        h_t         : (B, T_frames, D)  uniform mean of all SPEAR layers (detached)
        out_lengths : (B,)              valid frame counts
        """
        with torch.no_grad():
            out = self._spear(audio, audio_lengths)
            # hidden_states: list of L tensors each (B, T, D)
            stacked = torch.stack(out["hidden_states"], dim=0)  # (L, B, T, D)
            if self._layernorm:
                # SUPERB-style: layer-norm each layer (no affine) before averaging.
                stacked = F.layer_norm(stacked, (stacked.size(-1),))
            h_t = stacked.mean(dim=0)                           # (B, T, D)
            self._last_input_T  = audio.size(1)
            self._last_output_T = h_t.size(1)

        out_lengths = self.output_lengths(audio_lengths)
        return h_t, out_lengths

    def output_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        if self._last_input_T == 0:
            return torch.div(audio_lengths, 640, rounding_mode="floor").clamp(min=1).long()
        ratio = self._last_output_T / self._last_input_T
        return (audio_lengths.float() * ratio).floor().clamp(min=1).long()
