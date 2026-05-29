"""Frozen SPEAR encoder with a learnable softmax-weighted sum over all layers."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SpearWeightedEncoder(nn.Module):
    """Wraps the frozen SPEAR-Large backbone and adds a learnable per-layer mix.

    The SPEAR model is always kept in eval mode and all its parameters are
    frozen.  The only trainable parameters here are:

        mix_logits  (L,)              softmax-normalised layer weights
        layer_norms ModuleList[LN(D)] one LayerNorm per transformer layer

    forward() returns the weighted sum h_t ∈ ℝ^{B×T×D}.  Gradients flow
    through mix_logits and layer_norms; they do NOT flow into SPEAR.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self._spear = AutoModel.from_pretrained(
            cfg.spear_model_id, trust_remote_code=True
        )
        for p in self._spear.parameters():
            p.requires_grad_(False)
        self._spear.eval()

        D, L = cfg.D, cfg.L
        # Uniform init: softmax(zeros) == 1/L for every layer.
        self.mix_logits = nn.Parameter(torch.zeros(L))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(D) for _ in range(L)])

        # Cached ratio for output_lengths(); populated on first forward.
        self._last_input_T: int = 0
        self._last_output_T: int = 0

    # Keep SPEAR frozen even when the outer module is set to train().
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
        h_t           : (B, T_frames, D)  learnable-weighted sum of SPEAR layers
        out_lengths   : (B,)              valid frame counts (computed after call)
        """
        with torch.no_grad():
            out = self._spear(audio, audio_lengths)
            hidden_states = list(out["hidden_states"])      # L × (B, T, D)
            self._last_input_T = audio.size(1)
            self._last_output_T = hidden_states[0].size(1)

        # Learnable weighted sum — gradients flow through mix_logits / layer_norms.
        normed = [ln(h) for ln, h in zip(self.layer_norms, hidden_states)]
        stacked = torch.stack(normed, dim=0)                # (L, B, T, D)
        w = F.softmax(self.mix_logits, dim=0).view(-1, 1, 1, 1)
        h_t = (w * stacked).sum(0)                          # (B, T, D)

        out_lengths = self.output_lengths(audio_lengths)
        return h_t, out_lengths

    def output_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """Map raw-audio sample counts → per-frame output counts."""
        if self._last_input_T == 0:
            # Pre-forward fallback: SPEAR/Zipformer typically downsamples ~640×.
            return torch.div(audio_lengths, 640, rounding_mode="floor").clamp(min=1).long()
        ratio = self._last_output_T / self._last_input_T
        return (audio_lengths.float() * ratio).floor().clamp(min=1).long()

    @property
    def layer_weights(self) -> torch.Tensor:
        """Detached softmax weights for logging. Shape: (L,)."""
        return F.softmax(self.mix_logits.detach(), dim=0)
