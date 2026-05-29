"""Shared frozen speech encoder used by all probing tasks.

Supports two encoder families selectable via model_family:
  spear — marcoyang/spear-xlarge-speech-audio (Zipformer, custom forward API)
  hf    — any standard HuggingFace speech encoder (wav2vec2, HuBERT, WavLM, …)
Only the probe head is trainable; the encoder is always frozen.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel


class FrozenEncoder(nn.Module):
    """Model-agnostic wrapper around a frozen HF speech encoder.

    'spear'
        SPEAR-XLarge (marcoyang/spear-xlarge-speech-audio).  Positional forward API:
            encoder(audio, audio_lengths)
        Returns a dict with 'hidden_states': list of 13 × (B, T, 1280).

    'hf'
        Standard HF speech encoders (wav2vec2, HuBERT, WavLM, …).
        Returns hidden_states tuple; CNN extractor layer at index 0 is dropped.
    """

    def __init__(self, model_id: str, model_family: str = "spear"):
        super().__init__()
        assert model_family in ("spear", "hf"), \
            f"model_family must be 'spear' or 'hf', got {model_family!r}"
        self.model_family = model_family

        trust = (model_family == "spear")
        self.encoder = AutoModel.from_pretrained(model_id, trust_remote_code=trust)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        if model_family == "spear":
            self.hidden_size = getattr(self.encoder.config, "hidden_size", 1280)
            self.num_layers  = getattr(self.encoder.config, "num_hidden_layers", 13)
            if self.hidden_size == 0:
                self.hidden_size = 1280
            if self.num_layers == 0:
                self.num_layers = 13
        else:
            cfg = self.encoder.config
            self.hidden_size = cfg.hidden_size
            self.num_layers  = cfg.num_hidden_layers

        self._last_input_T: int = 0
        self._last_output_T: int = 0
        self._last_attention_mask: torch.Tensor | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> List[torch.Tensor]:
        """
        audio:         (B, T_audio)  raw 16 kHz waveform, zero-padded
        audio_lengths: (B,)          true sample counts per example
        returns:       list of num_layers tensors, each (B, T_frames, hidden_size)
        """
        if self.model_family == "spear":
            out = self.encoder(audio, audio_lengths)
            hidden_states = list(out["hidden_states"])
            self._last_input_T  = audio.size(1)
            self._last_output_T = hidden_states[0].size(1)
            return hidden_states

        else:  # 'hf'
            B, T = audio.shape
            mask = torch.arange(T, device=audio.device).unsqueeze(0) < audio_lengths.unsqueeze(1)
            mask = mask.long()
            self._last_attention_mask = mask

            out = self.encoder(
                input_values=audio,
                attention_mask=mask,
                output_hidden_states=True,
            )
            # Drop CNN feature extractor layer at index 0
            return list(out.hidden_states[1:])

    def output_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """Map raw-audio sample counts → per-frame output counts."""
        if self.model_family == "spear":
            if self._last_input_T == 0:
                return torch.div(audio_lengths, 640, rounding_mode="floor").clamp(min=1).long()
            ratio = self._last_output_T / self._last_input_T
            return (audio_lengths.float() * ratio).floor().clamp(min=1).long()

        else:  # 'hf'
            if self._last_attention_mask is None:
                return torch.div(audio_lengths, 320, rounding_mode="floor").clamp(min=1).long()
            if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
                return self.encoder._get_feat_extract_output_lengths(audio_lengths).long()
            T_audio  = self._last_attention_mask.size(1)
            T_frames = getattr(self, "_last_output_T", None) or None
            if T_frames is None or T_audio == 0:
                return torch.div(audio_lengths, 320, rounding_mode="floor").clamp(min=1).long()
            ratio = T_frames / T_audio
            return (audio_lengths.float() * ratio).floor().clamp(min=1).long()


# Backwards-compatible alias
FrozenSpear = FrozenEncoder
