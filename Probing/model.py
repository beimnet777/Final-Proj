"""SPEAR encoder (frozen) + two probe heads.

The encoder is a HuggingFace AutoModel speech encoder, treated as a black box
that maps a raw 16 kHz waveform to one hidden state per transformer layer.
Only the probe is trainable.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from config import Config


# ---------------------------------------------------------------- Encoder ---


class FrozenSpear(nn.Module):
    """Wraps SPEAR-XLarge (marcoyang/spear-xlarge-speech-audio) and freezes it.

    Verified against the HF model card:
      - Custom Zipformer backbone, loaded via `trust_remote_code=True`.
      - Forward is positional: encoder(audio, audio_len)
            audio:      (B, T_audio)   raw 16 kHz waveform
            audio_len:  (B,)           true sample counts (handles padding)
      - Returns a dict whose `hidden_states` entry is a list of 13 tensors
        (B, T_frames, 1280), one per Zipformer stack. The list is already
        the 13 transformer-layer outputs -- no pre-transformer entry to drop,
        and no `output_hidden_states` flag is needed.
      - Hidden size = 1280, num layers = 13.
      - No HF-style helper for output frame counts, so `output_lengths`
        scales audio lengths by the empirical T_out/T_in ratio observed
        in the most recent forward pass (Zipformer's conv frontend uses a
        fixed downsampling factor, so the scaling is linear).
    """

    # Hard-coded for spear-xlarge-speech-audio per the HF model card.
    HIDDEN_SIZE = 1280
    NUM_LAYERS = 13

    def __init__(self, model_id: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        self.hidden_size = self.HIDDEN_SIZE
        self.num_layers = self.NUM_LAYERS

        # Cached after first forward; used by output_lengths().
        self._last_input_T: int = 0
        self._last_output_T: int = 0

    def train(self, mode: bool = True):
        # Keep the encoder in eval mode regardless of the outer module's
        # mode, so dropout / norm stats inside SPEAR stay frozen.
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor) -> List[torch.Tensor]:
        # audio:         (B, T_audio)
        # audio_lengths: (B,)
        out = self.encoder(audio, audio_lengths)
        hidden_states = out["hidden_states"]     # list of 13 tensors, each (B, T_frames, 1280)

        # Cache the downsampling ratio for output_lengths() below.
        self._last_input_T = audio.size(1)
        self._last_output_T = hidden_states[0].size(1)
        return list(hidden_states)

    def output_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        """Map raw-audio sample counts to per-frame output counts.

        Uses the empirical T_out / T_in_padded ratio from the most recent
        forward pass. Since Zipformer's conv frontend subsamples by a fixed
        factor independent of input length, this linear scale recovers each
        item's true frame count up to a +/-1 boundary error (fine for CTC).
        """
        if self._last_input_T == 0:
            # Pre-forward fallback (~25 Hz on 16 kHz audio: 16000/25 = 640).
            return torch.div(audio_lengths, 640, rounding_mode="floor").long()
        ratio = self._last_output_T / self._last_input_T
        return (audio_lengths.float() * ratio).floor().clamp(min=1).long()


# ---------------------------------------------------------------- Probes ----


class SingleLayerProbe(nn.Module):
    """Linear classifier on the last SPEAR transformer layer.

        layers      : list of L tensors, each (B, T, D)
        out logits  : (B, T, V)
    """

    def __init__(self, hidden_size: int, vocab_size: int, dropout: float = 0.1, layer_idx: int = -1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.layer_idx = layer_idx

    def forward(self, layers: List[torch.Tensor]) -> torch.Tensor:
        h = layers[self.layer_idx]                       # (B, T, D)  -- last layer only
        h = self.dropout(h)
        return self.classifier(h)            # (B, T, V)


class WeightedLayerSumProbe(nn.Module):
    """Learnable softmax mixture of all L SPEAR layers, then a linear head.

    The L raw mixing parameters live in `self.mix_logits`; softmax over them
    makes the per-layer weights a convex combination so they sum to 1.

        layers      : list of L tensors, each (B, T, D)
        out logits  : (B, T, V)
    """

    def __init__(self, num_layers: int, hidden_size: int, vocab_size: int,
                 dropout: float = 0.1):
        super().__init__()
        # Uniform init: softmax(zeros) = 1/L for each layer.
        self.mix_logits = nn.Parameter(torch.zeros(num_layers))         # (L,)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, layers: List[torch.Tensor]) -> torch.Tensor:
        # Normalize each layer's representations before mixing, to put them on a common scale. Each normed layer is (B, T, D).
        normed_layers = [ln(h) for ln, h in zip(self.layer_norms, layers)]
        # Stack along a new layer axis: (L, B, T, D)
        stacked = torch.stack(normed_layers, dim=0)
        # Broadcast weights to (L, 1, 1, 1):    softmax over layers -> sums to 1.
        w = F.softmax(self.mix_logits, dim=0).view(-1, 1, 1, 1)
        # Weighted sum across the L axis:       (B, T, D)
        h = (w * stacked).sum(dim=0)
        h = self.dropout(h)
        return self.classifier(h)             # (B, T, V)

    @property
    def layer_weights(self) -> torch.Tensor:
        """Detached softmax weights for logging/analysis. Shape: (L,)."""
        return F.softmax(self.mix_logits.detach(), dim=0)


# ------------------------------------------------------------- Factory -----


def build_model(cfg: Config) -> Tuple[FrozenSpear, nn.Module]:
    """Construct the frozen encoder and the chosen probe head."""
    encoder = FrozenSpear(cfg.spear_model_id)
    cfg.encoder_layer_count = encoder.num_layers  # populate runtime field

    if cfg.probe_type == "final":
        probe = SingleLayerProbe(
            hidden_size=encoder.hidden_size,
            vocab_size=cfg.vocab_size,
            dropout=cfg.probe_dropout,
            layer_idx=cfg.layer_idx, 
        )
    elif cfg.probe_type == "weighted":
        probe = WeightedLayerSumProbe(
            num_layers=encoder.num_layers,
            hidden_size=encoder.hidden_size,
            vocab_size=cfg.vocab_size,
            dropout=cfg.probe_dropout,
        )
    else:
        raise ValueError(f"Unknown probe_type: {cfg.probe_type!r}")

    return encoder, probe
