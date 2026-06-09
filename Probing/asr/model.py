"""Generic frozen speech encoder + probe heads.

Supports two encoder families selectable via --model_family:
  spear   — marcoyang/spear-xlarge-speech-audio (Zipformer, custom forward API)
  hf      — any standard HuggingFace speech encoder (wav2vec2, HuBERT, WavLM, …)
             loaded with output_hidden_states=True and standard attention_mask.
Only the probe is trainable.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from config import Config


# ---------------------------------------------------------------- Encoder ---


class FrozenEncoder(nn.Module):
    """Model-agnostic wrapper around a frozen HF speech encoder.

    Two families are supported (selected by model_family):

    'spear'
        SPEAR-XLarge (marcoyang/spear-xlarge-speech-audio).  Custom Zipformer
        backbone loaded via trust_remote_code=True.  Positional forward API:
            encoder(audio, audio_lengths)
        Returns a dict with a 'hidden_states' list of 13 × (B, T, 1280).
        Frame lengths estimated via empirical T_out/T_in ratio.

    'hf'
        Standard HuggingFace speech encoders (wav2vec2, HuBERT, WavLM, …).
        Keyword forward API:
            encoder(input_values=audio, attention_mask=mask,
                    output_hidden_states=True)
        Returns a ModelOutput with hidden_states tuple (includes CNN embedding
        layer at index 0, so we drop it to keep only transformer layers).
        Frame lengths derived from the attention_mask of the last forward pass.
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

        # Auto-detect hidden size and layer count from the loaded model.
        if model_family == "spear":
            # SPEAR hard-codes 1280 / 13 in its config.
            self.hidden_size = getattr(self.encoder.config, "hidden_size", 1280)
            self.num_layers  = getattr(self.encoder.config, "num_hidden_layers", 13)
            # Fallback: SPEAR's config key may differ.
            if self.hidden_size == 0:
                self.hidden_size = 1280
            if self.num_layers == 0:
                self.num_layers = 13
        else:
            cfg = self.encoder.config
            self.hidden_size = cfg.hidden_size
            # HF models include a CNN feature extractor layer at hidden_states[0];
            # the number of *transformer* layers is num_hidden_layers.
            self.num_layers = cfg.num_hidden_layers

        # For SPEAR: cached ratio used by output_lengths().
        self._last_input_T: int = 0
        self._last_output_T: int = 0
        # For HF: attention_mask of last forward (B, T_audio); used for frame counts.
        self._last_attention_mask: torch.Tensor | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()   # keep frozen encoder always in eval
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
            hidden_states = list(out["hidden_states"])          # list of 13 tensors
            self._last_input_T  = audio.size(1)
            self._last_output_T = hidden_states[0].size(1)
            return hidden_states

        else:  # 'hf'
            # Build attention_mask from audio_lengths.
            B, T = audio.shape
            mask = torch.arange(T, device=audio.device).unsqueeze(0) < audio_lengths.unsqueeze(1)
            mask = mask.long()
            self._last_attention_mask = mask

            out = self.encoder(
                input_values=audio,
                attention_mask=mask,
                output_hidden_states=True,
            )
            # hidden_states[0] is the CNN feature extractor output (not a
            # transformer layer) — drop it so indexing matches num_layers.
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
                # Pre-forward fallback: wav2vec2/HuBERT downsample by ~320.
                return torch.div(audio_lengths, 320, rounding_mode="floor").clamp(min=1).long()
            # Use the model's built-in helper when available (wav2vec2, HuBERT).
            if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
                return self.encoder._get_feat_extract_output_lengths(audio_lengths).long()
            # Generic fallback: infer ratio from mask length vs frame length.
            T_audio = self._last_attention_mask.size(1)
            T_frames = self._get_last_frame_count()
            if T_frames is None or T_audio == 0:
                return torch.div(audio_lengths, 320, rounding_mode="floor").clamp(min=1).long()
            ratio = T_frames / T_audio
            return (audio_lengths.float() * ratio).floor().clamp(min=1).long()

    def _get_last_frame_count(self) -> int | None:
        """Return the T_frames dimension from the last forward pass, if cached."""
        return getattr(self, "_last_output_T", None) or None


# Backwards-compatible alias so existing code that imports FrozenSpear still works.
FrozenSpear = FrozenEncoder


# ---------------------------------------------------------------- Probes ----


class SingleLayerProbe(nn.Module):
    """Single encoder layer → projector → dropout → linear.

    Matches SUPERB: Linear(upstream_dim, proj_dim) before the CTC head.

        layers      : list of L tensors, each (B, T, D)
        out logits  : (B, T, V)
    """

    def __init__(self, hidden_size: int, vocab_size: int, proj_dim: int = 1024,
                 dropout: float = 0.1, layer_idx: int = -1):
        super().__init__()
        self.layer_idx  = layer_idx
        self.projector  = nn.Linear(hidden_size, proj_dim)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(proj_dim, vocab_size)

    def forward(self, layers: List[torch.Tensor]) -> torch.Tensor:
        h = layers[self.layer_idx]           # (B, T, D)
        h = self.projector(h)                # (B, T, proj_dim)
        h = self.dropout(h)
        return self.classifier(h)            # (B, T, V)


class WeightedLayerSumProbe(nn.Module):
    """Weighted sum of all L layers → projector → dropout → linear.

    Matches SUPERB: weighted sum first, then Linear(upstream_dim, proj_dim) before head.

        layers      : list of L tensors, each (B, T, D)
        out logits  : (B, T, V)
    """

    def __init__(self, num_layers: int, hidden_size: int, vocab_size: int,
                 proj_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.mix_logits  = nn.Parameter(torch.zeros(num_layers))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.projector   = nn.Linear(hidden_size, proj_dim)
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = nn.Linear(proj_dim, vocab_size)

    def forward(self, layers: List[torch.Tensor]) -> torch.Tensor:
        normed_layers = [ln(h) for ln, h in zip(self.layer_norms, layers)]
        stacked = torch.stack(normed_layers, dim=0)                 # (L, B, T, D)
        w = F.softmax(self.mix_logits, dim=0).view(-1, 1, 1, 1)
        h = (w * stacked).sum(dim=0)                                # (B, T, D)
        h = self.projector(h)                                       # (B, T, proj_dim)
        h = self.dropout(h)
        return self.classifier(h)                                   # (B, T, V)

    @property
    def layer_weights(self) -> torch.Tensor:
        return F.softmax(self.mix_logits.detach(), dim=0)


class WeightedLSTMProbe(nn.Module):
    """Learnable softmax mixture of all L SPEAR layers, then a 2-layer BLSTM.

    Combines WeightedLayerSumProbe's layer mixing with the BLSTM downstream
    model.  SpecAugment applied to the mixed representation before the LSTM.

        layers      : list of L tensors, each (B, T, D)
        out logits  : (B, T, V)
    """

    def __init__(self, num_layers: int, hidden_size: int, vocab_size: int,
                 proj_dim: int = 1024, lstm_hidden: int = 1024, lstm_num_layers: int = 2,
                 dropout: float = 0.1,
                 time_mask_param: int = 50, freq_mask_param: int = 64):
        super().__init__()
        self.mix_logits  = nn.Parameter(torch.zeros(num_layers))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.projector   = nn.Linear(hidden_size, proj_dim)

        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)

    def _spec_augment(self, h: torch.Tensor) -> torch.Tensor:
        if self.time_mask_param > 0:
            B, T, D = h.shape
            for b in range(B):
                t = random.randint(0, self.time_mask_param)
                t0 = random.randint(0, max(0, T - t))
                h[b, t0:t0 + t, :] = 0.0
        if self.freq_mask_param > 0:
            D = h.size(2)
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, D - f))
            h[:, :, f0:f0 + f] = 0.0
        return h

    def forward(self, layers: List[torch.Tensor]) -> torch.Tensor:
        normed = [ln(h) for ln, h in zip(self.layer_norms, layers)]
        stacked = torch.stack(normed, dim=0)                    # (L, B, T, D)
        w = F.softmax(self.mix_logits, dim=0).view(-1, 1, 1, 1)
        h = (w * stacked).sum(dim=0)                            # (B, T, D)
        h = self.projector(h)                                   # (B, T, proj_dim)
        if self.training:
            h = h.clone()
            h = self._spec_augment(h)
        h, _ = self.lstm(h)                                     # (B, T, lstm_hidden*2)
        h = self.dropout(h)
        return self.classifier(h)                               # (B, T, V)

    @property
    def layer_weights(self) -> torch.Tensor:
        return F.softmax(self.mix_logits.detach(), dim=0)


class FixedWeightedLSTMProbe(nn.Module):
    """Uniform average of all L encoder layers, then a 2-layer BLSTM.

    The layer weights are fixed at 1/L.  Per-layer LayerNorm is non-affine so
    this baseline cannot learn an implicit layer reweighting through norm gains.
    """

    def __init__(self, num_layers: int, hidden_size: int, vocab_size: int,
                 proj_dim: int = 1024, lstm_hidden: int = 1024, lstm_num_layers: int = 2,
                 dropout: float = 0.1,
                 time_mask_param: int = 50, freq_mask_param: int = 64):
        super().__init__()
        self.num_layers = num_layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size, elementwise_affine=False)
            for _ in range(num_layers)
        ])
        self.projector = nn.Linear(hidden_size, proj_dim)

        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)

    def _spec_augment(self, h: torch.Tensor) -> torch.Tensor:
        if self.time_mask_param > 0:
            B, T, D = h.shape
            for b in range(B):
                t = random.randint(0, self.time_mask_param)
                t0 = random.randint(0, max(0, T - t))
                h[b, t0:t0 + t, :] = 0.0
        if self.freq_mask_param > 0:
            D = h.size(2)
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, D - f))
            h[:, :, f0:f0 + f] = 0.0
        return h

    def forward(self, layers: List[torch.Tensor]) -> torch.Tensor:
        normed = [ln(h) for ln, h in zip(self.layer_norms, layers)]
        h = torch.stack(normed, dim=0).mean(dim=0)               # (B, T, D)
        h = self.projector(h)                                   # (B, T, proj_dim)
        if self.training:
            h = h.clone()
            h = self._spec_augment(h)
        h, _ = self.lstm(h)                                     # (B, T, lstm_hidden*2)
        h = self.dropout(h)
        return self.classifier(h)                               # (B, T, V)

    @property
    def layer_weights(self) -> torch.Tensor:
        return torch.full((self.num_layers,), 1.0 / self.num_layers)


class LSTMProbe(nn.Module):
    """2-layer bidirectional LSTM probe on the final SPEAR layer + linear head.

    Matches the downstream architecture from the SUPERB benchmark:
      - 2-layer BLSTM with 1024 units per direction
      - SpecAugment (time + frequency masking) applied to the frozen
        representations before the LSTM to reduce overfitting
      - CTC loss used by the training loop

        layers      : list of L tensors, each (B, T, D)
        out logits  : (B, T, V)
    """

    def __init__(self, hidden_size: int, vocab_size: int,
                 proj_dim: int = 1024, lstm_hidden: int = 1024, num_layers: int = 2,
                 dropout: float = 0.1, layer_idx: int = -1,
                 time_mask_param: int = 50, freq_mask_param: int = 64):
        super().__init__()
        self.layer_idx = layer_idx
        self.projector = nn.Linear(hidden_size, proj_dim)

        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)  # ×2 for bidir

    def _spec_augment(self, h: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment-style masking to representation tensor (B, T, D)."""
        # Time masking: zero out a random contiguous block of frames.
        if self.time_mask_param > 0:
            B, T, D = h.shape
            for b in range(B):
                t = random.randint(0, self.time_mask_param)
                t0 = random.randint(0, max(0, T - t))
                h[b, t0:t0 + t, :] = 0.0
        # Frequency (feature) masking: zero out a random contiguous block of dims.
        if self.freq_mask_param > 0:
            D = h.size(2)
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, D - f))
            h[:, :, f0:f0 + f] = 0.0
        return h

    def forward(self, layers: List[torch.Tensor]) -> torch.Tensor:
        h = layers[self.layer_idx]           # (B, T, D)
        h = self.projector(h)                # (B, T, proj_dim)
        if self.training:
            h = h.clone()
            h = self._spec_augment(h)
        h, _ = self.lstm(h)                  # (B, T, lstm_hidden*2)
        h = self.dropout(h)
        return self.classifier(h)            # (B, T, V)


# ------------------------------------------------------------- Factory -----


def build_model(cfg: Config) -> Tuple[FrozenEncoder, nn.Module]:
    """Construct the frozen encoder and the chosen probe head."""
    encoder = FrozenEncoder(cfg.model_id, model_family=cfg.model_family)
    cfg.encoder_layer_count = encoder.num_layers  # populate runtime field

    if cfg.probe_type == "final":
        probe = SingleLayerProbe(
            hidden_size=encoder.hidden_size,
            vocab_size=cfg.vocab_size,
            proj_dim=cfg.proj_dim,
            dropout=cfg.probe_dropout,
            layer_idx=cfg.layer_idx,
        )
    elif cfg.probe_type == "weighted":
        probe = WeightedLayerSumProbe(
            num_layers=encoder.num_layers,
            hidden_size=encoder.hidden_size,
            vocab_size=cfg.vocab_size,
            proj_dim=cfg.proj_dim,
            dropout=cfg.probe_dropout,
        )
    elif cfg.probe_type == "weighted_lstm":
        probe = WeightedLSTMProbe(
            num_layers=encoder.num_layers,
            hidden_size=encoder.hidden_size,
            vocab_size=cfg.vocab_size,
            proj_dim=cfg.proj_dim,
            lstm_hidden=cfg.lstm_hidden,
            lstm_num_layers=cfg.lstm_layers,
            dropout=cfg.probe_dropout,
            time_mask_param=cfg.time_mask_param,
            freq_mask_param=cfg.freq_mask_param,
        )
    elif cfg.probe_type == "fixed_weighted_lstm":
        probe = FixedWeightedLSTMProbe(
            num_layers=encoder.num_layers,
            hidden_size=encoder.hidden_size,
            vocab_size=cfg.vocab_size,
            proj_dim=cfg.proj_dim,
            lstm_hidden=cfg.lstm_hidden,
            lstm_num_layers=cfg.lstm_layers,
            dropout=cfg.probe_dropout,
            time_mask_param=cfg.time_mask_param,
            freq_mask_param=cfg.freq_mask_param,
        )
    elif cfg.probe_type == "lstm":
        probe = LSTMProbe(
            hidden_size=encoder.hidden_size,
            vocab_size=cfg.vocab_size,
            proj_dim=cfg.proj_dim,
            lstm_hidden=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            dropout=cfg.probe_dropout,
            layer_idx=cfg.layer_idx,
            time_mask_param=cfg.time_mask_param,
            freq_mask_param=cfg.freq_mask_param,
        )
    else:
        raise ValueError(f"Unknown probe_type: {cfg.probe_type!r}")

    return encoder, probe
