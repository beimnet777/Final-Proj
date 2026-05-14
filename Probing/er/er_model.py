"""ER probe heads: mean-pool frozen encoder representations → linear classifier.

Two heads matching the SUPERB ER specification:
    MeanPoolLinearProbe         — single layer → mean pool → linear (probe_type='final')
    WeightedMeanPoolLinearProbe — softmax mix of all layers → mean pool → linear
                                  (probe_type='weighted')

Both take (hidden_states, frame_lengths) as input so masked mean-pooling can
exclude padded frames from the average.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

# Allow importing FrozenEncoder from the parent Probing/ directory.
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import FrozenEncoder
from er_config import ERConfig


# ------------------------------------------------------ Shared utility ---


def _masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean-pool x over valid (non-padded) frames.

    x       : (B, T, D)
    lengths : (B,)  — number of valid frames per example
    returns : (B, D)
    """
    B, T, D = x.shape
    # Boolean mask (B, T): True where frame is valid.
    mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask_f = mask.unsqueeze(-1).float()           # (B, T, 1)
    return (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)


# ---------------------------------------------------------- Probe heads ---


class MeanPoolLinearProbe(nn.Module):
    """Single encoder layer → masked mean pool → dropout → linear.

    This is the 'final' probe: it uses a single layer (layer_idx) from the
    encoder's hidden states and classifies the mean-pooled representation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        frame_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_states : list of (B, T, D) — one tensor per encoder layer
        frame_lengths : (B,) — valid frame counts
        returns       : (B, num_classes) logits
        """
        x = hidden_states[self.layer_idx]           # (B, T, D)
        x = _masked_mean(x, frame_lengths)          # (B, D)
        x = self.drop(x)
        return self.linear(x)                       # (B, C)


class WeightedMeanPoolLinearProbe(nn.Module):
    """Learnable softmax-weighted sum of all layers → mean pool → linear.

    This is the 'weighted' probe, matching SUPERB's weighted-sum approach.
    A LayerNorm is applied per-layer before mixing (same as in the ASR probe).
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        # Per-layer normalisation before mixing.
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )
        # Scalar logits; softmax gives the mixture weights.
        self.weights = nn.Parameter(torch.zeros(num_layers))
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_classes)

    @property
    def layer_weights(self) -> torch.Tensor:
        """Softmax mixture weights as a detached CPU tensor (for logging)."""
        return torch.softmax(self.weights, dim=0).detach().cpu()

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        frame_lengths: torch.Tensor,
    ) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)      # (L,)
        # Weighted sum across layers: (B, T, D).
        x = sum(
            w[i] * self.layer_norm[i](hidden_states[i])
            for i in range(self.num_layers)
        )
        x = _masked_mean(x, frame_lengths)          # (B, D)
        x = self.drop(x)
        return self.linear(x)                       # (B, C)


# ----------------------------------------------------------- Factory ------


def build_er_model(cfg: ERConfig):
    """Construct the frozen encoder and the chosen ER probe head.

    Returns (encoder, probe).
    cfg.encoder_layer_count is set as a side-effect once the encoder is loaded.
    """
    encoder = FrozenEncoder(cfg.model_id, model_family=cfg.model_family)
    cfg.encoder_layer_count = encoder.num_layers

    if cfg.probe_type == "final":
        probe = MeanPoolLinearProbe(
            hidden_size=encoder.hidden_size,
            num_classes=cfg.num_classes,
            dropout=cfg.probe_dropout,
            layer_idx=cfg.layer_idx,
        )
    elif cfg.probe_type == "weighted":
        probe = WeightedMeanPoolLinearProbe(
            num_layers=encoder.num_layers,
            hidden_size=encoder.hidden_size,
            num_classes=cfg.num_classes,
            dropout=cfg.probe_dropout,
        )
    else:
        raise ValueError(f"Unknown probe_type for ER: {cfg.probe_type!r}")

    n_probe = sum(p.numel() for p in probe.parameters())
    print(f"[build_er_model] probe_type={cfg.probe_type}  params={n_probe:,}")
    return encoder, probe
