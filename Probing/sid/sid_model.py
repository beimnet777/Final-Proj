"""SID probe heads: mean-pool frozen encoder representations → linear classifier.

Architecture (SUPERB SID spec):
    mean pool over valid frames → dropout → linear(hidden_size, num_speakers)

Two variants:
    MeanPoolLinearProbe         — single encoder layer  (probe_type='final')
    WeightedMeanPoolLinearProbe — softmax mix of layers (probe_type='weighted')

Both accept (hidden_states, frame_lengths) so padding is excluded from the mean.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import FrozenEncoder
from sid_config import SIDConfig


# ------------------------------------------------------ Shared utility ---


def _masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean-pool x over valid (non-padded) frames.

    x       : (B, T, D)
    lengths : (B,)  — number of valid frames per example
    returns : (B, D)
    """
    B, T, D = x.shape
    mask  = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask_f = mask.unsqueeze(-1).float()           # (B, T, 1)
    return (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)


# ---------------------------------------------------------- Probe heads ---


class MeanPoolLinearProbe(nn.Module):
    """Single encoder layer → masked mean pool → dropout → linear."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
        layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        frame_lengths: torch.Tensor,
    ) -> torch.Tensor:
        x = hidden_states[self.layer_idx]   # (B, T, D)
        x = _masked_mean(x, frame_lengths)  # (B, D)
        x = self.drop(x)
        return self.linear(x)               # (B, num_classes)


class WeightedMeanPoolLinearProbe(nn.Module):
    """Learnable softmax-weighted sum of all layers → mean pool → linear."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )
        self.weights = nn.Parameter(torch.zeros(num_layers))
        self.drop    = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden_size, num_classes)

    @property
    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.weights, dim=0).detach().cpu()

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        frame_lengths: torch.Tensor,
    ) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        x = sum(
            w[i] * self.layer_norm[i](hidden_states[i])
            for i in range(self.num_layers)
        )                                   # (B, T, D)
        x = _masked_mean(x, frame_lengths)  # (B, D)
        x = self.drop(x)
        return self.linear(x)               # (B, num_classes)


# ----------------------------------------------------------- Factory ------


def build_sid_model(cfg: SIDConfig):
    """Construct the frozen encoder and the chosen SID probe head.

    Returns (encoder, probe).
    cfg.encoder_layer_count is set as a side-effect.
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
        raise ValueError(f"Unknown probe_type for SID: {cfg.probe_type!r}")

    n_probe = sum(p.numel() for p in probe.parameters())
    print(
        f"[build_sid_model] probe={cfg.probe_type}"
        f"  hidden={encoder.hidden_size}  layers={encoder.num_layers}"
        f"  speakers={cfg.num_classes}  probe_params={n_probe:,}"
    )
    return encoder, probe
