"""Frozen adapter from a disentanglement checkpoint to the standard probes.

The adapter deliberately exposes exactly one representation (z_t, z_L, or z_P)
as a one-layer upstream.  The task-specific data, probe heads, optimizers,
schedules, validation selection, and metrics remain those in ``Probing/``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import List

import torch
import torch.nn as nn


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Disentanglement.config import DISConfig
from Disentanglement.model import build_dis_model


VALID_SOURCES = ("z_t", "z_L", "z_P")


def _checkpoint_model_state(payload: Mapping) -> Mapping[str, torch.Tensor]:
    if "model_state" in payload:
        return payload["model_state"]
    if "model" in payload:
        return payload["model"]
    return payload


class CheckpointRepresentationEncoder(nn.Module):
    """Expose one frozen SAE representation using the standard probe API."""

    def __init__(self, checkpoint_path: str | Path, source: str) -> None:
        super().__init__()
        if source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {VALID_SOURCES}, got {source!r}")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"disentanglement checkpoint not found: {checkpoint_path}")

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(f"checkpoint must contain a mapping, got {type(payload).__name__}")

        cfg = DISConfig()
        for key, value in payload.get("analysis_config", {}).items():
            setattr(cfg, key, value)
        cfg.device = "cpu"

        model = build_dis_model(cfg)
        state = _checkpoint_model_state(payload)
        current = model.state_dict()
        compatible = {
            key: value for key, value in state.items()
            if key in current and torch.is_tensor(value)
            and tuple(value.shape) == tuple(current[key].shape)
        }
        missing, _unexpected = model.load_state_dict(compatible, strict=False)

        loaded_sae = sorted(key for key in compatible if str(key).startswith("sae."))
        if not loaded_sae:
            raise ValueError(
                f"checkpoint/model mismatch: no compatible SAE tensors loaded from {checkpoint_path}"
            )

        route_topk_buffers = {
            "sae.route_topk_enabled",
            "sae.route_topk_idx",
            "sae.route_topk_quotas",
        }
        allowed_missing_prefixes = (
            "encoder._spear.",
            "pr_head.", "sid_head.", "grl_head.", "pr_grl_head.",
            "prosody_head.", "prosody_grl_head.", "emotion_head.",
            "emotion_grl_head.", "emotion_u_grl_head.",
        )
        material_missing = [
            key for key in missing
            if key not in route_topk_buffers
            and not key.startswith(allowed_missing_prefixes)
        ]
        if material_missing:
            raise ValueError(
                "checkpoint/model mismatch after compatible load: "
                f"missing={material_missing[:8]}"
            )

        self.model = model
        self.source = source
        self.checkpoint_path = checkpoint_path
        self.num_layers = 1
        if source == "z_t":
            self.hidden_size = int(cfg.K)
        elif bool(getattr(cfg, "projection_disentanglement", False)):
            self.hidden_size = int(getattr(cfg, "projection_dim", cfg.K))
        else:
            self.hidden_size = int(cfg.K)
        self._last_output_lengths: torch.Tensor | None = None

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        ignored_shape = sorted(
            key for key, value in state.items()
            if key in current and torch.is_tensor(value)
            and tuple(value.shape) != tuple(current[key].shape)
        )
        print(
            "[CheckpointRepresentationEncoder] "
            f"checkpoint={checkpoint_path} source={source} hidden={self.hidden_size} "
            f"compatible={len(compatible)} sae={len(loaded_sae)} "
            f"shape_ignored={len(ignored_shape)}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, audio: torch.Tensor,
                audio_lengths: torch.Tensor) -> List[torch.Tensor]:
        out = self.model(
            audio,
            audio_lengths,
            stage=2,
            grl_lambda=0.0,
            grl_p_lambda=0.0,
            grl_prosody_lambda=0.0,
            grl_emotion_lambda=0.0,
            emit_emotion=False,
        )
        representation = out[self.source].detach().float()
        self._last_output_lengths = out["out_lengths"].detach()
        if representation.size(-1) != self.hidden_size:
            raise RuntimeError(
                f"configured hidden size {self.hidden_size} does not match "
                f"{self.source} output {representation.size(-1)}"
            )
        return [representation]

    def output_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        if self._last_output_lengths is not None:
            return self._last_output_lengths.to(audio_lengths.device)
        return self.model.encoder.output_lengths(audio_lengths).long()
