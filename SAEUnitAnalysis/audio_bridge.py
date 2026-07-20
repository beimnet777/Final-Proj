"""Reusable SPEAR-feature to log-mel bridge.

The bridge is deliberately trained on ordinary, unswapped frozen SPEAR
representations. It learns feature inversion, not the L/P intervention, and can
therefore be frozen and reused across SAE checkpoints with the same SPEAR
feature-domain signature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import AnalysisError


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = 24_000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    n_mels: int = 100
    f_min: float = 0.0
    f_max: float = 12_000.0
    log_floor: float = 1e-5
    center: bool = False
    reflect_pad: bool = True


@dataclass(frozen=True)
class BridgeConfig:
    input_dim: int = 1280
    hidden_dim: int = 384
    residual_layers: int = 6
    kernel_size: int = 5
    dropout: float = 0.05
    spear_sample_rate: int = 16_000
    spear_hop_samples: int = 640
    spear_model_id: str = "marcoyang/spear-xlarge-speech-audio"
    spear_revision: str = ""
    spear_layernorm: bool = True
    mel: MelConfig = field(default_factory=MelConfig)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.GroupNorm(1, width)
        self.conv = nn.Conv1d(
            width, 2 * width, kernel_size, padding=padding, dilation=dilation,
        )
        self.out = nn.Conv1d(width, width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.conv(F.silu(self.norm(x))).chunk(2, dim=1)
        update = self.out(value * torch.sigmoid(gate))
        return x + self.dropout(update)


class SpearMelBridge(nn.Module):
    """Temporal bridge from 25 Hz SPEAR features to vocoder-rate log-mels."""

    def __init__(self, config: BridgeConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        dilations = (1, 3, 9)
        self.blocks = nn.ModuleList([
            ResidualTemporalBlock(
                config.hidden_dim,
                config.kernel_size,
                dilations[index % len(dilations)],
                config.dropout,
            )
            for index in range(config.residual_layers)
        ])
        self.output_norm = nn.GroupNorm(1, config.hidden_dim)
        self.output_projection = nn.Conv1d(config.hidden_dim, config.mel.n_mels, 1)

    def forward(self, h: torch.Tensor, target_frames: int) -> torch.Tensor:
        if h.ndim != 3 or h.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected SPEAR features [B,T,{self.config.input_dim}], got {tuple(h.shape)}."
            )
        if int(target_frames) < 1:
            raise ValueError("target_frames must be positive.")
        x = self.input_projection(self.input_norm(h)).transpose(1, 2)
        # Interpolate the learned hidden trajectory rather than a predicted mel;
        # subsequent temporal blocks can correct local spectral transitions.
        x = F.interpolate(x, size=int(target_frames), mode="linear", align_corners=False)
        for block in self.blocks:
            x = block(x)
        return self.output_projection(F.silu(self.output_norm(x)))

    def target_frames_for_spear(self, spear_frames: int) -> int:
        duration = (
            int(spear_frames) * self.config.spear_hop_samples
            / self.config.spear_sample_rate
        )
        return max(1, int(round(duration * self.config.mel.sample_rate / self.config.mel.hop_length)))


class LogMelFrontend(nn.Module):
    """Differentiable log-mel frontend configured to match the vocoder."""

    def __init__(self, config: MelConfig) -> None:
        super().__init__()
        try:
            import torchaudio
        except ImportError as exc:
            raise AnalysisError("Audio bridge training requires torchaudio.") from exc
        self.config = config
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=1.0,
            center=config.center,
            pad=0,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio[None]
        if self.config.reflect_pad:
            amount = max(0, (self.config.n_fft - self.config.hop_length) // 2)
            if amount:
                audio = F.pad(audio.unsqueeze(1), (amount, amount), mode="reflect").squeeze(1)
        magnitude_mel = self.transform(audio)
        return torch.log(torch.clamp(magnitude_mel, min=self.config.log_floor))


def mel_frame_lengths(
    sample_lengths: torch.Tensor,
    config: MelConfig,
) -> torch.Tensor:
    lengths = sample_lengths.long()
    if config.reflect_pad:
        lengths = lengths + 2 * max(0, (config.n_fft - config.hop_length) // 2)
    if config.center:
        # torch.stft centre padding contributes n_fft // 2 on both sides.
        lengths = lengths + config.n_fft
    return torch.div(
        lengths - config.n_fft, config.hop_length, rounding_mode="floor",
    ).add(1).clamp(min=1)


def masked_bridge_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    *,
    delta_weight: float = .5,
    acceleration_weight: float = .25,
) -> tuple[torch.Tensor, dict[str, float]]:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target mismatch: {prediction.shape} != {target.shape}")
    positions = torch.arange(prediction.shape[-1], device=prediction.device)[None]
    mask = positions < lengths[:, None]

    def masked_l1(a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        expanded = valid[:, None].expand_as(a)
        selected = (a - b).abs().masked_select(expanded)
        return selected.mean() if selected.numel() else a.sum() * 0.0

    base = masked_l1(prediction, target, mask)
    first_prediction = prediction[..., 1:] - prediction[..., :-1]
    first_target = target[..., 1:] - target[..., :-1]
    first_mask = mask[:, 1:] & mask[:, :-1]
    first = masked_l1(first_prediction, first_target, first_mask)
    second_prediction = first_prediction[..., 1:] - first_prediction[..., :-1]
    second_target = first_target[..., 1:] - first_target[..., :-1]
    second_mask = first_mask[:, 1:] & first_mask[:, :-1]
    second = masked_l1(second_prediction, second_target, second_mask)
    loss = base + float(delta_weight) * first + float(acceleration_weight) * second
    return loss, {
        "mel_l1": float(base.detach()),
        "delta_l1": float(first.detach()),
        "acceleration_l1": float(second.detach()),
        "loss": float(loss.detach()),
    }


def bridge_domain_signature(config: BridgeConfig) -> dict[str, Any]:
    return {
        "input_dim": int(config.input_dim),
        "spear_sample_rate": int(config.spear_sample_rate),
        "spear_hop_samples": int(config.spear_hop_samples),
        "spear_model_id": str(config.spear_model_id),
        "spear_revision": str(config.spear_revision),
        "spear_layernorm": bool(config.spear_layernorm),
    }


def save_bridge(
    path: Path,
    model: SpearMelBridge,
    *,
    training: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "spear_logmel_bridge_v1",
        "config": {**asdict(model.config), "mel": asdict(model.config.mel)},
        "state_dict": model.state_dict(),
        "domain_signature": bridge_domain_signature(model.config),
        "training": training or {},
    }, path)


def _bridge_config(raw: dict[str, Any]) -> BridgeConfig:
    raw = dict(raw)
    raw["mel"] = MelConfig(**dict(raw.get("mel", {})))
    return BridgeConfig(**raw)


def load_bridge(path: Path, device: str | torch.device = "cpu") -> tuple[SpearMelBridge, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "spear_logmel_bridge_v1":
        raise AnalysisError(f"Unsupported audio bridge format in {path}.")
    model = SpearMelBridge(_bridge_config(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def validate_bridge_domain(config: BridgeConfig, resolved_config: dict[str, Any]) -> None:
    expected = bridge_domain_signature(config)
    actual = {
        "input_dim": int(resolved_config.get("D", -1)),
        "spear_sample_rate": int(resolved_config.get("sample_rate", -1)),
        "spear_hop_samples": 640,
        "spear_model_id": str(resolved_config.get("spear_model_id", "")),
        "spear_revision": str(resolved_config.get("spear_revision", "")),
        "spear_layernorm": bool(resolved_config.get("spear_layernorm", False)),
    }
    mismatches = {
        key: {"bridge": expected[key], "checkpoint": actual[key]}
        for key in expected if expected[key] != actual[key]
    }
    if mismatches:
        raise AnalysisError(f"Audio bridge/checkpoint feature-domain mismatch: {mismatches}")
