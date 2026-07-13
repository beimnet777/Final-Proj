"""Checkpoint helpers specific to MSP initialization.

Exact resume is handled by :mod:`training_runtime`.  This module deliberately
handles only cross-run SAE initialization: task heads are dataset-specific and
must never be imported accidentally from a LibriSpeech or differently-sized MSP
run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch


def checkpoint_model_state(payload: Mapping) -> Mapping[str, torch.Tensor]:
    """Return model weights from current, legacy, or raw state-dict files."""
    if "model_state" in payload:
        return payload["model_state"]
    if "model" in payload:
        return payload["model"]
    return payload


def load_sae_initialization(model: torch.nn.Module, checkpoint: str | Path) -> dict:
    """Load only shape-compatible ``sae.*`` tensors from ``checkpoint``.

    Returns a small audit record and raises if no SAE tensors were loaded.  This
    makes a wrong checkpoint format or architecture mismatch visible instead of
    allowing a silently-from-scratch run.
    """
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint must contain a mapping, got {type(payload).__name__}")
    state = checkpoint_model_state(payload)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint model state must be a mapping")

    current = model.state_dict()
    candidates = {k: v for k, v in state.items() if str(k).startswith("sae.")}
    compatible = {
        k: v for k, v in candidates.items()
        if k in current and torch.is_tensor(v) and tuple(v.shape) == tuple(current[k].shape)
    }
    mismatched = sorted(k for k in candidates if k in current and k not in compatible)
    unexpected = sorted(k for k in candidates if k not in current)
    if not compatible:
        raise ValueError(
            f"no shape-compatible SAE tensors found in {checkpoint}; "
            f"SAE candidates={len(candidates)} mismatched={len(mismatched)}")

    result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": sorted(compatible),
        "mismatched": mismatched,
        "unexpected": unexpected,
        "missing_model_keys": list(result.missing_keys),
        "source_step": payload.get("step"),
        "source_format": ("model_state" if "model_state" in payload else
                          "model" if "model" in payload else "raw_state_dict"),
    }
