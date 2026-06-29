from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .types import ResolvedModel
from .utils import AnalysisError, read_structured


CRITICAL_DEFAULTS = {
    "sample_rate": 16000,
    "spear_model_id": "marcoyang/spear-xlarge-speech-audio",
    "routing_tau_end": 0.1,
}


def _unwrap(payload: Any) -> tuple[dict[str, Any], str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise AnalysisError("Checkpoint must contain a state dictionary or checkpoint mapping.")
    metadata: dict[str, Any] = {}
    if isinstance(payload.get("analysis_config"), dict):
        metadata.update(payload["analysis_config"])
    if isinstance(payload.get("config"), dict):
        metadata = {**payload["config"], **metadata}
    if isinstance(payload.get("model"), dict):
        return payload["model"], "msp:model", metadata
    if isinstance(payload.get("model_state"), dict):
        return payload["model_state"], "legacy:model_state", metadata
    if payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload, "raw_state_dict", metadata
    raise AnalysisError("Checkpoint has no 'model', 'model_state', or raw tensor state dictionary.")


def _normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[7:]
        # SPEAR is frozen and reloaded from spear_model_id. Keeping a second
        # full backbone in the checkpoint mapping can otherwise double host RAM.
        if key.startswith("encoder._spear."):
            continue
        out[key] = value
    return out


def _shape(state: dict[str, Any], key: str) -> tuple[int, ...] | None:
    value = state.get(key)
    return tuple(value.shape) if torch.is_tensor(value) else None


def _infer_structural(state: dict[str, Any]) -> dict[str, Any]:
    enc = _shape(state, "sae.enc_weight")
    if not enc or len(enc) != 2:
        raise AnalysisError("Checkpoint is missing two-dimensional sae.enc_weight.")
    K, D = enc
    inferred: dict[str, Any] = {"K": K, "D": D}
    routing = _shape(state, "routing.logits")
    if routing:
        inferred["n_routes"] = routing[1]
    inferred["has_routing"] = routing is not None or "block_idx" in state
    inferred["fixed_blocks"] = "block_idx" in state or "block_m_L" in state
    inferred["routing_dynamic"] = any(k.startswith("routing.router.") for k in state)
    inferred["projection_disentanglement"] = any(k.startswith("proj_L.") for k in state)
    inferred["projection_reconstruct"] = any(k.startswith("up_L.") for k in state)
    inferred["prosody"] = any(k.startswith("prosody_head.") for k in state)
    inferred["emotion"] = any(k.startswith("emotion_head.") for k in state)
    inferred["has_z_u"] = inferred.get("n_routes", 0) == 3 or "block_m_U" in state
    sid = _shape(state, "sid_head.fc.weight")
    if sid:
        inferred["num_speakers"] = sid[0]
    vocab = _shape(state, "pr_head.fc.weight")
    if vocab:
        inferred["vocab_size"] = vocab[0]
    emo = _shape(state, "emotion_head.fc.weight")
    if emo:
        inferred["emotion_num_classes"] = emo[0]

    if inferred["projection_disentanglement"]:
        linear = _shape(state, "proj_L.proj.weight")
        first = _shape(state, "proj_L.proj.0.weight")
        last = _shape(state, "proj_L.proj.2.weight")
        if linear:
            inferred.update(projection_dim=linear[0], projection_nonlinear=False)
        elif first and last:
            inferred.update(projection_dim=last[0], projection_hidden=first[0], projection_nonlinear=True)
    if "vib_logvar" in state:
        inferred["vib_zL_weight"] = 1.0
    return inferred


def _project_metadata(checkpoint: Path) -> dict[str, Any]:
    """Best-effort lookup in this repository's generated experiment index."""
    root = Path(__file__).resolve().parents[1]
    index = root / "Disentanglement" / "experiments.json"
    if not index.exists():
        return {}
    try:
        entries = json.loads(index.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(entries, dict):
        entries = entries.get("experiments", entries.get("runs", []))
    if not isinstance(entries, list):
        return {}
    cp = str(checkpoint)
    name = checkpoint.parent.name
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        recorded = str(entry.get("checkpoint", ""))
        if recorded and (recorded in cp or Path(recorded).parent.name == name):
            text = str(entry.get("config", ""))
            out: dict[str, Any] = {}
            match = re.search(r"spear_ln=(True|False)", text)
            if match:
                out["spear_layernorm"] = match.group(1) == "True"
            match = re.search(r"hard_gumbel=(True|False)", text)
            if match:
                out["hard_gumbel_routing"] = match.group(1) == "True"
            return out
    return {}


def load_checkpoint(path: str | Path) -> ResolvedModel:
    checkpoint = Path(path).resolve()
    if not checkpoint.exists():
        raise AnalysisError(f"Checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state, source_format, embedded = _unwrap(payload)
    state = _normalize_state(state)
    structural = _infer_structural(state)

    sidecar = {}
    candidates = [
        checkpoint.with_suffix(checkpoint.suffix + ".analysis.yaml"),
        checkpoint.with_suffix(".analysis.yaml"),
        checkpoint.parent / "analysis_config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            sidecar = read_structured(candidate)
            break

    config = {**CRITICAL_DEFAULTS, **structural, **_project_metadata(checkpoint), **embedded, **sidecar}
    warnings: list[str] = []
    # The MSP trainer has a stable extraction contract even though old files did
    # not serialize it.
    if source_format == "msp:model":
        config.setdefault("topk", 256)
        config.setdefault("spear_layernorm", True)
        config.setdefault("hard_gumbel_routing", True)
        config.setdefault("n_routes", 2)
        config.setdefault("routing_tau_end", 0.1)

    capabilities = {
        "features": True,
        "routes": bool(structural["has_routing"]),
        "unit_routes": bool(structural["has_routing"] and not structural["projection_disentanglement"]),
        "causal": bool(structural["has_routing"] and not structural["projection_disentanglement"]),
        "swap": bool(structural["has_routing"] and not structural["projection_disentanglement"]),
        "projection_views": bool(structural["projection_disentanglement"]),
    }
    if structural["projection_disentanglement"]:
        warnings.append("Projection checkpoint: SAE units do not have one-to-one L/P assignments.")
    if not structural["has_routing"]:
        warnings.append("Checkpoint has no trained route state; route-dependent analyses are unavailable.")
    return ResolvedModel(checkpoint, state, config, source_format, capabilities, warnings)


def unresolved_critical(resolved: ResolvedModel) -> list[str]:
    return [name for name in ("topk", "spear_layernorm") if name not in resolved.config]


def route_information(resolved: ResolvedModel) -> tuple[np.ndarray, np.ndarray]:
    """Return dominant route and route probability per SAE unit."""
    state, cfg = resolved.state, resolved.config
    K = int(cfg["K"])
    if "block_idx" in state:
        route = state["block_idx"].detach().cpu().numpy().astype(np.int16)
        probs = np.ones(K, dtype=np.float32)
        return route, probs
    if "routing.logits" in state:
        logits = state["routing.logits"].detach().float()
        p = torch.softmax(logits, dim=-1)
        return p.argmax(-1).cpu().numpy().astype(np.int16), p.max(-1).values.cpu().numpy()
    return np.full(K, -1, dtype=np.int16), np.zeros(K, dtype=np.float32)


def build_model(resolved: ResolvedModel, device: str):
    """Build the existing speech model while keeping all analysis code outside it."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from Disentanglement.config import DISConfig
        from Disentanglement.model import build_dis_model
    except Exception as exc:
        raise AnalysisError(f"Could not import the Disentanglement model: {exc}") from exc

    c = DISConfig()
    cfg = resolved.config
    for key, value in cfg.items():
        try:
            setattr(c, key, value)
        except Exception:
            pass
    c.device = device
    c.K, c.D = int(cfg["K"]), int(cfg["D"])
    c.topk = int(cfg.get("topk", min(256, c.K)))
    c.n_routes = int(cfg.get("n_routes", 2))
    c.num_speakers = int(cfg.get("num_speakers", getattr(c, "num_speakers", 1)))
    c.vocab_size = int(cfg.get("vocab_size", getattr(c, "vocab_size", 2)))
    c.emotion_num_classes = int(cfg.get("emotion_num_classes", 4))
    c.prosody = bool(cfg.get("prosody", False))
    c.emotion = bool(cfg.get("emotion", False))
    c.grl_phoneme_weight = 1.0 if any(k.startswith("pr_grl_head.") for k in resolved.state) else 0.0
    c.grl_prosody_weight = 1.0 if any(k.startswith("prosody_grl_head.") for k in resolved.state) else 0.0
    c.grl_emotion_weight = 1.0 if any(k.startswith("emotion_grl_head.") for k in resolved.state) else 0.0
    c.projection_disentanglement = bool(cfg.get("projection_disentanglement", False))
    c.projection_reconstruct = bool(cfg.get("projection_reconstruct", False))
    c.projection_nonlinear = bool(cfg.get("projection_nonlinear", False))
    c.projection_dim = int(cfg.get("projection_dim", 128))
    c.projection_hidden = int(cfg.get("projection_hidden", 512))
    c.routing_dynamic = bool(cfg.get("routing_dynamic", False))
    # Raw-unit extraction handles fixed block Top-K itself. Avoid requiring
    # non-serialized block budgets merely to instantiate the shared model.
    c.fixed_blocks = False
    c.no_routing = False
    c.spear_layernorm = bool(cfg.get("spear_layernorm", False))

    model = build_dis_model(c).to(device)
    current = model.state_dict()
    compatible = {k: v for k, v in resolved.state.items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    for required in ("sae.enc_weight", "sae.dec_weight", "sae.b_pre"):
        if required not in compatible:
            raise AnalysisError(f"Checkpoint could not load required tensor {required}.")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
