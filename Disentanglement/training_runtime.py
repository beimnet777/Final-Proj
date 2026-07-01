"""Small, trainer-agnostic helpers for resumable local/Colab experiments."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Sampler

FORMAT_VERSION = 2


def resolve_amp_precision(requested: str, *, cuda_available: Optional[bool] = None,
                          bf16_supported: Optional[bool] = None) -> tuple[bool, bool]:
    """Return ``(use_bf16, use_fp16)`` for a requested precision policy."""
    requested = str(requested).lower()
    if requested not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError(f"unknown precision {requested!r}")
    cuda = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    bf16 = (torch.cuda.is_bf16_supported() if bf16_supported is None
            else bool(bf16_supported)) if cuda else False
    if requested == "fp32":
        return False, False
    if requested == "bf16":
        if not cuda or not bf16:
            raise RuntimeError("bf16 requested but the CUDA device does not support it")
        return True, False
    if requested == "fp16":
        if not cuda:
            raise RuntimeError("fp16 requested but CUDA is unavailable")
        return False, True
    # auto: BF16 on capable GPUs, otherwise FP16 on CUDA, FP32 on CPU.
    return (True, False) if bf16 else ((False, True) if cuda else (False, False))


class StatefulRandomSampler(Sampler[int]):
    """Random sampler whose permutation/cursor can be checkpointed exactly."""
    def __init__(self, data_source, seed: int = 0) -> None:
        self.data_source = data_source
        self.generator = torch.Generator().manual_seed(int(seed))
        self.permutation: list[int] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.position >= len(self.permutation) or len(self.permutation) != len(self.data_source):
            self.permutation = torch.randperm(len(self.data_source), generator=self.generator).tolist()
            self.position = 0
        while self.position < len(self.permutation):
            index = self.permutation[self.position]
            self.position += 1
            yield index

    def state_dict(self) -> dict[str, Any]:
        return {"permutation": self.permutation, "position": self.position,
                "generator_state": self.generator.get_state()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.permutation = list(state.get("permutation", []))
        self.position = int(state.get("position", 0))
        if state.get("generator_state") is not None:
            self.generator.set_state(state["generator_state"])


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def config_dict(cfg: Any) -> dict[str, Any]:
    source = asdict(cfg) if is_dataclass(cfg) else vars(cfg)
    return {k: jsonable(v) for k, v in source.items() if not k.startswith("_")}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p)):
        h.update(str(path).encode())
        if path.is_file():
            h.update(str(path.stat().st_size).encode())
            h.update(sha256_file(path).encode())
        elif path.exists():
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                rel = child.relative_to(path)
                stat = child.stat()
                h.update(str(rel).encode())
                h.update(str(stat.st_size).encode())
    return h.hexdigest()


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_rng_byte_tensor(value: Any) -> torch.Tensor:
    """Canonicalize a serialized RNG state for Torch's RNG setter APIs.

    Loading a checkpoint with ``map_location='cuda'`` also moves the RNG-state
    tensor, but ``torch.set_rng_state`` and ``torch.cuda.set_rng_state_all``
    require CPU ByteTensors. Lists/arrays are accepted for compatibility with
    checkpoints that passed through a non-Torch serializer.
    """
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu")


def restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_rng_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([
            _cpu_rng_byte_tensor(cuda_state) for cuda_state in state["cuda"]
        ])


def git_commit(root: Optional[Path] = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_metrics(path: Path, values: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(values), sort_keys=True) + "\n")
        f.flush()


def mirror_file(source: Path, mirror_dir: Optional[Path]) -> Optional[Path]:
    if not mirror_dir:
        return None
    source, mirror_dir = Path(source), Path(mirror_dir)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    target = mirror_dir / source.name
    tmp = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
    shutil.copy2(source, tmp)
    os.replace(tmp, target)
    return target


def compact_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    prefixes = ("encoder._spear.", "spear.", "encoder.spear.")
    return {k: v for k, v in model.state_dict().items()
            if not any(k.startswith(prefix) for prefix in prefixes)}


def checkpoint_payload(
    *, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
    step: int, best_metric: float, cfg: Any, scheduler: Any = None,
    scaler: Any = None, auxiliary: Optional[Mapping[str, Any]] = None,
    dataset_hash: str = "", preset: str = "", kind: str = "resume",
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "step": int(step),
        "best_metric": float(best_metric),
        "model_state": compact_model_state(model),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": rng_state(),
        "auxiliary": dict(auxiliary or {}),
        "analysis_config": config_dict(cfg),
        "dataset_fingerprint": dataset_hash,
        "preset": preset,
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "saved_at_unix": time.time(),
    }


def validate_resume(checkpoint: Mapping[str, Any], *, dataset_hash: str,
                    preset: str, cfg: Any) -> None:
    if int(checkpoint.get("format_version", 0)) < FORMAT_VERSION:
        raise ValueError("resume checkpoint is legacy/inference-only; exact resume requires format_version=2")
    got_hash = str(checkpoint.get("dataset_fingerprint", ""))
    if dataset_hash and got_hash != dataset_hash:
        raise ValueError(f"dataset fingerprint mismatch: checkpoint={got_hash} current={dataset_hash}")
    got_preset = str(checkpoint.get("preset", ""))
    if preset and got_preset != preset:
        raise ValueError(f"preset mismatch: checkpoint={got_preset!r} current={preset!r}")
    old = checkpoint.get("analysis_config", {})
    now = config_dict(cfg)
    for key in ("K", "topk", "n_routes", "hard_gumbel_routing", "spear_model_id", "spear_revision",
                "batch_size", "gradient_accumulation_steps", "precision",
                "club_enabled", "club_grad_norm", "club_grad_norm_target"):
        if key in old and key in now and old[key] != now[key]:
            raise ValueError(f"resume configuration mismatch for {key}: {old[key]!r} != {now[key]!r}")
    # Checkpoints created before CLUB gradient normalisation existed have no
    # key and therefore represent club_grad_norm=False. Never silently turn the
    # new backward rule on halfway through an exact-resume run.
    if bool(now.get("club_grad_norm", False)) and not bool(old.get("club_grad_norm", False)):
        raise ValueError(
            "resume configuration mismatch for club_grad_norm: checkpoint=False current=True")


def restore_training_state(checkpoint: Mapping[str, Any], *, model: torch.nn.Module,
                           optimizer=None, scheduler=None, scaler=None) -> tuple[int, float]:
    model.load_state_dict(checkpoint["model_state"], strict=False)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    restore_rng_state(checkpoint.get("rng_state"))
    return int(checkpoint.get("step", 0)), float(checkpoint.get("best_metric", float("inf")))


class SegmentLimit:
    def __init__(self, start_step: int, segment_steps: int = 0,
                 max_runtime_minutes: float = 0.0) -> None:
        self.last_step = (start_step + segment_steps) if segment_steps > 0 else None
        self.deadline = (time.monotonic() + 60.0 * max_runtime_minutes
                         if max_runtime_minutes > 0 else None)

    def reached(self, step: int) -> bool:
        return ((self.last_step is not None and step >= self.last_step)
                or (self.deadline is not None and time.monotonic() >= self.deadline))


def resolve_microbatch(requested: str, effective: int) -> tuple[int, int]:
    if requested != "auto":
        micro = int(requested)
    elif not torch.cuda.is_available():
        micro = min(2, effective)
    else:
        gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        micro = 2 if gib < 20 else 4 if gib < 40 else 8 if gib < 70 else effective
    if micro <= 0 or effective % micro:
        raise ValueError("microbatch size must be positive and divide effective_batch_size exactly")
    return micro, effective // micro
