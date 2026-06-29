from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class AnalysisError(RuntimeError):
    """An actionable input/configuration error, safe to show at the CLI."""


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def read_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise AnalysisError(
                f"{path} uses YAML syntax, but PyYAML is not installed. "
                "Install SAEUnitAnalysis/requirements.txt or use JSON syntax."
            ) from exc
        obj = yaml.safe_load(text)
    if not isinstance(obj, dict):
        raise AnalysisError(f"{path} must contain a mapping/object at its root.")
    return obj


def fingerprint(paths: Iterable[Path], extra: Any = None) -> str:
    h = hashlib.sha256()
    for path in paths:
        p = Path(path)
        h.update(str(p.resolve()).encode())
        if p.exists():
            stat = p.stat()
            h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
            if stat.st_size < 10_000_000:
                h.update(p.read_bytes())
    if extra is not None:
        h.update(json.dumps(jsonable(extra), sort_keys=True).encode())
    return h.hexdigest()[:16]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k, "")) for k in fields})


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    flat = p.reshape(-1)
    order = np.argsort(flat)
    ranked = flat[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out.reshape(p.shape)


def bootstrap_ci(values: np.ndarray, seed: int = 42, n: int = 500) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))

