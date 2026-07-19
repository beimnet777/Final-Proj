from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ANALYSES = (
    "deadness", "health", "atlas", "selectivity", "factor_metrics", "clustering", "similarity",
    "geometry", "causal", "swap",
)


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    level: str
    kind: str
    source: str


@dataclass
class BundleSpec:
    root: Path
    sample_rate: int
    manifest_path: Path
    alignments_path: Path | None
    split_map: dict[str, str]
    factors: list[FactorSpec]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedModel:
    checkpoint: Path
    state: dict[str, Any]
    config: dict[str, Any]
    source_format: str
    capabilities: dict[str, bool]
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    output_dir: Path
    completed: list[str]
    artifacts: dict[str, Path]
    warnings: list[str]
    summary: dict[str, Any]
