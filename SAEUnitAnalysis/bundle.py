from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from .types import BundleSpec, FactorSpec
from .utils import AnalysisError, read_structured, write_json


STANDARD_FACTORS = {
    "phone": FactorSpec("phone", "linguistic", "frame", "categorical", "alignment"),
    "speaker_id": FactorSpec("speaker_id", "paralinguistic", "utterance", "categorical", "speaker_id"),
}


def _table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise AnalysisError(f"Required table not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise AnalysisError(f"Reading {path.name} requires pyarrow.") from exc
    return pd.read_csv(path)


class AnalysisBundle:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        config_path = self.root / "dataset.yaml"
        if not config_path.exists():
            raise AnalysisError(f"Analysis bundle is missing {config_path}.")
        raw = read_structured(config_path)
        if int(raw.get("schema_version", 1)) != 1:
            raise AnalysisError("Only analysis-bundle schema_version=1 is supported.")

        manifest_name = raw.get("manifest")
        if not manifest_name:
            manifest_name = "utterances.parquet" if (self.root / "utterances.parquet").exists() else "utterances.csv"
        align_name = raw.get("alignments")
        if align_name is None:
            if (self.root / "alignments.parquet").exists():
                align_name = "alignments.parquet"
            elif (self.root / "alignments.csv").exists():
                align_name = "alignments.csv"

        factors = self._parse_factors(raw.get("factors", []))
        self.spec = BundleSpec(
            root=self.root,
            sample_rate=int(raw.get("sample_rate", 16000)),
            manifest_path=self.root / manifest_name,
            alignments_path=(self.root / align_name) if align_name else None,
            split_map={"train": "train", "validation": "val", "test": "test", **raw.get("splits", {})},
            factors=factors,
            raw=raw,
        )
        self.utterances = _table(self.spec.manifest_path)
        self.alignments = _table(self.spec.alignments_path) if self.spec.alignments_path else None
        self._validate()

    def _parse_factors(self, declared: list[dict[str, Any]]) -> list[FactorSpec]:
        if declared:
            out = []
            for f in declared:
                out.append(FactorSpec(
                    name=str(f["name"]), family=str(f["family"]),
                    level=str(f.get("level", "utterance")),
                    kind=str(f.get("type", f.get("kind", "categorical"))),
                    source=str(f.get("source", f["name"])),
                ))
            return out
        # Automatic standard core: the dissertation-facing Libri analysis is
        # intentionally phone-vs-speaker, not a broad metadata/prosody sweep.
        # Extra factors can still be declared explicitly in dataset.yaml and
        # enabled with --factor-scope broad.
        return list(STANDARD_FACTORS.values())

    def _validate(self) -> None:
        required = {"utterance_id", "audio_path", "split", "transcript"}
        missing = required - set(self.utterances.columns)
        if missing:
            raise AnalysisError(f"Manifest is missing required columns: {sorted(missing)}")
        if self.utterances["utterance_id"].duplicated().any():
            raise AnalysisError("Manifest utterance_id values must be unique.")
        self.utterances["utterance_id"] = self.utterances["utterance_id"].astype(str)
        for i, row in self.utterances.iterrows():
            path = self.audio_path(row)
            if not path.exists():
                raise AnalysisError(f"Audio file for {row['utterance_id']} does not exist: {path}")
        if self.alignments is not None:
            cols = {"utterance_id", "start_sec", "end_sec", "phone"}
            miss = cols - set(self.alignments.columns)
            if miss:
                raise AnalysisError(f"Alignments are missing columns: {sorted(miss)}")
            self.alignments["utterance_id"] = self.alignments["utterance_id"].astype(str)
            self.alignments["start_sec"] = pd.to_numeric(self.alignments["start_sec"], errors="coerce")
            self.alignments["end_sec"] = pd.to_numeric(self.alignments["end_sec"], errors="coerce")
            if self.alignments[["start_sec", "end_sec"]].isna().any().any():
                raise AnalysisError("Alignment start_sec/end_sec values must be finite numbers.")
            if (self.alignments["start_sec"] < 0).any():
                raise AnalysisError("Alignment start_sec values must be non-negative.")
            phone_text = self.alignments["phone"].fillna("").astype(str).str.strip()
            if phone_text.eq("").any():
                raise AnalysisError("Alignment phone labels must be non-empty.")
            self.alignments["phone"] = phone_text
            known = set(self.utterances["utterance_id"])
            unknown = set(self.alignments["utterance_id"]) - known
            if unknown:
                raise AnalysisError(f"Alignments reference {len(unknown)} unknown utterances.")
            if (self.alignments["end_sec"] <= self.alignments["start_sec"]).any():
                raise AnalysisError("Every alignment must have end_sec > start_sec.")
            if self.alignments.duplicated(["utterance_id", "start_sec", "end_sec", "phone"]).any():
                raise AnalysisError("Alignment rows must not contain exact duplicates.")
            for utterance_id, group in self.alignments.groupby("utterance_id", sort=False):
                ordered = group.sort_values(["start_sec", "end_sec"])
                previous_end = ordered["end_sec"].shift(1)
                if (ordered["start_sec"] < previous_end - 1e-5).fillna(False).any():
                    raise AnalysisError(f"Phone alignments overlap for utterance {utterance_id}.")

        available = []
        for f in self.spec.factors:
            if f.source.startswith("computed:"):
                available.append(f)
            elif f.source == "alignment" and self.alignments is not None:
                available.append(f)
            elif f.source in self.utterances.columns:
                available.append(f)
        self.spec.factors = available

    def audio_path(self, row: pd.Series | dict) -> Path:
        path = Path(str(row["audio_path"]))
        return path if path.is_absolute() else self.root / path

    def split(self, logical: str) -> pd.DataFrame:
        value = self.spec.split_map.get(logical, logical)
        return self.utterances[self.utterances["split"].astype(str) == str(value)].copy()

    def require(self, analysis: str) -> None:
        factor_names = {f.name for f in self.spec.factors}
        if analysis in {"factor_metrics", "causal", "swap", "all"}:
            if "phone" not in factor_names:
                raise AnalysisError(f"Analysis '{analysis}' requires independent phone alignments.")
        if analysis in {"factor_metrics", "causal", "swap", "all"}:
            for split in ("train", "validation", "test"):
                if self.split(split).empty:
                    raise AnalysisError(f"Analysis '{analysis}' requires a non-empty {split} split.")
            if "speaker_id" not in factor_names:
                raise AnalysisError(f"Analysis '{analysis}' requires speaker_id labels.")

    def validation_report(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "sample_rate": self.spec.sample_rate,
            "utterances": int(len(self.utterances)),
            "alignments": int(len(self.alignments)) if self.alignments is not None else 0,
            "splits": self.utterances["split"].astype(str).value_counts().to_dict(),
            "factors": [f.__dict__ for f in self.spec.factors],
        }

    def write_validation_report(self, path: Path) -> None:
        write_json(path, self.validation_report())
