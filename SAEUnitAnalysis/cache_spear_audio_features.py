"""Create one compact, indexed SPEAR feature cache for direct vocoder training."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .bundle import AnalysisBundle
from .checkpoint import load_checkpoint
from .extraction import _speaker_balanced_sample, calibrate
from .train_audio_bridge import BundleAudioDataset, _collate, _device
from .utils import AnalysisError, fingerprint, set_seed


FEATURE_FILE = "features.f16"
INDEX_FILE = "index.csv"
MANIFEST_FILE = "manifest.json"


def _selected_rows(
    bundle: AnalysisBundle,
    max_train_utterances: int,
    max_validation_utterances: int,
    seed: int,
) -> pd.DataFrame:
    parts = []
    for logical, limit, offset in (
        ("train", max_train_utterances, 0),
        ("validation", max_validation_utterances, 1),
    ):
        rows = bundle.split(logical)
        if int(limit) > 0 and len(rows) > int(limit):
            rows = _speaker_balanced_sample(rows, n=int(limit), seed=seed + offset)
        rows = rows.copy()
        rows["logical_split"] = logical
        parts.append(rows)
    selected = pd.concat(parts, ignore_index=True)
    if selected.empty or not (selected["logical_split"] == "train").any():
        raise AnalysisError("Direct vocoder cache requires a non-empty training split.")
    if not (selected["logical_split"] == "validation").any():
        raise AnalysisError("Direct vocoder cache requires a non-empty validation split.")
    return selected


def _cache_key(bundle: AnalysisBundle, rows: pd.DataFrame, config: dict[str, Any]) -> str:
    return fingerprint(
        [bundle.spec.manifest_path, bundle.root / "dataset.yaml"],
        extra={
            "utterance_ids": rows["utterance_id"].astype(str).tolist(),
            "sample_rate": int(bundle.spec.sample_rate),
            "spear_hop_samples": 320,
            "input_dim": int(config["D"]),
            "spear_model_id": str(config.get("spear_model_id", "")),
            "spear_revision": str(config.get("spear_revision", "")),
            "spear_layernorm": bool(config.get("spear_layernorm", False)),
        },
    )


@torch.no_grad()
def build_spear_audio_cache(
    checkpoint: Path,
    data_root: Path,
    output_dir: Path,
    *,
    device: str | None = None,
    batch_size: int = 4,
    max_train_utterances: int = 0,
    max_validation_utterances: int = 0,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    set_seed(seed)
    device = _device(device)
    bundle = AnalysisBundle(data_root)
    resolved = load_checkpoint(checkpoint)
    model = calibrate(resolved, bundle, device)
    model.eval()
    rows = _selected_rows(
        bundle, max_train_utterances, max_validation_utterances, seed,
    )
    key = _cache_key(bundle, rows, resolved.config)
    output_dir = output_dir.resolve()
    manifest_path = output_dir / MANIFEST_FILE
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("cache_key") != key:
            raise AnalysisError(
                f"Existing cache at {output_dir} belongs to another data/domain selection; "
                "choose a new directory or pass --overwrite."
            )
        feature_path = output_dir / FEATURE_FILE
        index_path = output_dir / INDEX_FILE
        if feature_path.exists() and index_path.exists():
            print(f"[spear-cache] compatible cache already complete: {output_dir}", flush=True)
            return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_feature = output_dir / f".{FEATURE_FILE}.tmp"
    temporary_index = output_dir / f".{INDEX_FILE}.tmp"
    temporary_manifest = output_dir / f".{MANIFEST_FILE}.tmp"
    for path in (temporary_feature, temporary_index, temporary_manifest):
        path.unlink(missing_ok=True)

    dataset = BundleAudioDataset(bundle, rows)
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=0,
        collate_fn=_collate,
    )
    row_lookup = rows.set_index(rows["utterance_id"].astype(str), drop=False)
    records: list[dict[str, Any]] = []
    offset = 0
    with temporary_feature.open("wb") as feature_stream:
        completed = 0
        for audio, lengths, utterance_ids in loader:
            audio = audio.to(device)
            lengths = lengths.to(device)
            features, feature_lengths = model.encoder(audio, lengths)
            for index, utterance_id in enumerate(utterance_ids):
                n_frames = int(feature_lengths[index].item())
                values = (
                    features[index, :n_frames]
                    .detach().float().cpu().numpy().astype("<f2", copy=False)
                )
                values.tofile(feature_stream)
                source = row_lookup.loc[str(utterance_id)]
                records.append({
                    "utterance_id": str(utterance_id),
                    "logical_split": str(source["logical_split"]),
                    "speaker_id": str(source.get("speaker_id", "")),
                    "offset_frames": int(offset),
                    "n_frames": int(n_frames),
                    "audio_samples": int(lengths[index].item()),
                    "audio_path": str(source["audio_path"]),
                })
                offset += n_frames
            completed += len(utterance_ids)
            print(
                f"[spear-cache] extracted {completed}/{len(rows)} utterances "
                f"({offset:,} frames)", flush=True,
            )

    pd.DataFrame(records).to_csv(temporary_index, index=False)
    expected_bytes = int(offset) * int(resolved.config["D"]) * np.dtype("<f2").itemsize
    actual_bytes = temporary_feature.stat().st_size
    if expected_bytes != actual_bytes:
        raise AnalysisError(
            f"Incomplete feature cache: wrote {actual_bytes} bytes, expected {expected_bytes}."
        )
    implied_hops = np.asarray([
        float(record["audio_samples"]) / max(int(record["n_frames"]), 1)
        for record in records
    ])
    observed_hop = float(np.median(implied_hops))
    if not 304.0 <= observed_hop <= 336.0:
        raise AnalysisError(
            f"Observed SPEAR timing is {observed_hop:.2f} audio samples/frame; "
            "the direct HiFi-GAN contract expects approximately 320."
        )
    manifest = {
        "format": "spear_audio_feature_cache_v1",
        "cache_key": key,
        "checkpoint_used_only_for_spear_domain": str(resolved.checkpoint),
        "data": str(bundle.root),
        "feature_file": FEATURE_FILE,
        "index_file": INDEX_FILE,
        "dtype": "float16",
        "input_dim": int(resolved.config["D"]),
        "sample_rate": int(bundle.spec.sample_rate),
        "spear_hop_samples": 320,
        "observed_median_samples_per_frame": observed_hop,
        "spear_model_id": str(resolved.config.get("spear_model_id", "")),
        "spear_revision": str(resolved.config.get("spear_revision", "")),
        "spear_layernorm": bool(resolved.config.get("spear_layernorm", False)),
        "total_frames": int(offset),
        "feature_bytes": int(actual_bytes),
        "train_utterances": int(sum(record["logical_split"] == "train" for record in records)),
        "validation_utterances": int(sum(record["logical_split"] == "validation" for record in records)),
        "test_utterances": 0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "seed": int(seed),
    }
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_feature, output_dir / FEATURE_FILE)
    os.replace(temporary_index, output_dir / INDEX_FILE)
    os.replace(temporary_manifest, output_dir / MANIFEST_FILE)
    print(
        f"[spear-cache] complete: {output_dir} "
        f"({actual_bytes / 1024**3:.2f} GiB, {offset:,} frames)", flush=True,
    )
    return output_dir


class SpearAudioFeatureCache:
    """Read-only memory-mapped access to cached SPEAR frames and source audio."""

    def __init__(self, root: Path, data_root: Path | None = None) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / MANIFEST_FILE
        if not manifest_path.exists():
            raise AnalysisError(f"SPEAR audio cache is missing {manifest_path}.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "spear_audio_feature_cache_v1":
            raise AnalysisError(f"Unsupported SPEAR audio cache format at {self.root}.")
        self.index = pd.read_csv(self.root / self.manifest["index_file"], dtype={"speaker_id": str})
        self.input_dim = int(self.manifest["input_dim"])
        total_values = int(self.manifest["total_frames"]) * self.input_dim
        self.features = np.memmap(
            self.root / self.manifest["feature_file"], dtype="<f2", mode="r",
            shape=(total_values,),
        )
        self.data_root = Path(data_root or self.manifest["data"]).resolve()

    def split(self, logical_split: str) -> pd.DataFrame:
        return self.index[self.index["logical_split"] == logical_split].reset_index(drop=True)

    def feature(self, row: pd.Series | dict[str, Any]) -> np.ndarray:
        start = int(row["offset_frames"]) * self.input_dim
        count = int(row["n_frames"]) * self.input_dim
        return np.asarray(self.features[start:start + count]).reshape(-1, self.input_dim)

    def audio_path(self, row: pd.Series | dict[str, Any]) -> Path:
        path = Path(str(row["audio_path"]))
        return path if path.is_absolute() else self.data_root / path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache SPEAR features for direct HiFi-GAN training.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-utterances", type=int, default=0)
    parser.add_argument("--max-validation-utterances", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        build_spear_audio_cache(
            args.checkpoint, args.data, args.output_dir,
            device=args.device, batch_size=args.batch_size,
            max_train_utterances=args.max_train_utterances,
            max_validation_utterances=args.max_validation_utterances,
            seed=args.seed, overwrite=args.overwrite,
        )
    except AnalysisError as exc:
        raise SystemExit(f"[spear-cache] ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
