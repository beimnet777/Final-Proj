"""Train a checkpoint-independent SPEAR-to-log-mel inversion bridge."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .audio_bridge import (
    BridgeConfig,
    LogMelFrontend,
    MelConfig,
    SpearMelBridge,
    masked_bridge_loss,
    save_bridge,
)
from .bundle import AnalysisBundle
from .checkpoint import load_checkpoint
from .extraction import _batch_audio, _read_audio, _speaker_balanced_sample, calibrate
from .utils import AnalysisError, set_seed, write_json


class BundleAudioDataset(Dataset):
    def __init__(self, bundle: AnalysisBundle, rows: pd.DataFrame) -> None:
        self.bundle = bundle
        self.rows = rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, str]:
        row = self.rows.iloc[int(index)]
        return (
            _read_audio(self.bundle.audio_path(row), self.bundle.spec.sample_rate),
            str(row["utterance_id"]),
        )


def _collate(items: list[tuple[np.ndarray, str]]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    audio, lengths = _batch_audio([item[0] for item in items])
    return audio, lengths, [item[1] for item in items]


def _device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _rows(
    bundle: AnalysisBundle,
    split: str,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    rows = bundle.split(split)
    if int(limit) > 0 and len(rows) > int(limit):
        rows = _speaker_balanced_sample(rows, n=int(limit), seed=seed)
    return rows


def _resample_batch(
    audio: torch.Tensor,
    lengths: torch.Tensor,
    source_rate: int,
    target_rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(source_rate) == int(target_rate):
        return audio, lengths
    try:
        import torchaudio.functional as AF
    except ImportError as exc:
        raise AnalysisError("Audio bridge training requires torchaudio.") from exc
    resampled = AF.resample(audio, int(source_rate), int(target_rate))
    target_lengths = torch.round(lengths.float() * float(target_rate) / float(source_rate)).long()
    return resampled, target_lengths.clamp(max=resampled.shape[-1])


def _target_log_mels(
    audio: torch.Tensor,
    lengths: torch.Tensor,
    source_rate: int,
    frontend: LogMelFrontend,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact per-utterance mels before padding the mel batch."""
    mels = []
    for index, length in enumerate(lengths.tolist()):
        utterance = audio[index:index + 1, :int(length)]
        utterance, _ = _resample_batch(
            utterance,
            torch.tensor([int(length)], device=audio.device),
            source_rate,
            frontend.config.sample_rate,
        )
        mels.append(frontend(utterance)[0])
    mel_lengths = torch.tensor([mel.shape[-1] for mel in mels], device=audio.device)
    width = int(mel_lengths.max())
    target = torch.zeros(
        len(mels), frontend.config.n_mels, width,
        device=audio.device, dtype=mels[0].dtype,
    )
    for index, mel in enumerate(mels):
        target[index, :, :mel.shape[-1]] = mel
    return target, mel_lengths


def _run_epoch(
    loader: DataLoader,
    *,
    encoder_model: Any,
    bridge: SpearMelBridge,
    frontend: LogMelFrontend,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    source_sample_rate: int,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    bridge.train(training)
    totals: dict[str, float] = {}
    examples = 0
    for audio, lengths, _ in loader:
        audio = audio.to(device)
        lengths = lengths.to(device)
        with torch.no_grad():
            h, h_lengths = encoder_model.encoder(audio, lengths)
            h_mask = torch.arange(h.shape[1], device=device)[None] < h_lengths[:, None]
            h = h.masked_fill(~h_mask[..., None], 0)
            target, target_lengths = _target_log_mels(
                audio, lengths, source_sample_rate, frontend,
            )
        with torch.set_grad_enabled(training):
            prediction = bridge(h, target.shape[-1])
            loss, pieces = masked_bridge_loss(prediction, target, target_lengths)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), float(grad_clip))
                optimizer.step()
        batch_size = int(audio.shape[0])
        examples += batch_size
        for key, value in pieces.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
    return {key: value / max(examples, 1) for key, value in totals.items()} | {
        "examples": int(examples),
    }


def train_bridge(
    checkpoint: Path,
    data_root: Path,
    output_dir: Path,
    *,
    device: str | None = None,
    seed: int = 42,
    epochs: int = 8,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    max_train_utterances: int = 0,
    max_validation_utterances: int = 0,
    hidden_dim: int = 384,
    residual_layers: int = 6,
    mel_config: MelConfig | None = None,
) -> Path:
    set_seed(seed)
    device = _device(device)
    bundle = AnalysisBundle(data_root)
    resolved = load_checkpoint(checkpoint)
    model = calibrate(resolved, bundle, device)
    model.eval()
    mel_config = mel_config or MelConfig()
    bridge_config = BridgeConfig(
        input_dim=int(resolved.config["D"]),
        hidden_dim=int(hidden_dim),
        residual_layers=int(residual_layers),
        spear_sample_rate=int(resolved.config.get("sample_rate", bundle.spec.sample_rate)),
        spear_hop_samples=640,
        spear_model_id=str(resolved.config.get("spear_model_id", "")),
        spear_revision=str(resolved.config.get("spear_revision", "")),
        spear_layernorm=bool(resolved.config.get("spear_layernorm", False)),
        mel=mel_config,
    )
    bridge = SpearMelBridge(bridge_config).to(device)
    frontend = LogMelFrontend(mel_config).to(device)
    train_rows = _rows(bundle, "train", max_train_utterances, seed)
    validation_rows = _rows(bundle, "validation", max_validation_utterances, seed + 1)
    if train_rows.empty or validation_rows.empty:
        raise AnalysisError("Audio bridge training requires non-empty train and validation splits.")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        BundleAudioDataset(bundle, train_rows), batch_size=int(batch_size), shuffle=True,
        generator=generator, num_workers=0, collate_fn=_collate,
    )
    validation_loader = DataLoader(
        BundleAudioDataset(bundle, validation_rows), batch_size=int(batch_size), shuffle=False,
        num_workers=0, collate_fn=_collate,
    )
    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay),
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best = float("inf")
    best_path = output_dir / "best.pt"
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_epoch(
            train_loader, encoder_model=model, bridge=bridge, frontend=frontend,
            optimizer=optimizer, device=device, source_sample_rate=bundle.spec.sample_rate,
            grad_clip=grad_clip,
        )
        validation_metrics = _run_epoch(
            validation_loader, encoder_model=model, bridge=bridge, frontend=frontend,
            optimizer=None, device=device, source_sample_rate=bundle.spec.sample_rate,
            grad_clip=grad_clip,
        )
        row = {
            "epoch": int(epoch),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"[audio-bridge] epoch {epoch}/{epochs} "
            f"train={train_metrics.get('loss', float('nan')):.5f} "
            f"validation={validation_metrics.get('loss', float('nan')):.5f}",
            flush=True,
        )
        training_record = {
            "epoch": int(epoch),
            "checkpoint_used_only_for_spear_domain": str(checkpoint.resolve()),
            "data": str(data_root.resolve()),
            "training_split": "train",
            "validation_split": "validation",
            "test_examples_used": 0,
            "train_utterances": int(len(train_rows)),
            "validation_utterances": int(len(validation_rows)),
            "seed": int(seed),
            "history": history,
        }
        save_bridge(output_dir / "last.pt", bridge, training=training_record)
        if float(validation_metrics.get("loss", float("inf"))) < best:
            best = float(validation_metrics["loss"])
            save_bridge(best_path, bridge, training=training_record | {"best_validation_loss": best})
    write_json(output_dir / "training_manifest.json", {
        "format": "spear_logmel_bridge_training_v1",
        "best_checkpoint": str(best_path),
        "best_validation_loss": best,
        "checkpoint_used_only_for_spear_domain": str(checkpoint.resolve()),
        "data": str(data_root.resolve()),
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "test_examples_used": 0,
        "train_utterances": int(len(train_rows)),
        "validation_utterances": int(len(validation_rows)),
        "history": history,
    })
    return best_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a frozen-SPEAR to log-mel bridge.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-utterances", type=int, default=0)
    parser.add_argument("--max-validation-utterances", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--residual-layers", type=int, default=6)
    parser.add_argument("--mel-sample-rate", type=int, default=24_000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=100)
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument("--f-max", type=float, default=12_000.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    mel = MelConfig(
        sample_rate=args.mel_sample_rate,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f_min=args.f_min,
        f_max=args.f_max,
    )
    try:
        path = train_bridge(
            args.checkpoint, args.data, args.output_dir,
            device=args.device, seed=args.seed, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            max_train_utterances=args.max_train_utterances,
            max_validation_utterances=args.max_validation_utterances,
            hidden_dim=args.hidden_dim, residual_layers=args.residual_layers,
            mel_config=mel,
        )
    except AnalysisError as exc:
        raise SystemExit(f"[audio-bridge] ERROR: {exc}") from exc
    print(f"[audio-bridge] best checkpoint: {path}")


if __name__ == "__main__":
    main()
