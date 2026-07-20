"""Render a self-contained ten-pair direct-HiFi-GAN listening demonstration."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .audio_vocoder import save_waveform
from .bundle import AnalysisBundle
from .causal import _decode, _resample
from .checkpoint import load_checkpoint, route_information
from .direct_hifigan import load_direct_hifigan
from .extraction import _batch_audio, _encode_sparse, _read_audio, calibrate
from .train_audio_bridge import _device
from .utils import AnalysisError, write_json


DEMO_MODES = (
    "recipient_source",
    "donor_source",
    "original_spear_direct",
    "sae_baseline",
    "P_from_donor",
    "L_from_donor",
)


def _validate_domain(
    payload: dict[str, Any],
    generator,
    resolved_config: dict[str, Any],
    bundle: AnalysisBundle,
) -> None:
    expected = dict(payload.get("training", {}).get("spear_domain", {}))
    if not expected:
        expected = {
            "input_dim": int(generator.config.input_dim),
            "sample_rate": int(generator.config.sample_rate),
            "spear_hop_samples": int(generator.config.spear_hop_samples),
        }
    actual = {
        "input_dim": int(resolved_config.get("D", -1)),
        "sample_rate": int(resolved_config.get("sample_rate", bundle.spec.sample_rate)),
        "spear_hop_samples": 320,
        "spear_model_id": str(resolved_config.get("spear_model_id", "")),
        "spear_revision": str(resolved_config.get("spear_revision", "")),
        "spear_layernorm": bool(resolved_config.get("spear_layernorm", False)),
    }
    mismatches = {
        key: {"vocoder": value, "sae_checkpoint": actual.get(key)}
        for key, value in expected.items()
        if key in actual and actual[key] != value
    }
    if mismatches:
        raise AnalysisError(f"Direct HiFi-GAN/checkpoint domain mismatch: {mismatches}")


@torch.inference_mode()
def _encode_utterances(
    utterance_ids: list[str],
    *,
    bundle: AnalysisBundle,
    model,
    resolved,
    device: str,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    metadata = bundle.utterances.copy()
    metadata.index = metadata["utterance_id"].astype(str)
    unknown = sorted(set(utterance_ids) - set(metadata.index))
    if unknown:
        raise AnalysisError(f"Demo pair registry contains {len(unknown)} unknown utterances.")
    encoded: dict[str, dict[str, Any]] = {}
    for start in range(0, len(utterance_ids), int(batch_size)):
        ids = utterance_ids[start:start + int(batch_size)]
        rows = [metadata.loc[utterance_id] for utterance_id in ids]
        waves = [
            _read_audio(bundle.audio_path(row), bundle.spec.sample_rate)
            for row in rows
        ]
        audio, lengths = _batch_audio(waves)
        audio, lengths = audio.to(device), lengths.to(device)
        features, feature_lengths = model.encoder(audio, lengths)
        indices, values = _encode_sparse(features, model, resolved)
        for local, (utterance_id, row, waveform) in enumerate(zip(ids, rows, waves)):
            frames = int(feature_lengths[local].item())
            dense = torch.zeros(
                frames, int(resolved.config["K"]),
                device=device, dtype=values.dtype,
            )
            dense.scatter_(1, indices[local, :frames].long(), values[local, :frames])
            encoded[utterance_id] = {
                "h": features[local, :frames].detach().float().cpu().numpy(),
                "z": dense.detach().float().cpu().numpy(),
                "audio": waveform,
                "speaker_id": str(row.get("speaker_id", "")),
                "transcript": str(row.get("transcript", "")),
                "frames": frames,
            }
        print(
            f"[direct-demo] encoded {min(start + len(ids), len(utterance_ids))}/"
            f"{len(utterance_ids)} utterances", flush=True,
        )
    return encoded


@torch.inference_mode()
def _synthesize(generator, features: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))[None].to(device)
    return generator(tensor)[0, 0].float().clamp(-1, 1).cpu()


def _report(output_dir: Path, manifest: pd.DataFrame) -> Path:
    labels = {
        "recipient_source": "Recipient reference",
        "donor_source": "Donor reference",
        "original_spear_direct": "Original SPEAR → direct HiFi-GAN",
        "sae_baseline": "SAE reconstruction → direct HiFi-GAN",
        "P_from_donor": "Recipient L + donor P",
        "L_from_donor": "Donor L + recipient P",
    }
    sections = []
    for pair, group in manifest.groupby("pair", sort=True):
        first = group.iloc[0]
        players = []
        for mode in DEMO_MODES:
            selected = group[group["mode"] == mode]
            if selected.empty:
                continue
            path = Path(str(selected.iloc[0]["audio_path"]))
            relative = path.relative_to(output_dir)
            players.append(
                "<div class='clip'><b>" + html.escape(labels[mode]) + "</b>"
                f"<audio controls preload='none' src='../{html.escape(str(relative))}'></audio></div>"
            )
        sections.append(
            f"<section><h2>Pair {int(pair)} · speaker "
            f"{html.escape(str(first['recipient_speaker']))} → "
            f"{html.escape(str(first['donor_speaker']))}</h2>"
            f"<p>{html.escape(str(first['transcript']))}</p>"
            "<div class='grid'>" + "".join(players) + "</div></section>"
        )
    page = """<!doctype html><html><head><meta charset='utf-8'>
    <title>Direct SPEAR HiFi-GAN final demonstration</title>
    <style>body{font-family:system-ui;margin:2rem;background:#f5f7fb;color:#162033}
    section{background:white;padding:1rem 1.4rem;margin:1rem 0;border-radius:12px;box-shadow:0 2px 10px #1d2b4b18}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:.8rem}
    .clip{display:flex;flex-direction:column;gap:.35rem}audio{width:100%}
    .note{background:#e8f2ff;padding:1rem;border-radius:8px}</style></head><body>
    <h1>Direct SPEAR-conditioned HiFi-GAN</h1>
    <p class='note'>Ten held-out registered pairs. Original-SPEAR and SAE-baseline clips are reconstruction gates. P swapping should move speaker identity while preserving recipient content; L swapping is the complementary intervention.</p>
    """ + "".join(sections) + "</body></html>"
    path = output_dir / "report" / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


def render_direct_demo(
    checkpoint: Path,
    data_root: Path,
    direct_hifigan_checkpoint: Path,
    pair_manifest: Path,
    output_dir: Path,
    *,
    device: str | None = None,
    pairs: int = 10,
    batch_size: int = 4,
    length_tolerance: float = 0.10,
) -> Path:
    device = _device(device)
    bundle = AnalysisBundle(data_root)
    resolved = load_checkpoint(checkpoint)
    if not resolved.capabilities.get("swap", False):
        raise AnalysisError("The selected SAE checkpoint does not support L/P swapping.")
    model = calibrate(resolved, bundle, device).eval()
    generator, direct_payload = load_direct_hifigan(direct_hifigan_checkpoint, device)
    _validate_domain(direct_payload, generator, resolved.config, bundle)
    registry = pd.read_csv(pair_manifest).head(max(0, int(pairs))).copy()
    required = {"pair", "recipient", "donor", "recipient_speaker", "donor_speaker"}
    missing = required - set(registry.columns)
    if missing:
        raise AnalysisError(f"Demo pair registry is missing columns: {sorted(missing)}")
    if len(registry) != int(pairs):
        raise AnalysisError(f"Requested {pairs} demo pairs but registry provides {len(registry)}.")
    if registry["recipient_speaker"].astype(str).nunique() != len(registry):
        raise AnalysisError("Demo pairs must use distinct recipient speakers.")
    if registry["donor_speaker"].astype(str).nunique() != len(registry):
        raise AnalysisError("Demo pairs must use distinct donor speakers.")
    ids = list(dict.fromkeys(
        registry["recipient"].astype(str).tolist()
        + registry["donor"].astype(str).tolist()
    ))
    encoded = _encode_utterances(
        ids, bundle=bundle, model=model, resolved=resolved,
        device=device, batch_size=batch_size,
    )
    route, _ = route_information(resolved)
    linguistic = np.flatnonzero(route == 0)
    paralinguistic = np.flatnonzero(route == 1)
    if not len(linguistic) or not len(paralinguistic):
        raise AnalysisError("SAE checkpoint has no non-empty L/P route assignment.")
    decoder = resolved.state["sae.dec_weight"].detach().float().cpu().numpy()
    output_dir = output_dir.resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, pair in registry.iterrows():
        pair_id = int(pair["pair"])
        registered_pair = int(pair.get("registered_pair", pair_id))
        recipient_id, donor_id = str(pair["recipient"]), str(pair["donor"])
        recipient, donor = encoded[recipient_id], encoded[donor_id]
        if recipient["speaker_id"] != str(pair["recipient_speaker"]):
            raise AnalysisError(f"Recipient speaker mismatch for pair {pair_id}.")
        if donor["speaker_id"] != str(pair["donor_speaker"]):
            raise AnalysisError(f"Donor speaker mismatch for pair {pair_id}.")
        length_ratio = donor["frames"] / max(recipient["frames"], 1)
        if abs(length_ratio - 1.0) > float(length_tolerance) + 1e-8:
            raise AnalysisError(
                f"Pair {pair_id} violates {length_tolerance:.0%} length tolerance: "
                f"ratio={length_ratio:.4f}."
            )
        za, zb = recipient["z"], donor["z"]
        baseline = _decode(za, resolved)

        def replace(units: np.ndarray) -> np.ndarray:
            donor_values = _resample(zb[:, units], len(za))
            return baseline + (donor_values - za[:, units]) @ decoder[:, units].T

        original_reconstruction = _synthesize(generator, recipient["h"], device)
        reconstruction_duration_ratio = len(original_reconstruction) / max(
            len(recipient["audio"]), 1,
        )
        if abs(reconstruction_duration_ratio - 1.0) > 0.05:
            raise AnalysisError(
                f"Pair {pair_id} direct reconstruction duration ratio is "
                f"{reconstruction_duration_ratio:.4f}; expected within 5% of the source."
            )
        signals: dict[str, torch.Tensor | np.ndarray] = {
            "recipient_source": recipient["audio"],
            "donor_source": donor["audio"],
            "original_spear_direct": original_reconstruction,
            "sae_baseline": _synthesize(generator, baseline, device),
            "P_from_donor": _synthesize(generator, replace(paralinguistic), device),
            "L_from_donor": _synthesize(generator, replace(linguistic), device),
        }
        for mode, waveform in signals.items():
            path = audio_dir / f"pair{pair_id:02d}_{mode}.wav"
            save_waveform(path, waveform, generator.config.sample_rate)
            rows.append({
                "pair": pair_id, "mode": mode, "audio_path": str(path),
                "registered_pair": registered_pair,
                "recipient": recipient_id, "donor": donor_id,
                "recipient_speaker": recipient["speaker_id"],
                "donor_speaker": donor["speaker_id"],
                "recipient_frames": recipient["frames"],
                "donor_frames": donor["frames"],
                "length_ratio": float(length_ratio),
                "reconstruction_duration_ratio": float(reconstruction_duration_ratio),
                "transcript": recipient["transcript"],
                "sample_rate": int(generator.config.sample_rate),
            })
        print(f"[direct-demo] rendered pair {pair_id + 1}/{len(registry)}", flush=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "audio_manifest.csv", index=False)
    write_json(output_dir / "render_manifest.json", {
        "format": "direct_spear_hifigan_demo_v1",
        "sae_checkpoint": str(checkpoint.resolve()),
        "direct_hifigan_checkpoint": str(direct_hifigan_checkpoint.resolve()),
        "pair_registry": str(pair_manifest.resolve()),
        "data": str(bundle.root), "pairs": int(len(registry)),
        "modes": list(DEMO_MODES), "sample_rate": int(generator.config.sample_rate),
        "test_utterances_used": int(len(ids)), "length_tolerance": float(length_tolerance),
    })
    report = _report(output_dir, manifest)
    print(f"[direct-demo] report: {report}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render final direct-HiFi-GAN reconstruction/swap audio.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--direct-hifigan", required=True, type=Path)
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--length-tolerance", type=float, default=0.10)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        render_direct_demo(
            args.checkpoint, args.data, args.direct_hifigan,
            args.pair_manifest, args.output_dir,
            device=args.device, pairs=args.pairs,
            batch_size=args.batch_size, length_tolerance=args.length_tolerance,
        )
    except AnalysisError as exc:
        raise SystemExit(f"[direct-demo] ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
