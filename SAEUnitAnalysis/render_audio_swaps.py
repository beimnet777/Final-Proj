"""Render a bounded, auditable set of latent swaps as waveform audio."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .audio_bridge import LogMelFrontend, load_bridge, validate_bridge_domain
from .audio_vocoder import load_vocoder, save_waveform
from .bundle import AnalysisBundle
from .causal import _decode, _matched_non_p_units, _resample
from .checkpoint import load_checkpoint
from .extraction import FeatureCache, _read_audio, calibrate
from .utils import AnalysisError, write_json


PERSISTED_AUDIO_MODES = (
    "recipient_source",
    "donor_source",
    "oracle_mel_vocoder",
    "original_spear_bridge",
    "sae_baseline",
    "P_from_donor",
    "L_from_donor",
    "matched_P_subset_from_donor",
    "matched_nonP_from_donor",
    "P_zero",
)


def _device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_result(result_dir: Path) -> tuple[dict[str, Any], FeatureCache, pd.DataFrame]:
    manifest_path = result_dir / "run_manifest.json"
    pairs_path = result_dir / "tables" / "swap_pairs.csv"
    if not manifest_path.exists():
        raise AnalysisError(f"Analysis result lacks {manifest_path}.")
    if not pairs_path.exists():
        raise AnalysisError(
            f"Analysis result lacks {pairs_path}. Re-run the upgraded swap analysis first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_path = manifest.get("cache")
    if not cache_path or not Path(cache_path).exists():
        raise AnalysisError(
            "Audio rendering requires the persisted sparse feature cache for this result."
        )
    return manifest, FeatureCache.load(Path(cache_path)), pd.read_csv(pairs_path)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return audio
    try:
        import torchaudio.functional as AF
    except ImportError as exc:
        raise AnalysisError("Audio rendering requires torchaudio.") from exc
    tensor = torch.from_numpy(audio)[None]
    return AF.resample(tensor, int(source_rate), int(target_rate))[0].numpy()


def _generate(
    h: np.ndarray | torch.Tensor,
    *,
    bridge,
    vocoder,
    target_mel_frames: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(h, torch.Tensor):
        h = torch.from_numpy(np.asarray(h, dtype=np.float32))
    with torch.inference_mode():
        log_mel = bridge(h[None].to(device), int(target_mel_frames))
        waveform = vocoder(log_mel)
    return waveform[0], log_mel[0]


def _audio_report(output_dir: Path, manifest: pd.DataFrame) -> Path:
    cards = []
    labels = {
        "recipient_source": "Recipient source",
        "donor_source": "Donor reference",
        "oracle_mel_vocoder": "Oracle mel → vocoder",
        "original_spear_bridge": "Original SPEAR → bridge",
        "sae_baseline": "SAE reconstruction",
        "P_from_donor": "Recipient L + donor P",
        "L_from_donor": "Donor L + recipient P",
        "matched_P_subset_from_donor": "Matched P-subset swap",
        "matched_nonP_from_donor": "Matched non-P control",
        "P_zero": "Zero-P control",
    }
    for pair, group in manifest.groupby("pair", sort=True):
        players = []
        for mode in PERSISTED_AUDIO_MODES:
            row = group[group["mode"] == mode]
            if row.empty:
                continue
            relative = Path(str(row.iloc[0]["audio_path"])).relative_to(output_dir)
            players.append(
                "<div class='clip'><b>" + html.escape(labels.get(mode, mode)) + "</b>"
                f"<audio controls preload='none' src='../{html.escape(str(relative))}'></audio></div>"
            )
        interpolation = group[group["mode"].astype(str).str.startswith("P_interpolate_")].copy()
        if len(interpolation):
            interpolation = interpolation.sort_values("interpolation_alpha")
            interpolation_players = []
            for _, row in interpolation.iterrows():
                relative = Path(str(row["audio_path"])).relative_to(output_dir)
                interpolation_players.append(
                    f"<div class='clip'><b>α={float(row['interpolation_alpha']):.2f}</b>"
                    f"<audio controls preload='none' src='../{html.escape(str(relative))}'></audio></div>"
                )
            players.append(
                "<div style='grid-column:1/-1'><h3>P-route interpolation</h3>"
                "<div class='grid'>" + "".join(interpolation_players) + "</div></div>"
            )
        first = group.iloc[0]
        cards.append(
            f"<section><h2>Pair {int(pair)} · recipient {html.escape(str(first['recipient_speaker']))} "
            f"→ donor {html.escape(str(first['donor_speaker']))}</h2>"
            f"<p>{html.escape(str(first.get('transcript', '')))}</p>"
            "<div class='grid'>" + "".join(players) + "</div></section>"
        )
    grid_html = ""
    grid_manifest_path = output_dir / "audio_grid_manifest.csv"
    if grid_manifest_path.exists():
        grid = pd.read_csv(grid_manifest_path)
        if len(grid):
            columns = sorted(grid["grid_column"].unique().tolist())
            header = "<th>recipient content</th>" + "".join(
                f"<th>donor {html.escape(str(grid[grid.grid_column == column].iloc[0]['donor_speaker']))}</th>"
                for column in columns
            )
            body = []
            for row_index, row_group in grid.groupby("grid_row", sort=True):
                first = row_group.iloc[0]
                cells = [
                    "<th>" + html.escape(str(first["recipient_speaker"])) + "<br><small>"
                    + html.escape(str(first.get("transcript", ""))) + "</small></th>"
                ]
                by_column = row_group.set_index("grid_column")
                for column in columns:
                    if column not in by_column.index:
                        cells.append("<td>—</td>")
                        continue
                    relative = Path(str(by_column.loc[column, "audio_path"])).relative_to(output_dir)
                    cells.append(
                        f"<td><audio controls preload='none' src='../{html.escape(str(relative))}'></audio></td>"
                    )
                body.append("<tr>" + "".join(cells) + "</tr>")
            grid_html = (
                "<section><h2>Content × donor-speaker grid</h2>"
                "<p>Rows hold recipient linguistic content fixed; columns hold donor identity fixed.</p>"
                "<div style='overflow-x:auto'><table><thead><tr>" + header
                + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div></section>"
            )
    page = """<!doctype html><html><head><meta charset='utf-8'><title>Audio latent swaps</title>
    <style>body{font-family:system-ui;margin:2rem;background:#f5f7fb;color:#162033}
    section{background:white;padding:1rem 1.4rem;margin:1rem 0;border-radius:12px;box-shadow:0 2px 10px #1d2b4b18}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.8rem}
    .clip{display:flex;flex-direction:column;gap:.35rem}audio{width:100%}.warn{background:#fff3bf;padding:1rem;border-radius:8px}
    table{border-collapse:collapse;width:100%}th,td{padding:.5rem;border:1px solid #e5e9f2;vertical-align:top}</style>
    </head><body><h1>Waveform latent-swap evidence</h1>
    <p class='warn'>The oracle, original-SPEAR and SAE-baseline clips are reconstruction gates. Route swaps are interpretable only if these controls preserve intelligibility and identity. Griffin-Lim output is diagnostic only.</p>
    """ + grid_html + "".join(cards) + "</body></html>"
    report = output_dir / "report" / "index.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(page, encoding="utf-8")
    return report


def _attach_main_report(result_dir: Path, audio_report: Path) -> None:
    main = result_dir / "report" / "index.html"
    if not main.exists():
        return
    marker_start = "<!-- audio-conversion:start -->"
    marker_end = "<!-- audio-conversion:end -->"
    relative = Path(os.path.relpath(audio_report, main.parent))
    section = (
        marker_start
        + "<h2>Waveform latent swapping</h2><p>Selected reconstruction gates, P/L swaps, "
          "and matched controls are available in the <a href='"
        + html.escape(str(relative))
        + "'>audio conversion report</a>. Full numerical evaluation is stored separately; "
          "only a bounded demonstration set is retained as audio.</p>"
        + marker_end
    )
    text = main.read_text(encoding="utf-8")
    if marker_start in text and marker_end in text:
        before, rest = text.split(marker_start, 1)
        _, after = rest.split(marker_end, 1)
        text = before + section + after
    else:
        text = text.replace("</main>", section + "</main>")
    main.write_text(text, encoding="utf-8")


def render_audio_swaps(
    result_dir: Path,
    bridge_checkpoint: Path,
    *,
    output_dir: Path | None = None,
    device: str | None = None,
    vocoder_backend: str = "bigvgan",
    vocoder_model_id: str = "nvidia/bigvgan_v2_24khz_100band_256x",
    bigvgan_repo: Path | None = None,
    max_pairs: int = 24,
    interpolation_pairs: int = 5,
    grid_size: int = 0,
    seed: int = 42,
) -> Path:
    device = _device(device)
    result_dir = result_dir.resolve()
    run_manifest, cache, pairs = _load_result(result_dir)
    checkpoint = Path(run_manifest["checkpoint"])
    bundle = AnalysisBundle(Path(run_manifest["data"]))
    resolved = load_checkpoint(checkpoint)
    bridge, bridge_payload = load_bridge(bridge_checkpoint, device)
    validate_bridge_domain(bridge.config, resolved.config)
    vocoder = load_vocoder(
        vocoder_backend, bridge.config.mel, device=device, model_id=vocoder_model_id,
        bigvgan_repo=bigvgan_repo,
    )
    frontend = LogMelFrontend(bridge.config.mel).to(device)
    encoder_model = calibrate(resolved, bundle, device).eval()
    decoder_weight = resolved.state["sae.dec_weight"].detach().float().cpu().numpy()
    matched_p, matched_non_p, matching_summary = _matched_non_p_units(cache, decoder_weight)
    output_dir = (output_dir or result_dir / "audio_conversion").resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata = bundle.utterances.set_index(bundle.utterances["utterance_id"].astype(str))
    cache_index = {
        str(utterance_id): int(index)
        for index, utterance_id in enumerate(cache.utterance_ids)
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for _, pair in pairs.head(max(0, int(max_pairs))).iterrows():
        pair_id = int(pair["pair"])
        recipient_id = str(pair["recipient"])
        donor_id = str(pair["donor"])
        if recipient_id not in cache_index or donor_id not in cache_index:
            raise AnalysisError("Swap pair and feature cache utterance IDs do not agree.")
        a, b = cache_index[recipient_id], cache_index[donor_id]
        recipient_row = metadata.loc[recipient_id]
        donor_row = metadata.loc[donor_id]
        recipient_audio = _read_audio(bundle.audio_path(recipient_row), bundle.spec.sample_rate)
        donor_audio = _read_audio(bundle.audio_path(donor_row), bundle.spec.sample_rate)
        original_tensor = torch.from_numpy(recipient_audio)[None].to(device)
        original_lengths = torch.tensor([len(recipient_audio)], device=device)
        with torch.inference_mode():
            original_h, original_h_lengths = encoder_model.encoder(original_tensor, original_lengths)
            original_h = original_h[0, :int(original_h_lengths[0])]
        za = cache.dense(cache.utterance_slice(a))
        zb = cache.dense(cache.utterance_slice(b))
        length = min(len(za), len(original_h))
        za = za[:length]
        original_h = original_h[:length]
        baseline_h = _decode(za, resolved)

        def replace(units: np.ndarray) -> np.ndarray:
            donor = _resample(zb[:, units], len(za))
            return baseline_h + (donor - za[:, units]) @ decoder_weight[:, units].T

        p_units = np.flatnonzero(cache.route == 1)
        l_units = np.flatnonzero(cache.route == 0)
        p_h = replace(p_units)
        l_h = replace(l_units)
        matched_p_h = replace(matched_p)
        matched_h = replace(matched_non_p)
        zero_p_h = baseline_h + (0.0 - za[:, p_units]) @ decoder_weight[:, p_units].T

        target_audio = _resample_audio(
            recipient_audio, bundle.spec.sample_rate, bridge.config.mel.sample_rate,
        )
        with torch.inference_mode():
            target_log_mel = frontend(torch.from_numpy(target_audio)[None].to(device))
            oracle_waveform = vocoder(target_log_mel)[0]
        target_frames = int(target_log_mel.shape[-1])
        generated = {
            "recipient_source": torch.from_numpy(target_audio),
            "donor_source": torch.from_numpy(_resample_audio(
                donor_audio, bundle.spec.sample_rate, bridge.config.mel.sample_rate,
            )),
            "oracle_mel_vocoder": oracle_waveform,
            "original_spear_bridge": _generate(
                original_h, bridge=bridge, vocoder=vocoder,
                target_mel_frames=target_frames, device=device,
            )[0],
            "sae_baseline": _generate(
                baseline_h, bridge=bridge, vocoder=vocoder,
                target_mel_frames=target_frames, device=device,
            )[0],
            "P_from_donor": _generate(
                p_h, bridge=bridge, vocoder=vocoder,
                target_mel_frames=target_frames, device=device,
            )[0],
            "L_from_donor": _generate(
                l_h, bridge=bridge, vocoder=vocoder,
                target_mel_frames=target_frames, device=device,
            )[0],
            "matched_P_subset_from_donor": _generate(
                matched_p_h, bridge=bridge, vocoder=vocoder,
                target_mel_frames=target_frames, device=device,
            )[0],
            "matched_nonP_from_donor": _generate(
                matched_h, bridge=bridge, vocoder=vocoder,
                target_mel_frames=target_frames, device=device,
            )[0],
            "P_zero": _generate(
                zero_p_h, bridge=bridge, vocoder=vocoder,
                target_mel_frames=target_frames, device=device,
            )[0],
        }
        if pair_id < int(interpolation_pairs):
            donor_p = _resample(zb[:, p_units], len(za))
            for alpha in (0.0, .25, .5, .75, 1.0):
                interpolated_h = baseline_h + (
                    float(alpha) * (donor_p - za[:, p_units])
                ) @ decoder_weight[:, p_units].T
                generated[f"P_interpolate_{alpha:.2f}"] = _generate(
                    interpolated_h, bridge=bridge, vocoder=vocoder,
                    target_mel_frames=target_frames, device=device,
                )[0]
        for mode, waveform in generated.items():
            path = audio_dir / f"pair{pair_id:04d}_{mode}.wav"
            save_waveform(path, waveform, vocoder.sample_rate)
            rows.append({
                "pair": pair_id,
                "mode": mode,
                "interpolation_alpha": (
                    float(mode.rsplit("_", 1)[-1])
                    if mode.startswith("P_interpolate_") else float("nan")
                ),
                "audio_path": str(path),
                "recipient": recipient_id,
                "donor": donor_id,
                "recipient_speaker": str(pair["recipient_speaker"]),
                "donor_speaker": str(pair["donor_speaker"]),
                "transcript": str(recipient_row.get("transcript", "")),
                "sample_rate": int(vocoder.sample_rate),
                "samples": int(np.asarray(waveform.detach().cpu()).size),
            })
    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "audio_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    grid_rows: list[dict[str, Any]] = []
    grid_registry_path = result_dir / "tables" / "swap_content_speaker_grid.csv"
    if int(grid_size) > 0 and grid_registry_path.exists():
        grid_registry = pd.read_csv(grid_registry_path)
        grid_registry = grid_registry[
            (grid_registry["grid_row"] < int(grid_size))
            & (grid_registry["grid_column"] < int(grid_size))
        ]
        for _, cell in grid_registry.iterrows():
            recipient_id, donor_id = str(cell["recipient"]), str(cell["donor"])
            if recipient_id not in cache_index or donor_id not in cache_index:
                continue
            a, b = cache_index[recipient_id], cache_index[donor_id]
            za = cache.dense(cache.utterance_slice(a))
            zb = cache.dense(cache.utterance_slice(b))
            baseline_h = _decode(za, resolved)
            p_units = np.flatnonzero(cache.route == 1)
            donor_p = _resample(zb[:, p_units], len(za))
            hybrid_h = baseline_h + (donor_p - za[:, p_units]) @ decoder_weight[:, p_units].T
            recipient_row = metadata.loc[recipient_id]
            recipient_audio = _read_audio(bundle.audio_path(recipient_row), bundle.spec.sample_rate)
            target_audio = _resample_audio(
                recipient_audio, bundle.spec.sample_rate, bridge.config.mel.sample_rate,
            )
            with torch.inference_mode():
                target_frames = int(frontend(torch.from_numpy(target_audio)[None].to(device)).shape[-1])
            waveform, _ = _generate(
                hybrid_h, bridge=bridge, vocoder=vocoder,
                target_mel_frames=target_frames, device=device,
            )
            path = audio_dir / (
                f"grid_r{int(cell['grid_row']):02d}_c{int(cell['grid_column']):02d}.wav"
            )
            save_waveform(path, waveform, vocoder.sample_rate)
            grid_rows.append({
                **cell.to_dict(), "audio_path": str(path),
                "transcript": str(recipient_row.get("transcript", "")),
                "sample_rate": int(vocoder.sample_rate),
            })
    grid_manifest_path = output_dir / "audio_grid_manifest.csv"
    if grid_rows:
        pd.DataFrame(grid_rows).to_csv(grid_manifest_path, index=False)
    elif grid_manifest_path.exists():
        grid_manifest_path.unlink()
    write_json(output_dir / "render_manifest.json", {
        "format": "spear_audio_swap_render_v1",
        "source_analysis": str(result_dir),
        "source_checkpoint": str(checkpoint),
        "bridge_checkpoint": str(bridge_checkpoint.resolve()),
        "bridge_training": bridge_payload.get("training", {}),
        "vocoder_backend": vocoder_backend,
        "vocoder_model_id": vocoder_model_id if vocoder_backend == "bigvgan" else None,
        "bigvgan_repo": str(bigvgan_repo.resolve()) if bigvgan_repo else None,
        "sample_rate": int(vocoder.sample_rate),
        "pairs_rendered": int(manifest["pair"].nunique()) if len(manifest) else 0,
        "interpolation_pairs": int(min(max_pairs, interpolation_pairs)),
        "audio_grid_size": int(grid_size),
        "audio_grid_cells": int(len(grid_rows)),
        "persisted_modes": list(PERSISTED_AUDIO_MODES),
        "matched_non_p_control": matching_summary,
        "random_seed": int(seed),
        "diagnostic_only": bool(vocoder_backend == "griffinlim"),
    })
    report = _audio_report(output_dir, manifest)
    _attach_main_report(result_dir, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render selected SAE latent swaps as audio.")
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--vocoder", choices=("bigvgan", "griffinlim"), default="bigvgan")
    parser.add_argument(
        "--vocoder-model-id", default="nvidia/bigvgan_v2_24khz_100band_256x",
    )
    parser.add_argument(
        "--bigvgan-repo", type=Path, default=None,
        help="Path to a clone of the official NVIDIA/BigVGAN repository.",
    )
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--interpolation-pairs", type=int, default=5)
    parser.add_argument(
        "--grid-size", type=int, default=0,
        help="Optional N for an N-by-N content/speaker audio grid (recommended: 5).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        report = render_audio_swaps(
            args.result_dir, args.bridge, output_dir=args.output_dir,
            device=args.device, vocoder_backend=args.vocoder,
            vocoder_model_id=args.vocoder_model_id,
            bigvgan_repo=args.bigvgan_repo,
            max_pairs=args.max_pairs, interpolation_pairs=args.interpolation_pairs,
            grid_size=args.grid_size,
            seed=args.seed,
        )
    except AnalysisError as exc:
        raise SystemExit(f"[audio-render] ERROR: {exc}") from exc
    print(f"[audio-render] report: {report}")


if __name__ == "__main__":
    main()
