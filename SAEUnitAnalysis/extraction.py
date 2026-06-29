from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .bundle import AnalysisBundle
from .checkpoint import build_model, route_information, unresolved_critical
from .types import ResolvedModel
from .utils import AnalysisError, fingerprint, write_json


@dataclass
class FeatureCache:
    path: Path
    utterance_ids: np.ndarray
    offsets: np.ndarray
    lengths: np.ndarray
    indices: np.ndarray
    values: np.ndarray
    phones: np.ndarray
    f0: np.ndarray
    energy: np.ndarray
    voicing: np.ndarray
    pooled_z: np.ndarray
    h_stats: np.ndarray
    h_sample: np.ndarray
    h_sample_frames: np.ndarray
    route: np.ndarray
    route_probability: np.ndarray
    K: int
    D: int

    @property
    def n_frames(self) -> int:
        return int(self.indices.shape[0])

    def utterance_slice(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i] + self.lengths[i]))

    def dense(self, frame_rows: np.ndarray | slice | None = None) -> np.ndarray:
        idx = self.indices if frame_rows is None else self.indices[frame_rows]
        val = self.values if frame_rows is None else self.values[frame_rows]
        out = np.zeros((len(idx), self.K), dtype=np.float32)
        np.put_along_axis(out, idx.astype(np.int64), val.astype(np.float32), axis=1)
        return out

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            utterance_ids=self.utterance_ids, offsets=self.offsets, lengths=self.lengths,
            indices=self.indices, values=self.values, phones=self.phones,
            f0=self.f0, energy=self.energy, voicing=self.voicing,
            pooled_z=self.pooled_z, h_stats=self.h_stats,
            h_sample=self.h_sample, h_sample_frames=self.h_sample_frames,
            route=self.route, route_probability=self.route_probability,
            K=np.asarray(self.K), D=np.asarray(self.D),
        )

    @classmethod
    def load(cls, path: Path) -> "FeatureCache":
        try:
            z = np.load(path, allow_pickle=False)
            return cls(
                path=path,
                utterance_ids=z["utterance_ids"], offsets=z["offsets"], lengths=z["lengths"],
                indices=z["indices"], values=z["values"], phones=z["phones"],
                f0=z["f0"], energy=z["energy"], voicing=z["voicing"],
                pooled_z=z["pooled_z"], h_stats=z["h_stats"],
                h_sample=z["h_sample"], h_sample_frames=z["h_sample_frames"],
                route=z["route"], route_probability=z["route_probability"],
                K=int(z["K"]), D=int(z["D"]),
            )
        except Exception as exc:
            raise AnalysisError(f"Feature cache is corrupt or incompatible: {path}: {exc}") from exc


def _read_audio(path: Path, target_sr: int) -> np.ndarray:
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except ImportError:
        if path.suffix.lower() != ".wav":
            raise AnalysisError("soundfile is required for non-WAV audio.")
        with wave.open(str(path), "rb") as w:
            sr, channels, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            if width != 2:
                raise AnalysisError("The standard-library fallback supports 16-bit PCM WAV only.")
            audio = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(1)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError as exc:
            raise AnalysisError(f"Resampling {path} requires librosa.") from exc
    return np.ascontiguousarray(audio, dtype=np.float32)


def _batch_audio(items: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(x) for x in items], dtype=torch.long)
    width = int(lengths.max())
    batch = torch.zeros(len(items), width)
    for i, x in enumerate(items):
        batch[i, :len(x)] = torch.from_numpy(x)
    return batch, lengths


def _block_spec(resolved: ResolvedModel) -> list[tuple[np.ndarray, int]] | None:
    state, cfg = resolved.state, resolved.config
    if "block_idx" not in state:
        return None
    block_idx = state["block_idx"].detach().cpu().numpy()
    budgets = cfg.get("block_topk") or cfg.get("topk_blocks")
    if budgets is None:
        return None
    if isinstance(budgets, dict):
        budgets = [budgets.get(name, 0) for name in ("L", "P", "U")]
    result = []
    for route, budget in enumerate(budgets):
        members = np.flatnonzero(block_idx == route)
        if len(members) and int(budget) > 0:
            result.append((members, min(int(budget), len(members))))
    return result


def _encode_sparse(h: torch.Tensor, model, resolved: ResolvedModel) -> tuple[torch.Tensor, torch.Tensor]:
    centred = h - model.sae.b_pre
    pre = F.linear(centred, model.sae.enc_weight)
    spec = _block_spec(resolved)
    if spec:
        all_idx, all_val = [], []
        for members, k in spec:
            member_t = torch.as_tensor(members, device=pre.device)
            vals, local = pre.index_select(-1, member_t).topk(k, dim=-1)
            all_val.append(vals)
            all_idx.append(member_t[local])
        return torch.cat(all_idx, -1), torch.cat(all_val, -1)
    k = int(resolved.config["topk"])
    values, indices = pre.topk(k, dim=-1)
    return indices, values


def _candidate_specs(resolved: ResolvedModel) -> list[dict[str, Any]]:
    cfg, K = resolved.config, int(resolved.config["K"])
    topks = [int(cfg["topk"])] if "topk" in cfg else [k for k in (64, 128, 256) if k <= K]
    lns = [bool(cfg["spear_layernorm"])] if "spear_layernorm" in cfg else [False, True]
    candidates = [{"topk": k, "spear_layernorm": ln} for k in topks for ln in lns]
    if resolved.config.get("fixed_blocks") and not (cfg.get("block_topk") or cfg.get("topk_blocks")):
        presets = {
            5120: [[160, 64, 32], [160, 96, 0]],
            16384: [[40, 16, 8], [80, 32, 16]],
        }.get(K, [])
        candidates = [
            {"topk": sum(p), "block_topk": p, "spear_layernorm": ln}
            for p in presets for ln in lns
        ]
    return candidates


@torch.no_grad()
def calibrate(resolved: ResolvedModel, bundle: AnalysisBundle, device: str) -> Any:
    missing = unresolved_critical(resolved)
    needs_blocks = resolved.config.get("fixed_blocks") and not (
        resolved.config.get("block_topk") or resolved.config.get("topk_blocks")
    )
    model = build_model(resolved, device)
    if not missing and not needs_blocks:
        return model

    candidates = _candidate_specs(resolved)
    if not candidates:
        raise AnalysisError(
            "Checkpoint omits extraction-critical settings and has no known calibration preset. "
            "Add <checkpoint>.analysis.yaml with topk, spear_layernorm, and optional block_topk."
        )
    rows = bundle.split("validation")
    if rows.empty:
        rows = bundle.split("test")
    rows = rows.head(2)
    if rows.empty:
        raise AnalysisError("Calibration requires at least one validation or test utterance.")
    audios = [_read_audio(bundle.audio_path(r), bundle.spec.sample_rate) for _, r in rows.iterrows()]
    audio, lengths = _batch_audio(audios)
    audio, lengths = audio.to(device), lengths.to(device)
    scores = []
    for candidate in candidates:
        resolved.config.update(candidate)
        model.encoder._layernorm = bool(candidate["spear_layernorm"])
        h, out_lengths = model.encoder(audio, lengths)
        idx, values = _encode_sparse(h, model, resolved)
        dense = torch.zeros(*h.shape[:-1], int(resolved.config["K"]), device=device, dtype=h.dtype)
        dense.scatter_(-1, idx, values)
        h_hat = model.sae.decode(dense)
        T = h.shape[1]
        mask = torch.arange(T, device=device)[None] < out_lengths[:, None]
        mse = ((h - h_hat).float().pow(2).mean(-1) * mask).sum() / mask.sum().clamp(min=1)
        scores.append((float(mse), dict(candidate)))
    scores.sort(key=lambda x: x[0])
    if len(scores) > 1 and scores[0][0] > 0 and scores[1][0] / scores[0][0] < 1.02:
        detail = ", ".join(f"{c}:{s:.5g}" for s, c in scores[:4])
        raise AnalysisError(
            "Checkpoint configuration is ambiguous after reconstruction calibration "
            f"({detail}). Add an analysis sidecar instead of accepting silent defaults."
        )
    resolved.config.update(scores[0][1])
    resolved.config["calibration_mse"] = scores[0][0]
    resolved.warnings.append(f"Inferred extraction settings by calibration: {scores[0][1]}")
    model.encoder._layernorm = bool(resolved.config["spear_layernorm"])
    return model


def _phone_frames(bundle: AnalysisBundle, utterance_id: str, n_frames: int, duration: float) -> np.ndarray:
    out = np.full(n_frames, "<unaligned>", dtype="U32")
    if bundle.alignments is None or duration <= 0:
        return out
    rows = bundle.alignments[bundle.alignments["utterance_id"] == utterance_id]
    centers = (np.arange(n_frames) + 0.5) * duration / n_frames
    for _, row in rows.iterrows():
        mask = (centers >= float(row["start_sec"])) & (centers < float(row["end_sec"]))
        out[mask] = str(row["phone"])
    return out


def _acoustics(audio: np.ndarray, sr: int, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_frames == 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z
    centers = ((np.arange(n_frames) + 0.5) * len(audio) / n_frames).astype(int)
    half = max(64, int(0.0125 * sr))
    energy = np.empty(n_frames, dtype=np.float32)
    for i, c in enumerate(centers):
        x = audio[max(0, c-half):min(len(audio), c+half)]
        energy[i] = float(np.log(np.mean(x*x) + 1e-8)) if len(x) else -18.0
    try:
        import librosa
        hop = max(1, len(audio) // n_frames)
        f0_raw = librosa.yin(audio, fmin=65, fmax=400, sr=sr, hop_length=hop)
        xp = np.linspace(0, 1, len(f0_raw))
        f0 = np.interp(np.linspace(0, 1, n_frames), xp, f0_raw).astype(np.float32)
        voiced = ((f0 >= 65) & (f0 <= 400)).astype(np.float32)
        f0 = np.where(voiced > 0, np.log(np.maximum(f0, 1)), 0).astype(np.float32)
    except Exception:
        f0 = np.zeros(n_frames, dtype=np.float32)
        voiced = np.zeros(n_frames, dtype=np.float32)
    return f0, energy, voiced


@torch.no_grad()
def extract(
    resolved: ResolvedModel,
    bundle: AnalysisBundle,
    model,
    cache_dir: Path,
    device: str,
    profile: str = "full",
) -> FeatureCache:
    key = fingerprint(
        [resolved.checkpoint, bundle.spec.manifest_path] + ([bundle.spec.alignments_path] if bundle.spec.alignments_path else []),
        {"config": resolved.config, "profile": profile},
    )
    path = cache_dir / f"features-{key}.npz"
    if path.exists():
        return FeatureCache.load(path)

    selected = []
    for logical in ("train", "validation", "test"):
        part = bundle.split(logical)
        if profile == "quick":
            part = part.head(24)
        selected.append(part)
    rows = pd.concat(selected, ignore_index=True)
    if rows.empty:
        raise AnalysisError("No train/validation/test utterances were selected from the bundle.")

    model.to(device).eval()
    batch_size = 1 if profile == "quick" else int(bundle.spec.raw.get("batch_size", 4))
    all_idx: list[np.ndarray] = []
    all_val: list[np.ndarray] = []
    all_phone: list[np.ndarray] = []
    all_f0: list[np.ndarray] = []
    all_energy: list[np.ndarray] = []
    all_voicing: list[np.ndarray] = []
    offsets, lens, ids = [], [], []
    pooled, h_stats, h_sample, h_sample_frames = [], [], [], []
    cursor = 0
    rng = np.random.default_rng(42)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows.iloc[start:start+batch_size]
        waves = [_read_audio(bundle.audio_path(r), bundle.spec.sample_rate) for _, r in batch_rows.iterrows()]
        audio, audio_lengths = _batch_audio(waves)
        audio, audio_lengths = audio.to(device), audio_lengths.to(device)
        h, out_lengths = model.encoder(audio, audio_lengths)
        idx, values = _encode_sparse(h, model, resolved)
        for local, (_, row) in enumerate(batch_rows.iterrows()):
            n = int(out_lengths[local])
            ii = idx[local, :n].detach().cpu().numpy().astype(np.int32)
            vv = values[local, :n].detach().float().cpu().numpy().astype(np.float16)
            hh = h[local, :n].detach().float().cpu().numpy()
            uid = str(row["utterance_id"])
            duration = len(waves[local]) / bundle.spec.sample_rate
            phone = _phone_frames(bundle, uid, n, duration)
            f0, energy, voicing = _acoustics(waves[local], bundle.spec.sample_rate, n)

            offsets.append(cursor); lens.append(n); ids.append(uid); cursor += n
            all_idx.append(ii); all_val.append(vv); all_phone.append(phone)
            all_f0.append(f0); all_energy.append(energy); all_voicing.append(voicing)
            dense_sum = np.zeros(int(resolved.config["K"]), dtype=np.float32)
            np.add.at(dense_sum, ii.reshape(-1), vv.astype(np.float32).reshape(-1))
            pooled.append(dense_sum / max(n, 1))
            hm = hh.mean(0)
            hs = hh.std(0)
            h_stats.append(np.concatenate([hm, hs]))
            take = min(n, 8 if profile == "full" else 2)
            chosen = np.sort(rng.choice(n, size=take, replace=False)) if take else np.zeros(0, dtype=int)
            h_sample.append(hh[chosen].astype(np.float16))
            h_sample_frames.append(np.asarray(offsets[-1]) + chosen)

    route, probability = route_information(resolved)
    cache = FeatureCache(
        path=path,
        utterance_ids=np.asarray(ids, dtype="U128"),
        offsets=np.asarray(offsets, dtype=np.int64), lengths=np.asarray(lens, dtype=np.int32),
        indices=np.concatenate(all_idx), values=np.concatenate(all_val),
        phones=np.concatenate(all_phone), f0=np.concatenate(all_f0),
        energy=np.concatenate(all_energy), voicing=np.concatenate(all_voicing),
        pooled_z=np.asarray(pooled, dtype=np.float16), h_stats=np.asarray(h_stats, dtype=np.float16),
        h_sample=np.concatenate(h_sample).astype(np.float16),
        h_sample_frames=np.concatenate(h_sample_frames).astype(np.int64),
        route=route, route_probability=probability,
        K=int(resolved.config["K"]), D=int(resolved.config["D"]),
    )
    cache.save()
    write_json(path.with_suffix(".json"), {
        "checkpoint": str(resolved.checkpoint), "utterances": len(ids),
        "frames": cache.n_frames, "K": cache.K, "D": cache.D,
        "profile": profile, "config": resolved.config,
    })
    return cache

