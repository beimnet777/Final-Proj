from __future__ import annotations

import math
import time
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


CACHE_SCHEMA_VERSION = 3


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
    # Fixed membership does not necessarily imply fixed active allocation.
    # Experiments trained with --no-per_block_topk must retain their global
    # Top-K competition during analysis.
    if cfg.get("per_block_topk") is False:
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


def _route_quota_spec(resolved: ResolvedModel) -> list[tuple[np.ndarray, int]] | None:
    """Recover the learned route-local Top-K contract stored by quota freeze."""
    state = resolved.state
    enabled = state.get("sae.route_topk_enabled")
    route_idx = state.get("sae.route_topk_idx")
    quotas = state.get("sae.route_topk_quotas")
    if enabled is None or route_idx is None or quotas is None:
        return None
    if not bool(torch.as_tensor(enabled).item()):
        return None
    route_idx_np = torch.as_tensor(route_idx).detach().cpu().numpy().astype(np.int16)
    quotas_np = torch.as_tensor(quotas).detach().cpu().numpy().astype(int)
    result: list[tuple[np.ndarray, int]] = []
    for route, budget in enumerate(quotas_np.tolist()):
        members = np.flatnonzero(route_idx_np == route)
        if int(budget) > len(members):
            raise AnalysisError(
                f"Stored route-local Top-K quota {budget} exceeds route {route} "
                f"membership {len(members)}."
            )
        if len(members) and int(budget) > 0:
            result.append((members, int(budget)))
    return result or None


def _encode_sparse(h: torch.Tensor, model, resolved: ResolvedModel) -> tuple[torch.Tensor, torch.Tensor]:
    centred = h - model.sae.b_pre
    pre = F.linear(centred, model.sae.enc_weight)
    # Learned quota-freeze buffers take precedence. They are the representation
    # actually used after route freezing and must not silently degrade to global
    # Top-K during post-hoc analysis.
    spec = _route_quota_spec(resolved) or _block_spec(resolved)
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


def _quick_sample(part: pd.DataFrame, *, n: int = 24, seed: int = 42) -> pd.DataFrame:
    """Deterministic speaker-balanced smoke subset.

    LibriSpeech manifests are speaker-sorted, so ``head(24)`` produces a
    one-speaker subset and confounds speaker identity with split. Prefer eight
    speakers with three utterances each when the data supports it.
    """
    if len(part) <= n:
        return part.copy()
    rng = np.random.default_rng(seed)
    if "speaker_id" not in part.columns:
        chosen = np.sort(rng.choice(len(part), size=n, replace=False))
        return part.iloc[chosen].copy()

    groups = []
    for speaker, group in part.groupby(part["speaker_id"].astype(str), sort=True):
        if len(group) >= 3:
            groups.append((str(speaker), group))
    if not groups:
        chosen = np.sort(rng.choice(len(part), size=n, replace=False))
        return part.iloc[chosen].copy()

    order = rng.permutation(len(groups))
    rows = []
    target_speakers = min(8, len(groups), max(1, n // 3))
    for gi in order[:target_speakers]:
        group = groups[int(gi)]
        frame = group[1]
        take = min(3, len(frame))
        local = np.sort(rng.choice(len(frame), size=take, replace=False))
        rows.append(frame.iloc[local])
    sampled = pd.concat(rows, axis=0) if rows else part.iloc[0:0]
    if len(sampled) < n:
        remaining = part.drop(index=sampled.index, errors="ignore")
        take = min(n - len(sampled), len(remaining))
        if take:
            local = np.sort(rng.choice(len(remaining), size=take, replace=False))
            sampled = pd.concat([sampled, remaining.iloc[local]], axis=0)
    return sampled.iloc[:n].sort_values(["speaker_id", "utterance_id"]).copy()


def _speaker_balanced_sample(part: pd.DataFrame, *, n: int, seed: int) -> pd.DataFrame:
    """Deterministically cap a split without inheriting manifest speaker order."""
    n = max(0, int(n))
    if n >= len(part):
        return part.copy()
    if n == 0:
        return part.iloc[0:0].copy()
    rng = np.random.default_rng(seed)
    if "speaker_id" not in part.columns:
        chosen = np.sort(rng.choice(len(part), size=n, replace=False))
        return part.iloc[chosen].copy()

    queues: list[list[Any]] = []
    for _, group in part.groupby(part["speaker_id"].astype(str), sort=True):
        indices = group.index.to_numpy(copy=True)
        indices = indices[rng.permutation(len(indices))]
        queues.append(indices.tolist())
    order = rng.permutation(len(queues)).tolist()
    chosen_indices: list[Any] = []
    cursor = 0
    while len(chosen_indices) < n and order:
        queue_id = int(order[cursor % len(order)])
        if queues[queue_id]:
            chosen_indices.append(queues[queue_id].pop())
        cursor += 1
        if cursor % len(order) == 0:
            order = [i for i in order if queues[int(i)]]
            if order:
                order = rng.permutation(order).tolist()
            cursor = 0
    sampled = part.loc[chosen_indices]
    return sampled.sort_values(["speaker_id", "utterance_id"]).copy()


def parse_split_limits(spec: str | dict[str, int] | None) -> dict[str, int]:
    if spec is None or spec == "":
        return {}
    if isinstance(spec, dict):
        raw = spec
    else:
        raw = {}
        for token in str(spec).split(","):
            if not token.strip() or "=" not in token:
                raise AnalysisError(
                    "--split-limits must look like train=3000,validation=1000,test=1000."
                )
            key, value = token.split("=", 1)
            try:
                raw[key.strip()] = int(value.strip())
            except ValueError as exc:
                raise AnalysisError(f"Invalid split limit: {token!r}.") from exc
    aliases = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}
    parsed: dict[str, int] = {}
    for key, value in raw.items():
        logical = aliases.get(str(key).strip().lower())
        if logical is None:
            raise AnalysisError(f"Unknown split in --split-limits: {key!r}.")
        if int(value) < 0:
            raise AnalysisError("Split limits must be non-negative.")
        parsed[logical] = int(value)
    return parsed


def _candidate_specs(resolved: ResolvedModel) -> list[dict[str, Any]]:
    cfg, K = resolved.config, int(resolved.config["K"])
    topks = [int(cfg["topk"])] if "topk" in cfg else [k for k in (64, 128, 256) if k <= K]
    lns = [bool(cfg["spear_layernorm"])] if "spear_layernorm" in cfg else [False, True]
    candidates = [{"topk": k, "spear_layernorm": ln} for k in topks for ln in lns]
    if (
        resolved.config.get("fixed_blocks")
        and cfg.get("per_block_topk") is not False
        and not (cfg.get("block_topk") or cfg.get("topk_blocks"))
    ):
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
    needs_blocks = (
        resolved.config.get("fixed_blocks")
        and resolved.config.get("per_block_topk") is not False
        and not (
        resolved.config.get("block_topk") or resolved.config.get("topk_blocks")
        )
    )
    model = build_model(resolved, device)
    if not missing and not needs_blocks:
        return model
    if needs_blocks:
        raise AnalysisError(
            "Fixed-block checkpoint is missing per-block extraction budgets. "
            "Refusing to infer block_topk by reconstruction calibration because "
            "using the wrong L/P/U active budget changes unit activity, deadness, "
            "examples, and selectivity. Add a checkpoint sidecar such as "
            "<checkpoint>.analysis.yaml with block_topk: [240, 16, 0], or ensure "
            "the checkpoint config serializes topk_L/topk_P/topk_U."
        )

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
    seed: int = 42,
    split_limits: dict[str, int] | None = None,
    compute_acoustics: bool = True,
) -> FeatureCache:
    split_limits = parse_split_limits(split_limits)
    key = fingerprint(
        [resolved.checkpoint, bundle.spec.manifest_path] + ([bundle.spec.alignments_path] if bundle.spec.alignments_path else []),
        {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "config": resolved.config,
            "profile": profile,
            "seed": int(seed) if profile == "quick" else None,
            "split_limits": split_limits,
            "compute_acoustics": bool(compute_acoustics),
        },
    )
    path = cache_dir / f"features-{key}.npz"
    if path.exists():
        return FeatureCache.load(path)

    selected = []
    selected_counts: dict[str, int] = {}
    for logical in ("train", "validation", "test"):
        part = bundle.split(logical)
        if profile == "quick":
            split_seed = int(seed) + {"train": 0, "validation": 1, "test": 2}[logical]
            part = _quick_sample(part, n=24, seed=split_seed)
        elif logical in split_limits:
            split_seed = int(seed) + {"train": 0, "validation": 1, "test": 2}[logical]
            part = _speaker_balanced_sample(part, n=split_limits[logical], seed=split_seed)
        selected_counts[logical] = int(len(part))
        selected.append(part)
    rows = pd.concat(selected, ignore_index=True)
    if rows.empty:
        raise AnalysisError("No train/validation/test utterances were selected from the bundle.")
    print(
        f"[SAEUnitAnalysis] extracting {len(rows)} utterances on {device} "
        f"(splits={selected_counts}, acoustics={bool(compute_acoustics)})",
        flush=True,
    )

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
    rng = np.random.default_rng(seed)
    extraction_started = time.monotonic()

    for start in range(0, len(rows), batch_size):
        batch_rows = rows.iloc[start:start+batch_size]
        waves = [_read_audio(bundle.audio_path(r), bundle.spec.sample_rate) for _, r in batch_rows.iterrows()]
        audio, audio_lengths = _batch_audio(waves)
        audio, audio_lengths = audio.to(device), audio_lengths.to(device)
        h, out_lengths = model.encoder(audio, audio_lengths)
        idx, values = _encode_sparse(h, model, resolved)
        for local, (_, row) in enumerate(batch_rows.iterrows()):
            n = int(out_lengths[local])
            index_dtype = (
                np.uint16
                if int(resolved.config["K"]) <= np.iinfo(np.uint16).max
                else np.uint32
            )
            ii = idx[local, :n].detach().cpu().numpy().astype(index_dtype)
            vv = values[local, :n].detach().float().cpu().numpy().astype(np.float16)
            hh = h[local, :n].detach().float().cpu().numpy()
            uid = str(row["utterance_id"])
            duration = len(waves[local]) / bundle.spec.sample_rate
            phone = _phone_frames(bundle, uid, n, duration)
            if compute_acoustics:
                f0, energy, voicing = _acoustics(waves[local], bundle.spec.sample_rate, n)
            else:
                f0 = np.zeros(n, dtype=np.float32)
                energy = np.zeros(n, dtype=np.float32)
                voicing = np.zeros(n, dtype=np.float32)

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
        completed = min(start + len(batch_rows), len(rows))
        if completed == len(rows) or completed % 100 < len(batch_rows):
            elapsed = max(time.monotonic() - extraction_started, 1e-6)
            rate = completed / elapsed
            eta_minutes = (len(rows) - completed) / max(rate, 1e-9) / 60.0
            print(
                f"[SAEUnitAnalysis] extracted {completed}/{len(rows)} "
                f"({rate:.2f} utt/s, ETA {eta_minutes:.1f} min)",
                flush=True,
            )

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
    print(
        f"[SAEUnitAnalysis] writing feature cache: {cache.n_frames} frames, "
        f"indices={cache.indices.dtype}",
        flush=True,
    )
    cache.save()
    write_json(path.with_suffix(".json"), {
        "checkpoint": str(resolved.checkpoint), "utterances": len(ids),
        "frames": cache.n_frames, "K": cache.K, "D": cache.D,
        "profile": profile, "config": resolved.config,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "seed": int(seed),
        "split_limits": split_limits,
        "selected_splits": selected_counts,
        "index_dtype": str(cache.indices.dtype),
        "compute_acoustics": bool(compute_acoustics),
    })
    return cache
