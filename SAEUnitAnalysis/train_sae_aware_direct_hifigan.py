"""Continue a direct HiFi-GAN on mixed original and frozen-SAE conditions.

This is intentionally separate from :mod:`train_direct_hifigan`: the original
SPEAR-only training path remains unchanged.  A mature periodic direct-HiFi-GAN
checkpoint supplies the generator, discriminators and optimizer states.  The
fixed SAE is frozen and used only to map cached SPEAR features onto the SAE
decoder manifold.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cache_spear_audio_features import SpearAudioFeatureCache
from .checkpoint import load_checkpoint
from .direct_hifigan import (
    DirectSpearHiFiGenerator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    load_direct_hifigan,
    save_direct_hifigan,
)
from .extraction import _block_spec, _route_quota_spec
from .train_audio_bridge import _device
from .train_direct_hifigan import (
    DirectVocoderDataset,
    _autocast,
    _mel_frontend,
    _rolling_checkpoint,
    _set_requires_grad,
    _worker_seed,
)
from .audio_vocoder import save_waveform
from .utils import AnalysisError, set_seed, write_json


class FrozenSAEReconstructor(nn.Module):
    """Exact inference-time Top-K encode/decode from one frozen SAE checkpoint."""

    def __init__(self, checkpoint: str | Path, cache_manifest: dict[str, Any]) -> None:
        super().__init__()
        self.checkpoint = Path(checkpoint).resolve()
        resolved = load_checkpoint(self.checkpoint)
        self.config = dict(resolved.config)
        self._validate_domain(cache_manifest)
        self.register_buffer(
            "enc_weight", resolved.state["sae.enc_weight"].detach().float().clone(),
        )
        self.register_buffer(
            "dec_weight", resolved.state["sae.dec_weight"].detach().float().clone(),
        )
        self.register_buffer(
            "b_pre", resolved.state["sae.b_pre"].detach().float().clone(),
        )
        self.topk = int(resolved.config["topk"])
        raw_spec = _route_quota_spec(resolved) or _block_spec(resolved)
        self._spec: list[tuple[str, int]] = []
        for index, (members, budget) in enumerate(raw_spec or []):
            name = f"route_members_{index}"
            self.register_buffer(name, torch.as_tensor(members, dtype=torch.long))
            self._spec.append((name, int(budget)))
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _validate_domain(self, cache_manifest: dict[str, Any]) -> None:
        expected = {
            "D": int(cache_manifest["input_dim"]),
            "sample_rate": int(cache_manifest["sample_rate"]),
            "spear_model_id": str(cache_manifest.get("spear_model_id", "")),
            "spear_revision": str(cache_manifest.get("spear_revision", "")),
            "spear_layernorm": bool(cache_manifest.get("spear_layernorm", False)),
        }
        mismatches = {
            key: {"cache": value, "sae_checkpoint": self.config.get(key)}
            for key, value in expected.items()
            if key in self.config and self.config.get(key) != value
        }
        if mismatches:
            raise AnalysisError(f"SPEAR cache/SAE domain mismatch: {mismatches}")
        if int(self.config["D"]) != int(cache_manifest["input_dim"]):
            raise AnalysisError(
                f"SAE input dimension {self.config['D']} does not match cache "
                f"dimension {cache_manifest['input_dim']}."
            )

    @torch.no_grad()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.enc_weight.shape[-1]:
            raise AnalysisError(
                f"Expected SPEAR features [B,T,{self.enc_weight.shape[-1]}], "
                f"got {tuple(features.shape)}."
            )
        pre = F.linear(features - self.b_pre, self.enc_weight)
        sparse = torch.zeros_like(pre)
        if self._spec:
            for name, budget in self._spec:
                members = getattr(self, name)
                values, local = pre.index_select(-1, members).topk(budget, dim=-1)
                sparse.scatter_(-1, members[local], values)
        else:
            values, indices = pre.topk(self.topk, dim=-1)
            sparse.scatter_(-1, indices, values)
        return F.linear(sparse, self.dec_weight) + self.b_pre


def _mixed_conditioning(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    sae_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select an exact per-batch SAE fraction without blending representations."""
    if original.shape != reconstructed.shape:
        raise AnalysisError(
            f"Original/SAE feature shapes differ: {original.shape} != {reconstructed.shape}."
        )
    batch = int(original.shape[0])
    count = int(round(batch * float(sae_fraction)))
    count = min(batch, max(0, count))
    mask = torch.zeros(batch, device=original.device, dtype=torch.bool)
    if count:
        mask[torch.randperm(batch, device=original.device)[:count]] = True
    mixed = torch.where(mask[:, None, None], reconstructed, original)
    return mixed, mask


@torch.no_grad()
def _validate_domains(
    generator: DirectSpearHiFiGenerator,
    reconstructor: FrozenSAEReconstructor,
    loader: DataLoader,
    frontend,
    device: str,
    max_batches: int,
) -> tuple[dict[str, float], list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    generator.eval()
    totals = {"original": 0.0, "sae": 0.0}
    examples = 0
    previews: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for batch_index, (features, audio, utterance_ids) in enumerate(loader):
        if int(max_batches) > 0 and batch_index >= int(max_batches):
            break
        features = features.to(device)
        audio = audio.to(device)
        reconstructed = reconstructor(features)
        original_generated = generator(features)
        sae_generated = generator(reconstructed)
        if original_generated.shape != audio.shape or sae_generated.shape != audio.shape:
            raise AnalysisError(
                "Generator/audio length mismatch during dual-domain validation: "
                f"original={tuple(original_generated.shape)} "
                f"sae={tuple(sae_generated.shape)} audio={tuple(audio.shape)}."
            )
        target_mel = frontend(audio.squeeze(1))
        original_error = F.l1_loss(frontend(original_generated.squeeze(1)), target_mel)
        sae_error = F.l1_loss(frontend(sae_generated.squeeze(1)), target_mel)
        n = len(utterance_ids)
        totals["original"] += float(original_error) * n
        totals["sae"] += float(sae_error) * n
        examples += n
        if len(previews) < 2:
            for index, utterance_id in enumerate(utterance_ids):
                previews.append((
                    str(utterance_id),
                    audio[index, 0].detach().cpu(),
                    original_generated[index, 0].detach().cpu(),
                    sae_generated[index, 0].detach().cpu(),
                ))
                if len(previews) >= 2:
                    break
    generator.train()
    return {
        "original_mel_l1": totals["original"] / max(examples, 1),
        "sae_mel_l1": totals["sae"] / max(examples, 1),
        "examples": int(examples),
    }, previews


def _save_dual_previews(
    output_dir: Path,
    step: int,
    previews: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
    sample_rate: int,
) -> None:
    step_dir = output_dir / "samples" / f"step_{step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for index, (utterance_id, reference, original, sae) in enumerate(previews):
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in utterance_id)
        prefix = step_dir / f"{index:02d}_{safe_id}"
        save_waveform(prefix.with_name(prefix.name + "_reference.wav"), reference, sample_rate)
        save_waveform(
            prefix.with_name(prefix.name + "_original_generated.wav"), original, sample_rate,
        )
        save_waveform(prefix.with_name(prefix.name + "_sae_generated.wav"), sae, sample_rate)


def _load_complete_state(
    path: Path,
    device: str,
    learning_rate: float,
) -> tuple[
    DirectSpearHiFiGenerator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    dict[str, Any],
]:
    generator, payload = load_direct_hifigan(path, device)
    missing = [
        key for key in ("mpd", "msd", "optimizer_g", "optimizer_d")
        if payload.get(key) is None
    ]
    if missing:
        raise AnalysisError(
            f"SAE-aware continuation requires a full periodic/resume checkpoint; "
            f"{path} is missing {missing}."
        )
    config = generator.config
    mpd = MultiPeriodDiscriminator(config).to(device)
    msd = MultiScaleDiscriminator(config).to(device)
    mpd.load_state_dict(payload["mpd"])
    msd.load_state_dict(payload["msd"])
    optimizer_g = torch.optim.AdamW(generator.parameters(), float(learning_rate))
    optimizer_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()), float(learning_rate),
    )
    optimizer_g.load_state_dict(payload["optimizer_g"])
    optimizer_d.load_state_dict(payload["optimizer_d"])
    for optimizer in (optimizer_g, optimizer_d):
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    generator.train()
    mpd.train()
    msd.train()
    return generator, mpd, msd, optimizer_g, optimizer_d, payload


def train_sae_aware_direct_hifigan(
    cache_dir: Path,
    data_root: Path,
    sae_checkpoint: Path,
    base_vocoder_checkpoint: Path,
    output_dir: Path,
    *,
    device: str | None = None,
    additional_steps: int = 20_000,
    batch_size: int = 16,
    segment_frames: int = 24,
    sae_fraction: float = 0.5,
    learning_rate: float = 2e-4,
    mel_weight: float = 45.0,
    validation_interval: int = 5_000,
    checkpoint_interval: int = 5_000,
    log_interval: int = 25,
    validation_batches: int = 32,
    num_workers: int = 8,
    keep_periodic: int = 3,
    seed: int = 42,
    fp16: bool = True,
) -> Path:
    if not 0.0 <= float(sae_fraction) <= 1.0:
        raise AnalysisError("--sae-fraction must lie in [0,1].")
    if int(additional_steps) < 1:
        raise AnalysisError("--additional-steps must be positive.")
    set_seed(seed)
    device = _device(device)
    cache = SpearAudioFeatureCache(cache_dir, data_root)
    if int(cache.manifest["sample_rate"]) != 16_000:
        raise AnalysisError("SAE-aware direct HiFi-GAN currently requires 16 kHz audio.")
    if int(cache.manifest["spear_hop_samples"]) != 320:
        raise AnalysisError("SAE-aware direct HiFi-GAN requires a 320-sample SPEAR hop.")
    reconstructor = FrozenSAEReconstructor(sae_checkpoint, cache.manifest).to(device)

    output_dir = output_dir.resolve()
    base_vocoder_checkpoint = base_vocoder_checkpoint.resolve()
    if output_dir == base_vocoder_checkpoint.parent:
        raise AnalysisError("SAE-aware output directory must not overwrite base vocoder training.")
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    load_path = last_path if last_path.exists() else base_vocoder_checkpoint
    generator, mpd, msd, optimizer_g, optimizer_d, payload = _load_complete_state(
        load_path, device, learning_rate,
    )
    if int(generator.config.input_dim) != int(cache.manifest["input_dim"]):
        raise AnalysisError(
            f"Vocoder input dimension {generator.config.input_dim} does not match "
            f"cache dimension {cache.manifest['input_dim']}."
        )
    loaded_step = int(payload.get("step", 0))
    previous_training = dict(payload.get("training", {}))
    if last_path.exists():
        adaptation = dict(previous_training.get("sae_aware_adaptation", {}))
        if not adaptation:
            raise AnalysisError(f"Existing {last_path} is not an SAE-aware continuation.")
        recorded_base = Path(str(adaptation.get("base_vocoder_checkpoint", ""))).resolve()
        if recorded_base != base_vocoder_checkpoint:
            raise AnalysisError(
                f"Existing continuation uses base {recorded_base}, requested "
                f"{base_vocoder_checkpoint}."
            )
        if Path(str(adaptation.get("sae_checkpoint", ""))).resolve() != sae_checkpoint.resolve():
            raise AnalysisError("Existing continuation uses a different SAE checkpoint.")
        if abs(float(adaptation.get("sae_fraction", -1)) - float(sae_fraction)) > 1e-12:
            raise AnalysisError("Existing continuation uses a different SAE mixture fraction.")
        if int(adaptation.get("additional_steps", -1)) != int(additional_steps):
            raise AnalysisError("Existing continuation uses a different additional-step target.")
        if abs(float(adaptation.get("learning_rate", -1)) - float(learning_rate)) > 1e-12:
            raise AnalysisError("Existing continuation uses a different learning rate.")
        adaptation_start = int(adaptation["adaptation_start_step"])
        history = list(previous_training.get("history", []))
    else:
        adaptation_start = loaded_step
        history = []
    del payload
    step = loaded_step
    target_step = adaptation_start + int(additional_steps)
    if step > target_step:
        raise AnalysisError(
            f"Continuation is already at step {step}, beyond requested target {target_step}."
        )

    train_dataset = DirectVocoderDataset(
        cache, "train", segment_frames, random_crop=True, seed=seed,
    )
    validation_dataset = DirectVocoderDataset(
        cache, "validation", segment_frames, random_crop=False, seed=seed + 1,
    )
    generator_state = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=int(batch_size), shuffle=True,
        drop_last=len(train_dataset) >= int(batch_size), num_workers=int(num_workers),
        pin_memory=str(device).startswith("cuda"), persistent_workers=int(num_workers) > 0,
        worker_init_fn=_worker_seed, generator=generator_state,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=min(int(batch_size), 4), shuffle=False, num_workers=0,
    )
    frontend = _mel_frontend(generator.config, device)
    use_amp = bool(fp16 and str(device).startswith("cuda"))
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_sae = min(
        (float(row["validation_sae_mel_l1"]) for row in history if "validation_sae_mel_l1" in row),
        default=float("inf"),
    )
    best_original = min(
        (float(row["validation_original_mel_l1"]) for row in history if "validation_original_mel_l1" in row),
        default=float("inf"),
    )
    best_sae_path = output_dir / "best_sae.pt"
    best_original_path = output_dir / "best_original.pt"
    adaptation_record = {
        "base_vocoder_checkpoint": str(base_vocoder_checkpoint),
        "sae_checkpoint": str(sae_checkpoint.resolve()),
        "adaptation_start_step": int(adaptation_start),
        "adaptation_target_step": int(target_step),
        "additional_steps": int(additional_steps),
        "sae_fraction": float(sae_fraction),
        "learning_rate": float(learning_rate),
    }
    epoch = 0
    while step < target_step:
        epoch += 1
        for features, audio, _ in train_loader:
            if step >= target_step:
                break
            step += 1
            adaptation_step = step - adaptation_start
            started = time.monotonic()
            features = features.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            with torch.no_grad():
                reconstructed = reconstructor(features)
                conditioning, sae_mask = _mixed_conditioning(
                    features, reconstructed, sae_fraction,
                )
            with _autocast(device, use_amp):
                generated = generator(conditioning)
                if generated.shape != audio.shape:
                    raise AnalysisError(
                        f"Generator output {tuple(generated.shape)} does not match "
                        f"audio {tuple(audio.shape)}."
                    )
                target_mel = frontend(audio.squeeze(1))
                generated_mel = frontend(generated.squeeze(1))
                mel_l1 = F.l1_loss(generated_mel, target_mel)
            if not torch.isfinite(generated).all() or not torch.isfinite(mel_l1):
                raise AnalysisError(f"Non-finite output/loss at adaptation step {adaptation_step}.")

            _set_requires_grad(mpd, True)
            _set_requires_grad(msd, True)
            optimizer_d.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                mpd_real, mpd_fake, _, _ = mpd(audio, generated.detach())
                msd_real, msd_fake, _, _ = msd(audio, generated.detach())
                loss_d = discriminator_loss(mpd_real, mpd_fake) + discriminator_loss(msd_real, msd_fake)
            if not torch.isfinite(loss_d):
                raise AnalysisError(f"Non-finite discriminator loss at adaptation step {adaptation_step}.")
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            optimizer_g.zero_grad(set_to_none=True)
            _set_requires_grad(mpd, False)
            _set_requires_grad(msd, False)
            with _autocast(device, use_amp):
                _, mpd_fake, mpd_real_maps, mpd_fake_maps = mpd(audio, generated)
                _, msd_fake, msd_real_maps, msd_fake_maps = msd(audio, generated)
                loss_adv = generator_adversarial_loss(mpd_fake) + generator_adversarial_loss(msd_fake)
                loss_fm = feature_matching_loss(mpd_real_maps, mpd_fake_maps)
                loss_fm = loss_fm + feature_matching_loss(msd_real_maps, msd_fake_maps)
                loss_g = float(mel_weight) * mel_l1 + loss_adv + loss_fm
            if not torch.isfinite(loss_g):
                raise AnalysisError(f"Non-finite generator loss at adaptation step {adaptation_step}.")
            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
            scaler_g.step(optimizer_g)
            scaler_g.update()
            _set_requires_grad(mpd, True)
            _set_requires_grad(msd, True)

            if adaptation_step == 1 or adaptation_step % int(log_interval) == 0:
                row = {
                    "step": int(step), "adaptation_step": int(adaptation_step),
                    "epoch": int(epoch), "generator_loss": float(loss_g.detach()),
                    "mel_l1": float(mel_l1.detach()),
                    "adversarial_loss": float(loss_adv.detach()),
                    "feature_matching_loss": float(loss_fm.detach()),
                    "discriminator_loss": float(loss_d.detach()),
                    "sae_examples": int(sae_mask.sum().item()),
                    "original_examples": int((~sae_mask).sum().item()),
                    "seconds_per_step": float(time.monotonic() - started),
                }
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(
                    f"[sae-aware-hifigan] step {adaptation_step:,}/{additional_steps:,} "
                    f"(global {step:,}) mel={row['mel_l1']:.5f} "
                    f"G={row['generator_loss']:.3f} D={row['discriminator_loss']:.3f} "
                    f"SAE/orig={row['sae_examples']}/{row['original_examples']} "
                    f"{row['seconds_per_step']:.3f}s/step",
                    flush=True,
                )

            should_validate = (
                adaptation_step == 1
                or adaptation_step % int(validation_interval) == 0
                or step == target_step
            )
            training_record = {
                "format": "direct_spear_hifigan_sae_aware_training_v1",
                "cache": str(cache.root), "data": str(cache.data_root),
                "spear_domain": {
                    "input_dim": int(cache.manifest["input_dim"]),
                    "sample_rate": int(cache.manifest["sample_rate"]),
                    "spear_hop_samples": int(cache.manifest["spear_hop_samples"]),
                    "spear_model_id": str(cache.manifest.get("spear_model_id", "")),
                    "spear_revision": str(cache.manifest.get("spear_revision", "")),
                    "spear_layernorm": bool(cache.manifest.get("spear_layernorm", False)),
                },
                "sae_aware_adaptation": adaptation_record,
                "test_examples_used": 0, "segment_frames": int(segment_frames),
                "seed": int(seed), "history": history,
            }
            if should_validate:
                validation, previews = _validate_domains(
                    generator, reconstructor, validation_loader, frontend,
                    device, validation_batches,
                )
                validation_row = {
                    "step": int(step), "adaptation_step": int(adaptation_step),
                    "validation_original_mel_l1": float(validation["original_mel_l1"]),
                    "validation_sae_mel_l1": float(validation["sae_mel_l1"]),
                    "validation_examples": int(validation["examples"]),
                }
                history.append(validation_row)
                training_record["history"] = history
                with (output_dir / "validation.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(validation_row, sort_keys=True) + "\n")
                _save_dual_previews(output_dir, step, previews, generator.config.sample_rate)
                print(
                    f"[sae-aware-hifigan] validation global step {step:,}: "
                    f"original={validation['original_mel_l1']:.5f} "
                    f"SAE={validation['sae_mel_l1']:.5f} "
                    f"({validation['examples']} examples)",
                    flush=True,
                )
                if float(validation["sae_mel_l1"]) < best_sae:
                    best_sae = float(validation["sae_mel_l1"])
                    save_direct_hifigan(
                        best_sae_path, generator, step=step,
                        training=training_record | {"best_validation_sae_mel_l1": best_sae},
                    )
                if float(validation["original_mel_l1"]) < best_original:
                    best_original = float(validation["original_mel_l1"])
                    save_direct_hifigan(
                        best_original_path, generator, step=step,
                        training=training_record | {
                            "best_validation_original_mel_l1": best_original,
                        },
                    )
            should_checkpoint = adaptation_step % int(checkpoint_interval) == 0
            if should_validate or should_checkpoint or step == target_step:
                save_direct_hifigan(
                    last_path, generator, step=step, training=training_record,
                    mpd=mpd, msd=msd, optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                )
            if should_checkpoint:
                _rolling_checkpoint(
                    output_dir, keep_periodic, generator=generator, step=step,
                    training=training_record, mpd=mpd, msd=msd,
                    optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                )

    manifest = {
        "format": "direct_spear_hifigan_sae_aware_training_v1",
        "device": str(device), "python": platform.python_version(),
        "torch": torch.__version__, "steps": int(step),
        "adaptation_steps": int(step - adaptation_start),
        "train_utterances": len(train_dataset),
        "validation_utterances": len(validation_dataset),
        "best_validation_sae_mel_l1": float(best_sae),
        "best_validation_original_mel_l1": float(best_original),
        "best_sae_checkpoint": str(best_sae_path),
        "best_original_checkpoint": str(best_original_path),
        "last_checkpoint": str(last_path),
        "sae_aware_adaptation": adaptation_record,
        "history": history,
    }
    write_json(output_dir / "training_manifest.json", manifest)
    return best_sae_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune direct HiFi-GAN on mixed original/SAE conditioning.",
    )
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--sae-checkpoint", required=True, type=Path)
    parser.add_argument("--base-vocoder-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--additional-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--segment-frames", type=int, default=24)
    parser.add_argument("--sae-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--mel-weight", type=float, default=45.0)
    parser.add_argument("--validation-interval", type=int, default=5_000)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--keep-periodic", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        best = train_sae_aware_direct_hifigan(
            args.cache, args.data_root, args.sae_checkpoint,
            args.base_vocoder_checkpoint, args.output_dir,
            device=args.device, additional_steps=args.additional_steps,
            batch_size=args.batch_size, segment_frames=args.segment_frames,
            sae_fraction=args.sae_fraction, learning_rate=args.learning_rate,
            mel_weight=args.mel_weight, validation_interval=args.validation_interval,
            checkpoint_interval=args.checkpoint_interval, log_interval=args.log_interval,
            validation_batches=args.validation_batches, num_workers=args.num_workers,
            keep_periodic=args.keep_periodic, seed=args.seed, fp16=not args.no_fp16,
        )
    except AnalysisError as exc:
        raise SystemExit(f"[sae-aware-hifigan] ERROR: {exc}") from exc
    print(f"[sae-aware-hifigan] best SAE-mel checkpoint: {best}")


if __name__ == "__main__":
    main()
