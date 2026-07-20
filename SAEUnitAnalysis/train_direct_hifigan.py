"""Train HiFi-GAN directly on cached frozen-SPEAR representations."""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .audio_bridge import LogMelFrontend, MelConfig
from .audio_vocoder import save_waveform
from .cache_spear_audio_features import SpearAudioFeatureCache
from .direct_hifigan import (
    DirectHiFiGANConfig,
    DirectSpearHiFiGenerator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    load_pretrained_knnvc_generator,
    save_direct_hifigan,
)
from .extraction import _read_audio
from .train_audio_bridge import _device
from .utils import AnalysisError, set_seed, write_json


KNNVC_REGULAR_GENERATOR = (
    "https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt"
)


class DirectVocoderDataset(Dataset):
    def __init__(
        self,
        cache: SpearAudioFeatureCache,
        split: str,
        segment_frames: int,
        *,
        random_crop: bool,
        seed: int,
    ) -> None:
        self.cache = cache
        self.rows = cache.split(split)
        self.segment_frames = int(segment_frames)
        self.random_crop = bool(random_crop)
        self.seed = int(seed)
        if self.rows.empty:
            raise AnalysisError(f"SPEAR audio cache has no {split!r} examples.")
        if self.segment_frames < 2:
            raise AnalysisError("--segment-frames must be at least 2.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows.iloc[int(index)]
        features = self.cache.feature(row).astype(np.float32, copy=True)
        audio = _read_audio(
            self.cache.audio_path(row), int(self.cache.manifest["sample_rate"]),
        ).astype(np.float32, copy=False)
        available = min(len(features), len(audio) // int(self.cache.manifest["spear_hop_samples"]))
        if available < 1:
            raise AnalysisError(f"Utterance {row['utterance_id']} has no aligned SPEAR/audio frames.")
        features = features[:available]
        audio = audio[:available * int(self.cache.manifest["spear_hop_samples"])]
        maximum_start = max(0, available - self.segment_frames)
        if self.random_crop and maximum_start:
            # Worker-local Python RNG is seeded by DataLoader worker initialization.
            start = random.randint(0, maximum_start)
        else:
            start = maximum_start // 2
        stop = start + self.segment_frames
        feature_segment = features[start:stop]
        hop = int(self.cache.manifest["spear_hop_samples"])
        audio_segment = audio[start * hop:stop * hop]
        if len(feature_segment) < self.segment_frames:
            feature_segment = np.pad(
                feature_segment,
                ((0, self.segment_frames - len(feature_segment)), (0, 0)),
            )
        target_samples = self.segment_frames * hop
        if len(audio_segment) < target_samples:
            audio_segment = np.pad(audio_segment, (0, target_samples - len(audio_segment)))
        return (
            torch.from_numpy(feature_segment),
            torch.from_numpy(audio_segment).unsqueeze(0),
            str(row["utterance_id"]),
        )


def _worker_seed(worker_id: int) -> None:
    seed = int(torch.initial_seed() % 2**32)
    np.random.seed(seed)
    random.seed(seed)


def _mel_frontend(config: DirectHiFiGANConfig, device: str) -> LogMelFrontend:
    return LogMelFrontend(MelConfig(
        sample_rate=config.sample_rate,
        n_fft=config.mel_n_fft,
        win_length=config.mel_win_length,
        hop_length=config.mel_hop_length,
        n_mels=config.mel_bins,
        f_min=config.mel_f_min,
        f_max=config.mel_f_max,
        center=False,
        reflect_pad=True,
    )).to(device)


def _autocast(device: str, enabled: bool):
    if enabled and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


@torch.no_grad()
def _validate(
    generator: DirectSpearHiFiGenerator,
    loader: DataLoader,
    frontend: LogMelFrontend,
    device: str,
    max_batches: int,
) -> tuple[dict[str, float], list[tuple[str, torch.Tensor, torch.Tensor]]]:
    generator.eval()
    total_l1 = 0.0
    total_examples = 0
    previews = []
    for batch_index, (features, audio, utterance_ids) in enumerate(loader):
        if int(max_batches) > 0 and batch_index >= int(max_batches):
            break
        features = features.to(device)
        audio = audio.to(device)
        generated = generator(features)
        if generated.shape[-1] != audio.shape[-1]:
            raise AnalysisError(
                f"Generator/audio length mismatch: {generated.shape[-1]} != {audio.shape[-1]}."
            )
        error = F.l1_loss(frontend(generated.squeeze(1)), frontend(audio.squeeze(1)))
        total_l1 += float(error) * len(utterance_ids)
        total_examples += len(utterance_ids)
        if len(previews) < 2:
            for index, utterance_id in enumerate(utterance_ids):
                previews.append((
                    str(utterance_id),
                    audio[index, 0].detach().cpu(),
                    generated[index, 0].detach().cpu(),
                ))
                if len(previews) >= 2:
                    break
    generator.train()
    return {
        "mel_l1": total_l1 / max(total_examples, 1),
        "examples": int(total_examples),
    }, previews


def _save_previews(
    output_dir: Path,
    step: int,
    previews: list[tuple[str, torch.Tensor, torch.Tensor]],
    sample_rate: int,
) -> None:
    step_dir = output_dir / "samples" / f"step_{step:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for index, (utterance_id, reference, generated) in enumerate(previews):
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in utterance_id)
        save_waveform(step_dir / f"{index:02d}_{safe_id}_reference.wav", reference, sample_rate)
        save_waveform(step_dir / f"{index:02d}_{safe_id}_generated.wav", generated, sample_rate)


def _load_resume(
    path: Path,
    generator: DirectSpearHiFiGenerator,
    mpd: MultiPeriodDiscriminator,
    msd: MultiScaleDiscriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: str,
) -> tuple[int, list[dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "direct_spear_hifigan_v1":
        raise AnalysisError(f"Unsupported resume checkpoint: {path}")
    raw_config = dict(payload["config"])
    for key in ("upsample_rates", "upsample_kernel_sizes", "resblock_kernel_sizes", "mpd_periods"):
        raw_config[key] = tuple(raw_config[key])
    raw_config["resblock_dilation_sizes"] = tuple(
        tuple(values) for values in raw_config["resblock_dilation_sizes"]
    )
    expected = DirectHiFiGANConfig(**raw_config)
    if expected != generator.config:
        raise AnalysisError("Resume checkpoint architecture does not match requested configuration.")
    generator.load_state_dict(payload["generator"])
    if payload.get("mpd") is not None:
        mpd.load_state_dict(payload["mpd"])
    if payload.get("msd") is not None:
        msd.load_state_dict(payload["msd"])
    if payload.get("optimizer_g") is not None:
        optimizer_g.load_state_dict(payload["optimizer_g"])
    if payload.get("optimizer_d") is not None:
        optimizer_d.load_state_dict(payload["optimizer_d"])
    for optimizer in (optimizer_g, optimizer_d):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    return int(payload.get("step", 0)), list(payload.get("training", {}).get("history", []))


def _rolling_checkpoint(
    output_dir: Path,
    keep: int,
    **save_kwargs: Any,
) -> None:
    step = int(save_kwargs["step"])
    path = output_dir / f"step_{step:08d}.pt"
    save_direct_hifigan(path, **save_kwargs)
    checkpoints = sorted(output_dir.glob("step_*.pt"))
    for old in checkpoints[:-max(1, int(keep))]:
        old.unlink()


def train_direct_hifigan(
    cache_dir: Path,
    output_dir: Path,
    *,
    data_root: Path | None = None,
    device: str | None = None,
    model_size: str = "full",
    max_steps: int = 250_000,
    batch_size: int = 16,
    segment_frames: int = 24,
    learning_rate: float = 2e-4,
    adam_b1: float = 0.8,
    adam_b2: float = 0.99,
    mel_weight: float = 45.0,
    adversarial_start_step: int = 5_000,
    validation_interval: int = 5_000,
    checkpoint_interval: int = 10_000,
    log_interval: int = 25,
    validation_batches: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    pretrained_generator: str | None = KNNVC_REGULAR_GENERATOR,
    resume: Path | None = None,
    keep_periodic: int = 3,
    fp16: bool = True,
) -> Path:
    set_seed(seed)
    device = _device(device)
    cache = SpearAudioFeatureCache(cache_dir, data_root)
    if int(cache.manifest["sample_rate"]) != 16_000 or int(cache.manifest["spear_hop_samples"]) != 320:
        raise AnalysisError("Direct HiFi-GAN currently requires 16 kHz audio and a 320-sample SPEAR hop.")
    config = (
        DirectHiFiGANConfig.smoke(cache.input_dim)
        if model_size == "smoke"
        else DirectHiFiGANConfig(input_dim=cache.input_dim)
    )
    config.validate()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = DirectVocoderDataset(
        cache, "train", segment_frames, random_crop=True, seed=seed,
    )
    validation_dataset = DirectVocoderDataset(
        cache, "validation", segment_frames, random_crop=False, seed=seed + 1,
    )
    generator_state = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=int(batch_size), shuffle=True, drop_last=len(train_dataset) >= batch_size,
        num_workers=int(num_workers), pin_memory=str(device).startswith("cuda"),
        persistent_workers=int(num_workers) > 0, worker_init_fn=_worker_seed,
        generator=generator_state,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=min(int(batch_size), 4), shuffle=False,
        num_workers=0,
    )
    generator = DirectSpearHiFiGenerator(config).to(device)
    mpd = MultiPeriodDiscriminator(config).to(device)
    msd = MultiScaleDiscriminator(config).to(device)
    frontend = _mel_frontend(config, device)
    optimizer_g = torch.optim.AdamW(
        generator.parameters(), float(learning_rate), betas=(float(adam_b1), float(adam_b2)),
    )
    optimizer_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        float(learning_rate), betas=(float(adam_b1), float(adam_b2)),
    )
    initialization: dict[str, Any] = {"kind": "random"}
    history: list[dict[str, Any]] = []
    start_step = 0
    if resume is not None:
        start_step, history = _load_resume(
            resume, generator, mpd, msd, optimizer_g, optimizer_d, device,
        )
        initialization = {"kind": "resume", "source": str(resume.resolve())}
    elif pretrained_generator and model_size == "full":
        initialization = {
            "kind": "knnvc_generator_warm_start",
            **load_pretrained_knnvc_generator(generator, pretrained_generator),
        }
        # Optimizer was created before weight loading but holds parameter references,
        # so it remains valid and starts with empty moments.
    use_amp = bool(fp16 and str(device).startswith("cuda"))
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_validation = min(
        (float(row["validation_mel_l1"]) for row in history if "validation_mel_l1" in row),
        default=float("inf"),
    )
    best_path = output_dir / "best.pt"
    step = int(start_step)
    epoch = 0
    generator.train()
    mpd.train()
    msd.train()
    while step < int(max_steps):
        epoch += 1
        for features, audio, _ in train_loader:
            if step >= int(max_steps):
                break
            step += 1
            started = time.monotonic()
            features = features.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            with _autocast(device, use_amp):
                generated = generator(features)
                if generated.shape != audio.shape:
                    raise AnalysisError(
                        f"Generator output {tuple(generated.shape)} does not match audio "
                        f"{tuple(audio.shape)}."
                    )
                target_mel = frontend(audio.squeeze(1))
                generated_mel = frontend(generated.squeeze(1))
                mel_l1 = F.l1_loss(generated_mel, target_mel)
            if not torch.isfinite(generated).all() or not torch.isfinite(mel_l1):
                raise AnalysisError(
                    f"Non-finite generator or mel loss at step {step}; refusing to save a corrupt model."
                )

            adversarial = step > int(adversarial_start_step)
            discriminator_value = 0.0
            if adversarial:
                _set_requires_grad(mpd, True)
                _set_requires_grad(msd, True)
                optimizer_d.zero_grad(set_to_none=True)
                with _autocast(device, use_amp):
                    mpd_real, mpd_fake, _, _ = mpd(audio, generated.detach())
                    msd_real, msd_fake, _, _ = msd(audio, generated.detach())
                    loss_d = discriminator_loss(mpd_real, mpd_fake) + discriminator_loss(msd_real, msd_fake)
                if not torch.isfinite(loss_d):
                    raise AnalysisError(
                        f"Non-finite discriminator loss at step {step}; refusing to continue."
                    )
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
                discriminator_value = float(loss_d.detach())

            optimizer_g.zero_grad(set_to_none=True)
            if adversarial:
                _set_requires_grad(mpd, False)
                _set_requires_grad(msd, False)
                with _autocast(device, use_amp):
                    _, mpd_fake, mpd_real_maps, mpd_fake_maps = mpd(audio, generated)
                    _, msd_fake, msd_real_maps, msd_fake_maps = msd(audio, generated)
                    loss_adv = generator_adversarial_loss(mpd_fake) + generator_adversarial_loss(msd_fake)
                    loss_fm = feature_matching_loss(mpd_real_maps, mpd_fake_maps)
                    loss_fm = loss_fm + feature_matching_loss(msd_real_maps, msd_fake_maps)
                    loss_g = float(mel_weight) * mel_l1 + loss_adv + loss_fm
            else:
                loss_adv = mel_l1.new_zeros(())
                loss_fm = mel_l1.new_zeros(())
                loss_g = float(mel_weight) * mel_l1
            if not torch.isfinite(loss_g):
                raise AnalysisError(
                    f"Non-finite generator loss at step {step}; refusing to continue."
                )
            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
            scaler_g.step(optimizer_g)
            scaler_g.update()
            _set_requires_grad(mpd, True)
            _set_requires_grad(msd, True)

            if step == 1 or step % int(log_interval) == 0:
                row = {
                    "step": int(step), "epoch": int(epoch),
                    "generator_loss": float(loss_g.detach()),
                    "mel_l1": float(mel_l1.detach()),
                    "adversarial_loss": float(loss_adv.detach()),
                    "feature_matching_loss": float(loss_fm.detach()),
                    "discriminator_loss": float(discriminator_value),
                    "adversarial_active": bool(adversarial),
                    "seconds_per_step": float(time.monotonic() - started),
                }
                with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(
                    f"[direct-hifigan] step {step:,}/{max_steps:,} "
                    f"mel={row['mel_l1']:.5f} G={row['generator_loss']:.3f} "
                    f"D={row['discriminator_loss']:.3f} "
                    f"{row['seconds_per_step']:.3f}s/step",
                    flush=True,
                )

            should_validate = step == 1 or step % int(validation_interval) == 0 or step == int(max_steps)
            training_record = {
                "cache": str(cache.root), "data": str(cache.data_root),
                "spear_domain": {
                    "input_dim": int(cache.manifest["input_dim"]),
                    "sample_rate": int(cache.manifest["sample_rate"]),
                    "spear_hop_samples": int(cache.manifest["spear_hop_samples"]),
                    "spear_model_id": str(cache.manifest.get("spear_model_id", "")),
                    "spear_revision": str(cache.manifest.get("spear_revision", "")),
                    "spear_layernorm": bool(cache.manifest.get("spear_layernorm", False)),
                },
                "test_examples_used": 0, "initialization": initialization,
                "max_steps": int(max_steps), "adversarial_start_step": int(adversarial_start_step),
                "segment_frames": int(segment_frames), "seed": int(seed),
                "history": history,
            }
            if should_validate:
                validation, previews = _validate(
                    generator, validation_loader, frontend, device, validation_batches,
                )
                validation_row = {
                    "step": int(step),
                    "validation_mel_l1": float(validation["mel_l1"]),
                    "validation_examples": int(validation["examples"]),
                }
                history.append(validation_row)
                training_record["history"] = history
                with (output_dir / "validation.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(validation_row, sort_keys=True) + "\n")
                _save_previews(output_dir, step, previews, config.sample_rate)
                print(
                    f"[direct-hifigan] validation step {step:,}: "
                    f"mel={validation['mel_l1']:.5f} ({validation['examples']} examples)",
                    flush=True,
                )
                if float(validation["mel_l1"]) < best_validation:
                    best_validation = float(validation["mel_l1"])
                    save_direct_hifigan(
                        best_path, generator, step=step,
                        training=training_record | {"best_validation_mel_l1": best_validation},
                    )
            should_checkpoint = step % int(checkpoint_interval) == 0
            if should_validate or should_checkpoint or step == int(max_steps):
                save_direct_hifigan(
                    output_dir / "last.pt", generator, step=step, training=training_record,
                    mpd=mpd, msd=msd, optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                )
            if should_checkpoint:
                _rolling_checkpoint(
                    output_dir, keep_periodic, generator=generator, step=step,
                    training=training_record, mpd=mpd, msd=msd,
                    optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                )

    write_json(output_dir / "training_manifest.json", {
        "format": "direct_spear_hifigan_training_v1",
        "cache": str(cache.root), "data": str(cache.data_root),
        "spear_domain": {
            "input_dim": int(cache.manifest["input_dim"]),
            "sample_rate": int(cache.manifest["sample_rate"]),
            "spear_hop_samples": int(cache.manifest["spear_hop_samples"]),
            "spear_model_id": str(cache.manifest.get("spear_model_id", "")),
            "spear_revision": str(cache.manifest.get("spear_revision", "")),
            "spear_layernorm": bool(cache.manifest.get("spear_layernorm", False)),
        },
        "device": str(device), "model_size": model_size,
        "python": platform.python_version(), "torch": torch.__version__,
        "test_examples_used": 0, "train_utterances": len(train_dataset),
        "validation_utterances": len(validation_dataset),
        "steps": int(step), "best_validation_mel_l1": float(best_validation),
        "best_checkpoint": str(best_path), "last_checkpoint": str(output_dir / "last.pt"),
        "initialization": initialization, "history": history,
    })
    return best_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a direct SPEAR-conditioned HiFi-GAN.")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-size", choices=("full", "smoke"), default="full")
    parser.add_argument("--max-steps", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--segment-frames", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--adam-b1", type=float, default=0.8)
    parser.add_argument("--adam-b2", type=float, default=0.99)
    parser.add_argument("--mel-weight", type=float, default=45.0)
    parser.add_argument("--adversarial-start-step", type=int, default=5_000)
    parser.add_argument("--validation-interval", type=int, default=5_000)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-generator", default=KNNVC_REGULAR_GENERATOR)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--keep-periodic", type=int, default=3)
    parser.add_argument("--no-fp16", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    pretrained = args.pretrained_generator
    if str(pretrained).strip().lower() in {"", "none", "null", "false"}:
        pretrained = None
    try:
        best = train_direct_hifigan(
            args.cache, args.output_dir, data_root=args.data_root,
            device=args.device, model_size=args.model_size,
            max_steps=args.max_steps, batch_size=args.batch_size,
            segment_frames=args.segment_frames, learning_rate=args.learning_rate,
            adam_b1=args.adam_b1, adam_b2=args.adam_b2,
            mel_weight=args.mel_weight,
            adversarial_start_step=args.adversarial_start_step,
            validation_interval=args.validation_interval,
            checkpoint_interval=args.checkpoint_interval,
            log_interval=args.log_interval,
            validation_batches=args.validation_batches,
            num_workers=args.num_workers, seed=args.seed,
            pretrained_generator=pretrained, resume=args.resume,
            keep_periodic=args.keep_periodic, fp16=not args.no_fp16,
        )
    except AnalysisError as exc:
        raise SystemExit(f"[direct-hifigan] ERROR: {exc}") from exc
    print(f"[direct-hifigan] best checkpoint: {best}")


if __name__ == "__main__":
    main()
