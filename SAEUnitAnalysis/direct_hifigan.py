"""Direct SPEAR-conditioned HiFi-GAN components.

The generator follows the HiFi-GAN V1 layout used by kNN-VC, but receives the
50 Hz, 1280-dimensional SPEAR representation directly. The 320x HiFi-GAN
upsampler gives exactly 320 waveform samples per input frame at 16 kHz. This
keeps the time base identical
to the frozen SPEAR encoder and permits warm-starting all compatible generator
layers from the public WavLM HiFi-GAN checkpoint.

Architecture conventions are adapted from the MIT-licensed kNN-VC HiFi-GAN:
https://github.com/bshall/knn-vc
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from .utils import AnalysisError


LRELU_SLOPE = 0.1


@dataclass(frozen=True)
class DirectHiFiGANConfig:
    input_dim: int = 1280
    projected_dim: int = 512
    sample_rate: int = 16_000
    spear_hop_samples: int = 320
    condition_upsample: int = 1
    upsample_rates: tuple[int, ...] = (10, 8, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (20, 16, 4, 4)
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5), (1, 3, 5), (1, 3, 5),
    )
    mpd_periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    discriminator_multiplier: float = 1.0
    msd_scales: int = 3
    mel_n_fft: int = 1024
    mel_win_length: int = 1024
    mel_hop_length: int = 320
    mel_bins: int = 80
    mel_f_min: float = 0.0
    mel_f_max: float = 8000.0

    def validate(self) -> None:
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("Upsample rates and kernels must have the same length.")
        if len(self.resblock_kernel_sizes) != len(self.resblock_dilation_sizes):
            raise ValueError("Each residual kernel requires a dilation tuple.")
        generator_hop = self.condition_upsample
        for rate in self.upsample_rates:
            generator_hop *= int(rate)
        if generator_hop != self.spear_hop_samples:
            raise ValueError(
                f"Generator upsamples by {generator_hop}, expected SPEAR hop "
                f"{self.spear_hop_samples}."
            )

    @classmethod
    def smoke(cls, input_dim: int = 1280) -> "DirectHiFiGANConfig":
        """Small architecture for wiring tests; it is not an audio-quality model."""
        return cls(
            input_dim=int(input_dim), projected_dim=32,
            upsample_initial_channel=64,
            resblock_kernel_sizes=(3,),
            resblock_dilation_sizes=((1, 3),),
            mpd_periods=(2, 3), discriminator_multiplier=0.125,
            msd_scales=1, mel_n_fft=256, mel_win_length=256,
            mel_hop_length=80, mel_bins=32,
        )


def _padding(kernel_size: int, dilation: int = 1) -> int:
    return (int(kernel_size) * int(dilation) - int(dilation)) // 2


def _init_conv(module: nn.Module) -> None:
    if isinstance(module, (Conv1d, Conv2d, ConvTranspose1d)):
        nn.init.normal_(module.weight, 0.0, 0.01)


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: Iterable[int]) -> None:
        super().__init__()
        dilations = tuple(int(value) for value in dilations)
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=dilation, padding=_padding(kernel_size, dilation),
            ))
            for dilation in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1, padding=_padding(kernel_size),
            ))
            for _ in dilations
        ])
        self.apply(_init_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for first, second in zip(self.convs1, self.convs2):
            update = first(F.leaky_relu(x, LRELU_SLOPE))
            update = second(F.leaky_relu(update, LRELU_SLOPE))
            x = x + update
        return x

    def remove_weight_norm(self) -> None:
        for layer in (*self.convs1, *self.convs2):
            remove_weight_norm(layer)


class DirectSpearHiFiGenerator(nn.Module):
    def __init__(self, config: DirectHiFiGANConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.lin_pre = nn.Linear(config.input_dim, config.projected_dim)
        self.conv_pre = weight_norm(Conv1d(
            config.projected_dim, config.upsample_initial_channel, 7, 1, padding=3,
        ))
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        channels = int(config.upsample_initial_channel)
        for rate, kernel in zip(config.upsample_rates, config.upsample_kernel_sizes):
            next_channels = channels // 2
            self.ups.append(weight_norm(ConvTranspose1d(
                channels, next_channels, kernel, rate, padding=(kernel - rate) // 2,
            )))
            for res_kernel, dilations in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes,
            ):
                self.resblocks.append(ResBlock(next_channels, res_kernel, dilations))
            channels = next_channels
        self.conv_post = weight_norm(Conv1d(channels, 1, 7, 1, padding=3))
        self.ups.apply(_init_conv)
        self.conv_pre.apply(_init_conv)
        self.conv_post.apply(_init_conv)
        nn.init.xavier_uniform_(self.lin_pre.weight)
        nn.init.zeros_(self.lin_pre.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected SPEAR features [B,T,{self.config.input_dim}], got "
                f"{tuple(features.shape)}."
            )
        x = features.transpose(1, 2)
        if self.config.condition_upsample != 1:
            x = F.interpolate(
                x, scale_factor=self.config.condition_upsample,
                mode="linear", align_corners=False,
            )
        x = x.transpose(1, 2)
        x = self.lin_pre(x).transpose(1, 2)
        x = self.conv_pre(x)
        n_kernels = len(self.config.resblock_kernel_sizes)
        for index, upsample in enumerate(self.ups):
            x = upsample(F.leaky_relu(x, LRELU_SLOPE))
            candidates = [
                self.resblocks[index * n_kernels + offset](x)
                for offset in range(n_kernels)
            ]
            x = torch.stack(candidates, dim=0).mean(dim=0)
        return torch.tanh(self.conv_post(F.leaky_relu(x, LRELU_SLOPE)))

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self.conv_pre)
        for layer in self.ups:
            remove_weight_norm(layer)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, multiplier: float = 1.0) -> None:
        super().__init__()
        self.period = int(period)
        channels = [
            max(4, int(value * multiplier))
            for value in (32, 128, 512, 1024, 1024)
        ]
        pairs = [(1, channels[0]), *zip(channels[:-1], channels[1:])]
        strides = (3, 3, 3, 3, 1)
        self.convs = nn.ModuleList([
            weight_norm(Conv2d(
                in_channels, out_channels, (5, 1), (stride, 1),
                padding=(2, 0),
            ))
            for (in_channels, out_channels), stride in zip(pairs, strides)
        ])
        self.conv_post = weight_norm(Conv2d(channels[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch, channels, time = x.shape
        if time % self.period:
            amount = self.period - time % self.period
            x = F.pad(x, (0, amount), mode="reflect")
            time += amount
        x = x.view(batch, channels, time // self.period, self.period)
        features = []
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return torch.flatten(x, 1), features


def _valid_groups(in_channels: int, out_channels: int, requested: int) -> int:
    for value in range(min(requested, in_channels, out_channels), 0, -1):
        if in_channels % value == 0 and out_channels % value == 0:
            return value
    return 1


class DiscriminatorS(nn.Module):
    def __init__(self, multiplier: float = 1.0, spectral: bool = False) -> None:
        super().__init__()
        norm = spectral_norm if spectral else weight_norm
        channels = [max(8, int(value * multiplier)) for value in (128, 128, 256, 512, 1024, 1024, 1024)]
        specifications = [
            (1, channels[0], 15, 1, 1),
            (channels[0], channels[1], 41, 2, 4),
            (channels[1], channels[2], 41, 2, 16),
            (channels[2], channels[3], 41, 4, 16),
            (channels[3], channels[4], 41, 4, 16),
            (channels[4], channels[5], 41, 1, 16),
            (channels[5], channels[6], 5, 1, 1),
        ]
        self.convs = nn.ModuleList([
            norm(Conv1d(
                in_channels, out_channels, kernel, stride,
                groups=_valid_groups(in_channels, out_channels, groups),
                padding=(kernel - 1) // 2,
            ))
            for in_channels, out_channels, kernel, stride, groups in specifications
        ])
        self.conv_post = norm(Conv1d(channels[-1], 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = []
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return torch.flatten(x, 1), features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, config: DirectHiFiGANConfig) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(period, config.discriminator_multiplier)
            for period in config.mpd_periods
        ])

    def forward(self, real: torch.Tensor, generated: torch.Tensor):
        return _run_discriminators(self.discriminators, real, generated)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, config: DirectHiFiGANConfig) -> None:
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(config.discriminator_multiplier, spectral=(index == 0))
            for index in range(config.msd_scales)
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2)
            for _ in range(max(0, config.msd_scales - 1))
        ])

    def forward(self, real: torch.Tensor, generated: torch.Tensor):
        real_outputs, generated_outputs, real_features, generated_features = [], [], [], []
        for index, discriminator in enumerate(self.discriminators):
            if index:
                real = self.meanpools[index - 1](real)
                generated = self.meanpools[index - 1](generated)
            real_score, real_map = discriminator(real)
            generated_score, generated_map = discriminator(generated)
            real_outputs.append(real_score)
            generated_outputs.append(generated_score)
            real_features.append(real_map)
            generated_features.append(generated_map)
        return real_outputs, generated_outputs, real_features, generated_features


def _run_discriminators(discriminators: nn.ModuleList, real: torch.Tensor, generated: torch.Tensor):
    real_outputs, generated_outputs, real_features, generated_features = [], [], [], []
    for discriminator in discriminators:
        real_score, real_map = discriminator(real)
        generated_score, generated_map = discriminator(generated)
        real_outputs.append(real_score)
        generated_outputs.append(generated_score)
        real_features.append(real_map)
        generated_features.append(generated_map)
    return real_outputs, generated_outputs, real_features, generated_features


def discriminator_loss(
    real_outputs: list[torch.Tensor], generated_outputs: list[torch.Tensor],
) -> torch.Tensor:
    return sum(
        torch.mean((1.0 - real) ** 2) + torch.mean(generated ** 2)
        for real, generated in zip(real_outputs, generated_outputs)
    )


def generator_adversarial_loss(outputs: list[torch.Tensor]) -> torch.Tensor:
    return sum(torch.mean((1.0 - output) ** 2) for output in outputs)


def feature_matching_loss(
    real_features: list[list[torch.Tensor]],
    generated_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    loss = next(iter(real_features[0])).new_zeros(())
    for real_discriminator, generated_discriminator in zip(real_features, generated_features):
        for real, generated in zip(real_discriminator, generated_discriminator):
            loss = loss + F.l1_loss(generated, real.detach())
    return 2.0 * loss


def load_pretrained_knnvc_generator(
    model: DirectSpearHiFiGenerator,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Warm-start all shape-compatible layers from a kNN-VC generator."""
    checkpoint_text = str(checkpoint)
    if checkpoint_text.startswith(("http://", "https://")):
        payload = torch.hub.load_state_dict_from_url(
            checkpoint_text, map_location="cpu", progress=True,
        )
    else:
        path = Path(checkpoint).expanduser().resolve()
        if not path.exists():
            raise AnalysisError(f"Pretrained HiFi-GAN checkpoint does not exist: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("generator", payload)
    target = model.state_dict()
    compatible = {
        name: value for name, value in source.items()
        if name in target and tuple(value.shape) == tuple(target[name].shape)
    }
    incompatible_shapes = {
        name: {"source": list(value.shape), "target": list(target[name].shape)}
        for name, value in source.items()
        if name in target and tuple(value.shape) != tuple(target[name].shape)
    }
    result = model.load_state_dict(compatible, strict=False)
    return {
        "source": checkpoint_text,
        "loaded_tensors": len(compatible),
        "total_target_tensors": len(target),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "incompatible_shapes": incompatible_shapes,
    }


def save_direct_hifigan(
    path: Path,
    generator: DirectSpearHiFiGenerator,
    *,
    step: int,
    training: dict[str, Any],
    mpd: MultiPeriodDiscriminator | None = None,
    msd: MultiScaleDiscriminator | None = None,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "direct_spear_hifigan_v1",
        "config": asdict(generator.config),
        "generator": generator.state_dict(),
        "mpd": mpd.state_dict() if mpd is not None else None,
        "msd": msd.state_dict() if msd is not None else None,
        "optimizer_g": optimizer_g.state_dict() if optimizer_g is not None else None,
        "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
        "step": int(step),
        "training": training,
    }, path)


def _config_from_mapping(raw: dict[str, Any]) -> DirectHiFiGANConfig:
    raw = dict(raw)
    for name in (
        "upsample_rates", "upsample_kernel_sizes", "resblock_kernel_sizes", "mpd_periods",
    ):
        if name in raw:
            raw[name] = tuple(raw[name])
    if "resblock_dilation_sizes" in raw:
        raw["resblock_dilation_sizes"] = tuple(
            tuple(values) for values in raw["resblock_dilation_sizes"]
        )
    return DirectHiFiGANConfig(**raw)


def load_direct_hifigan(
    path: Path,
    device: str | torch.device = "cpu",
) -> tuple[DirectSpearHiFiGenerator, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "direct_spear_hifigan_v1":
        raise AnalysisError(f"Unsupported direct HiFi-GAN checkpoint: {path}")
    model = DirectSpearHiFiGenerator(_config_from_mapping(payload["config"]))
    model.load_state_dict(payload["generator"])
    model.to(device).eval()
    return model, payload
