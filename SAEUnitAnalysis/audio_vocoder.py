"""Optional waveform backends for the SPEAR-to-mel bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
import sys

import numpy as np
import torch

from .audio_bridge import MelConfig
from .utils import AnalysisError


class Vocoder(Protocol):
    sample_rate: int

    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor: ...


class BigVGANVocoder:
    """Lazy adapter for NVIDIA BigVGAN's official ``from_pretrained`` API."""

    def __init__(
        self,
        model_id: str,
        mel_config: MelConfig,
        *,
        device: str = "cpu",
        use_cuda_kernel: bool = False,
        repo_path: Path | None = None,
    ) -> None:
        if repo_path is not None:
            repo_path = repo_path.resolve()
            if not (repo_path / "bigvgan.py").exists():
                raise AnalysisError(f"BigVGAN repository is invalid: {repo_path}")
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
        try:
            from bigvgan import BigVGAN
        except ImportError as exc:
            raise AnalysisError(
                "BigVGAN is optional and is not installed. Install NVIDIA/BigVGAN "
                "in the audio environment, or use --vocoder griffinlim for a diagnostic smoke test."
            ) from exc
        try:
            model = BigVGAN.from_pretrained(
                model_id, use_cuda_kernel=bool(use_cuda_kernel),
            )
        except TypeError:
            # Compatibility with releases predating the CUDA-kernel keyword.
            model = BigVGAN.from_pretrained(model_id)
        if hasattr(model, "remove_weight_norm"):
            model.remove_weight_norm()
        self.model = model.eval().to(device)
        self.device = device
        self.sample_rate = int(mel_config.sample_rate)
        self.mel_config = mel_config
        self._validate_model_config()

    def _validate_model_config(self) -> None:
        config = getattr(self.model, "h", None) or getattr(self.model, "config", None)
        if config is None:
            return

        def get(*names: str):
            for name in names:
                if isinstance(config, dict) and name in config:
                    return config[name]
                if hasattr(config, name):
                    return getattr(config, name)
            return None

        expected = {
            "sample_rate": self.mel_config.sample_rate,
            "n_fft": self.mel_config.n_fft,
            "win_length": self.mel_config.win_length,
            "hop_length": self.mel_config.hop_length,
            "n_mels": self.mel_config.n_mels,
        }
        actual = {
            "sample_rate": get("sampling_rate", "sample_rate"),
            "n_fft": get("n_fft"),
            "win_length": get("win_size", "win_length"),
            "hop_length": get("hop_size", "hop_length"),
            "n_mels": get("num_mels", "n_mel_channels", "n_mels"),
        }
        mismatch = {
            key: {"bridge": expected[key], "vocoder": int(value)}
            for key, value in actual.items()
            if value is not None and int(value) != int(expected[key])
        }
        if mismatch:
            raise AnalysisError(f"Bridge mel configuration does not match BigVGAN: {mismatch}")

    @torch.inference_mode()
    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        generated = self.model(log_mel.to(self.device))
        if isinstance(generated, (tuple, list)):
            generated = generated[0]
        if generated.ndim == 3 and generated.shape[1] == 1:
            generated = generated[:, 0]
        return generated.float().clamp(-1, 1)


class GriffinLimVocoder:
    """Dependency-light diagnostic inversion; never use it for quality claims."""

    def __init__(self, mel_config: MelConfig, *, device: str = "cpu", iterations: int = 32) -> None:
        try:
            import torchaudio
        except ImportError as exc:
            raise AnalysisError("Griffin-Lim diagnostic inversion requires torchaudio.") from exc
        self.sample_rate = int(mel_config.sample_rate)
        self.device = device
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=mel_config.n_fft // 2 + 1,
            n_mels=mel_config.n_mels,
            sample_rate=mel_config.sample_rate,
            f_min=mel_config.f_min,
            f_max=mel_config.f_max,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=mel_config.n_fft,
            win_length=mel_config.win_length,
            hop_length=mel_config.hop_length,
            power=1.0,
            n_iter=int(iterations),
        ).to(device)

    @torch.inference_mode()
    def __call__(self, log_mel: torch.Tensor) -> torch.Tensor:
        magnitude = torch.exp(log_mel.to(self.device))
        linear = self.inverse_mel(magnitude)
        waveform = self.griffin_lim(linear)
        peak = waveform.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
        return (waveform / peak).float()


def load_vocoder(
    backend: str,
    mel_config: MelConfig,
    *,
    device: str,
    model_id: str = "nvidia/bigvgan_v2_24khz_100band_256x",
    bigvgan_repo: Path | None = None,
) -> Vocoder:
    backend = str(backend).lower()
    if backend == "bigvgan":
        return BigVGANVocoder(
            model_id, mel_config, device=device, repo_path=bigvgan_repo,
        )
    if backend == "griffinlim":
        return GriffinLimVocoder(mel_config, device=device)
    raise AnalysisError(f"Unknown vocoder backend {backend!r}; choose bigvgan or griffinlim.")


def save_waveform(path: Path, waveform: torch.Tensor | np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise AnalysisError("Writing generated audio requires soundfile.") from exc
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().float().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32).squeeze()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(waveform, -1, 1), int(sample_rate), subtype="PCM_16")
