"""TensorBoard logger for the SAE reconstruction system."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

_LAYOUT = {
    "Training": {
        "Recon MSE": ["Multiline", ["train/recon"]],
    },
    "SAE": {
        "z_pre positive fraction": ["Multiline", ["sae/z_dense_density"]],
        "Active fraction of TopK": ["Multiline", ["sae/active_frac"]],
    },
    "Gradient Norms": {
        "SAE |g| (recon)": ["Multiline", ["grad_norms/recon"]],
    },
    "Validation": {
        "Val Recon MSE": ["Multiline", ["val/recon"]],
    },
}


class DISLogger:
    def __init__(self, log_dir: Path, run_name: str = "", disabled: bool = False) -> None:
        self._writer = None
        if disabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            out = Path(log_dir) / run_name if run_name else Path(log_dir)
            out.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(out), flush_secs=30)
            self._writer.add_custom_scalars(_LAYOUT)
        except ImportError:
            print("[DISLogger] tensorboard not installed — logging disabled")

    def _add(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_train(self, step: int, losses: Dict[str, float]) -> None:
        for k, v in losses.items():
            self._add(f"train/{k}", v, step)

    def log_val(self, step: int, recon: float) -> None:
        self._add("val/recon", recon, step)

    def log_sae(self, step: int, z_pre_pos_frac: float, topk: int) -> None:
        self._add("sae/z_dense_density", z_pre_pos_frac, step)
        self._add("sae/active_frac", topk / max(topk, 1), step)

    def log_grad_norms(self, step: int, norms: Dict[str, float]) -> None:
        for k, v in norms.items():
            self._add(f"grad_norms/{k}", v, step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
