"""Shared TensorBoard logger for ER / SID / PR probing tasks.

Wraps torch.utils.tensorboard.SummaryWriter with a consistent API so all
task-specific training loops can log the same signals without boilerplate.
Falls back to a no-op if tensorboard is not installed.

Key design decisions
--------------------
- Layer weights use individual add_scalar(f"layer_weights/layer_XX", ...) calls
  so all 13 traces land in the SAME event file and TensorBoard groups them
  under "layer_weights/" in the Scalars tab.  (add_scalars creates one
  subdirectory per key, which explodes the run selector with 13 extra entries.)
- add_custom_scalars is called once at init to define named dashboard panels in
  the "Custom Scalars" tab, giving a task-specific organised view.

Usage
-----
    from tb_logger import TBLogger

    tb = TBLogger(cfg.runs_dir / "tb", run_name="fold3", task="er")
    tb.log_train_step(step=100, loss=0.42, lr=1e-4)
    tb.log_eval(epoch=5, split="val", metrics={"acc": 0.71})
    tb.log_layer_weights(epoch=5, weights=[0.06, 0.08, ...])
    tb.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# --------------------------------------------------------- layout helpers ---


def _layer_tags(num_layers: int) -> List[str]:
    return [f"layer_weights/layer_{i:02d}" for i in range(num_layers)]


def _build_layout(task: str, num_layers: int) -> dict:
    """Return an add_custom_scalars layout dict for a given task."""
    lw_tags = _layer_tags(num_layers)

    train_panel: dict = {
        "Training": {
            "Loss":          ["Multiline", ["train/loss"]],
            "Learning Rate": ["Multiline", ["train/lr"]],
        },
    }

    if task in ("er", "sid"):
        eval_panel = {
            "Accuracy": {
                "Val & Test Accuracy": ["Multiline", ["val/acc", "test/acc"]],
            },
        }
    elif task == "pr":
        eval_panel = {
            "Phone Error Rate": {
                "PER (lower is better)": ["Multiline", ["val/per", "test/per"]],
            },
        }
    elif task == "asr":
        eval_panel = {
            "Error Rates": {
                "CER": ["Multiline", ["val/cer", "test/cer"]],
                "WER": ["Multiline", ["val/wer", "test/wer"]],
            },
        }
    else:
        eval_panel = {}

    lw_panel: dict = {}
    if lw_tags:
        lw_panel = {
            "Layer Weights": {
                "Softmax Mix (all layers)": ["Multiline", lw_tags],
            },
        }

    return {**train_panel, **eval_panel, **lw_panel}


# ----------------------------------------------------------------- logger ---


class TBLogger:
    """Lightweight TensorBoard writer with task-agnostic metric logging."""

    def __init__(
        self,
        log_dir: Union[Path, str],
        run_name: str = "",
        task: str = "",
        num_layers: int = 13,
    ) -> None:
        self._ok = _TB_AVAILABLE
        if self._ok:
            target = Path(log_dir) / run_name if run_name else Path(log_dir)
            target.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(target))
            if task:
                self._writer.add_custom_scalars(_build_layout(task, num_layers))
            print(f"[TBLogger] events → {target}")
        else:
            print("[TBLogger] tensorboard not installed — skipping TB logging")

    # ---------------------------------------------------------------- train

    def log_train_step(self, step: int, loss: float, lr: float) -> None:
        if not self._ok:
            return
        self._writer.add_scalar("train/loss", loss, step)
        self._writer.add_scalar("train/lr",   lr,   step)

    # ----------------------------------------------------------------- eval

    def log_eval(self, epoch: int, split: str, metrics: Dict[str, float]) -> None:
        """Log any numeric metrics dict under '{split}/{key}'."""
        if not self._ok:
            return
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                self._writer.add_scalar(f"{split}/{key}", float(val), epoch)

    # -------------------------------------------------------- layer weights

    def log_layer_weights(self, epoch: int, weights: Sequence[float]) -> None:
        """Log softmax layer-mixture weights as individual scalars.

        Uses add_scalar (not add_scalars) so all layers stay in one event
        file and the run selector stays uncluttered.
        """
        if not self._ok:
            return
        for i, w in enumerate(weights):
            self._writer.add_scalar(f"layer_weights/layer_{i:02d}", float(w), epoch)

    # ---------------------------------------------------------------- close

    def close(self) -> None:
        if self._ok:
            self._writer.flush()
            self._writer.close()
