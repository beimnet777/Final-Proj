"""TensorBoard logger for the disentanglement system.

Creates a SummaryWriter with task-specific named panels in the
'Custom Scalars' tab.  Falls back to a no-op if tensorboard is not installed.

Usage
-----
    tb = DISLogger(runs_dir / "tb", run_name="stage1_20250525")
    tb.log_train(step, losses_dict)
    tb.log_routing(step, counts_L, counts_P, counts_U, entropy)
    tb.log_layer_weights(step, weights)
    tb.log_probe(step, tag, metric_dict)
    tb.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

_LW_TAGS = [f"layer_weights/layer_{i:02d}" for i in range(13)]

_LAYOUT = {
    "Training": {
        "Recon MSE":     ["Multiline", ["train/recon"]],
        "PR CTC Loss":   ["Multiline", ["train/pr"]],
        "SID CE Loss":   ["Multiline", ["train/sid"]],
        "GRL Loss":      ["Multiline", ["train/grl"]],
        "Decorr":        ["Multiline", ["train/decorr"]],
        "Route":         ["Multiline", ["train/route"]],
        "Total":         ["Multiline", ["train/total"]],
    },
    "Routing": {
        "Group counts":  ["Multiline", ["routing/count_L", "routing/count_P", "routing/count_U"]],
        "Fraction":      ["Multiline", ["routing/frac_L",  "routing/frac_P",  "routing/frac_U"]],
        "Entropy (nats)":["Multiline", ["routing/entropy"]],
    },
    "SAE": {
        "z_pre positive fraction":          ["Multiline", ["sae/z_dense_density"]],
        "Active units per frame (L/P/U)":   ["Multiline", ["sae/active_L", "sae/active_P", "sae/active_U"]],
        "Active fraction of TopK budget":   ["Multiline", ["sae/active_frac_L", "sae/active_frac_P", "sae/active_frac_U"]],
    },
    "Fast Probes": {
        "SID acc (z̄_P → SID)":          ["Multiline", ["probe/sid_acc"]],
        "Leakage (z_L → SID)":           ["Multiline", ["probe/leak_sid"]],
        "PR CTC loss (val)":             ["Multiline", ["probe/pr_ctc_val"]],
    },
    "Layer Weights": {
        "Softmax mix": ["Multiline", _LW_TAGS],
    },
}


class DISLogger:
    def __init__(
        self,
        log_dir: Path,
        run_name: str = "",
        disabled: bool = False,
    ) -> None:
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

    # ---------------------------------------------------------------- core

    def _add(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    # ---------------------------------------------------------------- training losses

    def log_train(self, step: int, losses: Dict[str, float]) -> None:
        """losses keys: recon, pr, sid, grl, decorr, route, total."""
        for k, v in losses.items():
            self._add(f"train/{k}", v, step)

    # ---------------------------------------------------------------- routing

    def log_routing(
        self,
        step: int,
        count_L: int,
        count_P: int,
        count_U: int,
        entropy: float,
    ) -> None:
        total = max(count_L + count_P + count_U, 1)
        self._add("routing/count_L",  count_L,          step)
        self._add("routing/count_P",  count_P,          step)
        self._add("routing/count_U",  count_U,          step)
        self._add("routing/frac_L",   count_L / total,  step)
        self._add("routing/frac_P",   count_P / total,  step)
        self._add("routing/frac_U",   count_U / total,  step)
        self._add("routing/entropy",  entropy,          step)

    # ---------------------------------------------------------------- layer weights

    def log_layer_weights(self, step: int, weights: List[float]) -> None:
        for i, w in enumerate(weights):
            self._add(f"layer_weights/layer_{i:02d}", float(w), step)

    # ---------------------------------------------------------------- fast probes

    def log_probe(self, step: int, metrics: Dict[str, float]) -> None:
        """metrics keys: sid_acc, leak_sid, pr_ctc_val."""
        for k, v in metrics.items():
            self._add(f"probe/{k}", v, step)

    # ---------------------------------------------------------------- SAE monitoring

    def log_sae(
        self,
        step: int,
        z_pre_pos_frac: float,
        active_L: float,
        active_P: float,
        active_U: float,
        topk: int,
    ) -> None:
        """Log SAE activation statistics.

        z_pre_pos_frac : fraction of pre-TopK values that are positive
        active_L/P/U   : mean active features per frame in each bucket
                         (sum = topk always, since masks are one-hot)
        topk           : the TopK budget (for computing fractions)
        """
        self._add("sae/z_dense_density",  z_pre_pos_frac,        step)
        self._add("sae/active_L",         active_L,               step)
        self._add("sae/active_P",         active_P,               step)
        self._add("sae/active_U",         active_U,               step)
        self._add("sae/active_frac_L",    active_L / max(topk, 1), step)
        self._add("sae/active_frac_P",    active_P / max(topk, 1), step)
        self._add("sae/active_frac_U",    active_U / max(topk, 1), step)

    # ---------------------------------------------------------------- flush / close

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
