"""TensorBoard logger for the disentanglement system."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

_LAYOUT = {
    "Training": {
        "Recon MSE":         ["Multiline", ["train/recon"]],
        "Decor loss (raw)":  ["Multiline", ["train/decor"]],
        "Decor weighted":    ["Multiline", ["train/decor_weighted"]],
        "PR CTC":            ["Multiline", ["train/pr"]],
        "SID CE":            ["Multiline", ["train/sid"]],
        "GRL":               ["Multiline", ["train/grl"]],
        "Route entropy loss":["Multiline", ["train/route"]],
        "Total":             ["Multiline", ["train/total"]],
    },
    "Validation": {
        "Val Recon MSE":  ["Multiline", ["val/recon"]],
        "Val PR CTC":     ["Multiline", ["val/pr"]],
        "Val PER":        ["Multiline", ["val/per"]],
        "Val SID Acc":    ["Multiline", ["val/sid_acc"]],
    },
    "Routing": {
        "Feature counts (L/P/U)": ["Multiline", ["routing/count_L","routing/count_P","routing/count_U"]],
        "Fractions":              ["Multiline", ["routing/frac_L", "routing/frac_P", "routing/frac_U"]],
        "Entropy (nats)":         ["Multiline", ["routing/entropy", "routing/unit_entropy"]],
        "Specialisation":         ["Multiline", ["routing/specialized_frac_h_lt_0_5",
                                                  "routing/specialized_frac_h_lt_0_8",
                                                  "routing/top1_top2_margin"]],
        "Logit stats":            ["Multiline", ["routing/logit_std", "routing/logit_range"]],
    },
    "SAE": {
        "z_pre positive fraction": ["Multiline", ["sae/z_dense_density"]],
        "Dead latent fraction":    ["Multiline", ["train/dead_frac"]],
    },
    "Gradient Norms": {
        "Stage-1 |g|":           ["Multiline", ["grad_norms/recon","grad_norms/decor"]],
        "Raw |g| per loss":      ["Multiline", ["grad_norms/recon","grad_norms/pr_raw","grad_norms/sid_raw","grad_norms/grl","grad_norms/route"]],
        "Weighted |g| per loss": ["Multiline", ["grad_norms/recon","grad_norms/pr_weighted","grad_norms/sid_weighted","grad_norms/grl","grad_norms/route"]],
        "Ratio to recon (raw)":  ["Multiline", ["grad_norms/ratio_pr","grad_norms/ratio_sid","grad_norms/ratio_grl","grad_norms/ratio_route"]],
    },
    "Gradient Conflict": {
        "Recon vs tasks":   ["Multiline", ["grad_cos/recon_vs_pr","grad_cos/recon_vs_sid","grad_cos/recon_vs_grl","grad_cos/recon_vs_grl_p"]],
        "PR vs SID":        ["Multiline", ["grad_cos/pr_vs_sid"]],
        "Task vs adversary":["Multiline", ["grad_cos/pr_vs_grl","grad_cos/sid_vs_grl","grad_cos/pr_vs_grl_p","grad_cos/sid_vs_grl_p"]],
        "Adversary vs adversary":["Multiline", ["grad_cos/grl_vs_grl_p"]],
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

    def log_val(self, step: int, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            self._add(f"val/{k}", v, step)

    def log_routing(
        self,
        step: int,
        n_L: int,
        n_P: int,
        n_U: int,
        entropy: float,
        diagnostics: Optional[Dict[str, float]] = None,
    ) -> None:
        total = max(n_L + n_P + n_U, 1)
        self._add("routing/count_L",  n_L,          step)
        self._add("routing/count_P",  n_P,          step)
        self._add("routing/count_U",  n_U,          step)
        self._add("routing/frac_L",   n_L / total,  step)
        self._add("routing/frac_P",   n_P / total,  step)
        self._add("routing/frac_U",   n_U / total,  step)
        self._add("routing/entropy",  entropy,      step)
        if diagnostics:
            for k, v in diagnostics.items():
                self._add(f"routing/{k}", v, step)

    def log_sae(self, step: int, z_pre_pos_frac: float) -> None:
        self._add("sae/z_dense_density", z_pre_pos_frac, step)

    def log_grad_norms(self, step: int, norms: Dict[str, float]) -> None:
        for k, v in norms.items():
            self._add(f"grad_norms/{k}", v, step)
        recon = norms.get("recon", 0.0)
        if recon > 1e-8:
            for k in ("pr_raw", "sid_raw", "grl", "route"):
                if k in norms:
                    self._add(f"grad_norms/ratio_{k.replace('_raw','')}", norms[k] / recon, step)

    def log_grad_cosines(self, step: int, cosines: Dict[str, float]) -> None:
        """Pairwise gradient cosines between per-loss gradients on the shared SAE trunk."""
        for k, v in cosines.items():
            self._add(f"grad_cos/{k}", v, step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
