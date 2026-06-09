#!/usr/bin/env python3
"""Comprehensive analysis and plotting for all disentanglement experiments.

Per-experiment subfolders under analysis/<exp>/:
  training.png     — stage-1/2 loss curves (all losses, SID-CE included with caveat)
  validation.png   — val_recon + val_PER over steps
  grad_norms.png   — gradient norm evolution (stage-1 recon vs decor)
  routing.png      — L/P/U fractions + entropy over steps
  probe.png        — grouped bar chart of all probe metrics
  radar.png        — spider chart of 4 key disentanglement metrics
  summary.txt      — hyperparams, training observations, probe results

analysis/comparison/:
  scatter_disentanglement.png  — z_L→SID vs z_P→PR  (2 primary axes)
  scatter_speaker.png          — z_L→SID vs z_P→SID  (speaker push/pull)
  probe_heatmap.png            — experiments × probe metrics heatmap
  val_recon_groups.png         — val_recon convergence grouped by category
  routing_entropy_all.png      — routing entropy all runs (shows near-uniform)
  reconstruction_quality.png   — final val_recon bar chart per run

Special (decor_only only):
  analysis/decor_only/decor_analysis.png  — decorr loss + gradient spike
"""

from __future__ import annotations
import re
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

DIS_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = DIS_DIR / "logs"
OUT_DIR = DIS_DIR / "analysis"
STAGE1_LOG_DIR = LOG_DIR / "train" / "stage1"
STAGE2_LOG_DIR = LOG_DIR / "train" / "stage2"
PROBE_LOG_DIR = LOG_DIR / "probes"
HIST_PROBE_DIR = PROBE_LOG_DIR / "diagnostic_historical"
HIST_PROBE_SUCCESS_DIR = HIST_PROBE_DIR / "successful"

# ─────────────────────────────────────────── experiment registry

EXPERIMENTS = {
    "baseline": {
        "stage1_log": STAGE1_LOG_DIR / "sae_29858328.out",
        "stage2_log": None,
        "probe_log":  None,
        "hparams":    {"K": 5120, "topk": 256, "stage": 1},
        "desc": "Baseline SAE (stage-1 only, no disentanglement). Pure MSE reconstruction "
                "on frozen SPEAR-XL features. Provides the shared encoder for all stage-2 runs.",
        "category": "baseline",
    },
    "decor_only": {
        "stage1_log": STAGE1_LOG_DIR / "decor_only_29986441.out",
        "stage2_log": STAGE2_LOG_DIR / "main/decor_only_29986442.out",
        "probe_log":  HIST_PROBE_DIR / "probe_decor_only_29986443.out",
        "hparams":    {"K": 5120, "topk": 256, "decor_weight": 1.0,
                       "beta": 0.01, "grl": 0.01},
        "desc": "SAE with full K×K frame-level VICReg decorrelation loss (delta=1.0). "
                "Tests whether reducing feature correlation aids downstream disentanglement. "
                "Key finding: decor reduces off-diagonal correlation ~60% but disentanglement "
                "is worse than weakgrl — redistribution requires GRL, not just decorrelation.",
        "category": "sae_variant",
    },
    "K10240_t128": {
        "stage1_log": STAGE1_LOG_DIR / "stage1_K10240_t128_29970559.out",
        "stage2_log": STAGE2_LOG_DIR / "main/K10240_t128_29981421.out",
        "probe_log":  None,
        "hparams":    {"K": 10240, "topk": 128, "beta": 0.01,
                       "grl": 0.01, "n_routes": 2},
        "desc": "Double-width SAE (K=10240, topk=128, 1.25% sparsity vs 5% for K5120). "
                "Tests whether a larger latent space with sparser activations improves "
                "disentanglement. Probe results pending.",
        "category": "sae_variant",
    },
    "sid1_weakgrl": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "sweep/stage2_sid1_weakgrl_29888468.out",
        "probe_log":  HIST_PROBE_SUCCESS_DIR / "probe_A_29924130.out",
        "hparams":    {"beta": 0.01, "grl": 0.01, "rho": 0.001},
        "desc": "Core disentanglement baseline: GRL on z_L (beta=0.01, grl=0.01). "
                "Best result across all experiments — dominates both axis-1 (z_L→SID=0.104) "
                "and axis-2 (z_P→PR=0.573) simultaneously. No additional tricks.",
        "category": "grl_sweep",
    },
    "sid1_nogrl": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "sweep/stage2_sid1_nogrl_29888469.out",
        "probe_log":  HIST_PROBE_SUCCESS_DIR / "probe_C_29924131.out",
        "hparams":    {"beta": 0.01, "grl": 0.0, "rho": 0.001},
        "desc": "No GRL — routing only, no adversarial speaker removal from z_L. "
                "Shows routing alone cannot separate speaker from phoneme features.",
        "category": "grl_sweep",
    },
    "sid1_delayedgrl": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "sweep/stage2_sid1_delayedgrl_29888470.out",
        "probe_log":  None,
        "hparams":    {"beta": 0.01, "grl": 0.01, "grl_delay": 2000},
        "desc": "GRL with 2000-step warm-up delay. Tests whether letting routing "
                "stabilise before adversarial pressure improves disentanglement.",
        "category": "grl_sweep",
    },
    "sid1_highrho": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "sweep/stage2_sid1_highrho_29888471.out",
        "probe_log":  None,
        "hparams":    {"beta": 0.01, "grl": 0.01, "rho": 0.1},
        "desc": "Higher route entropy regularisation (rho=0.1 vs 0.001). "
                "Tests whether stronger anti-collapse pressure changes routing.",
        "category": "grl_sweep",
    },
    "beta_002": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "beta_sweep/stage2_beta_002_29937686.out",
        "probe_log":  HIST_PROBE_SUCCESS_DIR / "probe_beta_002_29937687.out",
        "hparams":    {"beta": 0.02, "grl": 0.01},
        "desc": "Increased SID adversary weight beta=0.02. Middle ground between "
                "weakgrl (0.01) and beta_003 (0.03).",
        "category": "beta_sweep",
    },
    "beta_003": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "beta_sweep/stage2_beta_003_29937688.out",
        "probe_log":  HIST_PROBE_SUCCESS_DIR / "probe_beta_003_29937689.out",
        "hparams":    {"beta": 0.03, "grl": 0.01},
        "desc": "Strong SID adversary beta=0.03. Better z_P→SID (0.938) but collapses "
                "z_P→PR (0.184). Higher beta aggressively removes speaker from z_L "
                "but phoneme info leaks into P as a side effect.",
        "category": "beta_sweep",
    },
    "dual_grl_03": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "experiments/dual_grl_03_29970545.out",
        "probe_log":  HIST_PROBE_DIR / "probe_dual_grl_03_29970546.out",
        "hparams":    {"beta": 0.03, "grl": 0.01, "grl_phoneme": 0.01},
        "desc": "Dual GRL: speaker adversary on z_L + phoneme adversary on z_P (beta=0.03). "
                "Best z_P→PR after weakgrl (0.425) but catastrophic z_L→SID (0.844). "
                "The phoneme adversary on z_P paradoxically drives speaker info into z_L.",
        "category": "routing_variant",
    },
    "ub": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "experiments/ub_29970549.out",
        "probe_log":  HIST_PROBE_DIR / "probe_ub_29970550.out",
        "hparams":    {"beta": 0.01, "grl": 0.01, "ub_weight": 0.01},
        "desc": "Undecided-bucket bottleneck (ub_weight=0.01). Forces features toward "
                "uncommitted U route. Balanced but mediocre on both axes (0.292, 0.293).",
        "category": "routing_variant",
    },
    "ste": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "experiments/ste_29970551.out",
        "probe_log":  HIST_PROBE_DIR / "probe_ste_29970552.out",
        "hparams":    {"beta": 0.01, "grl": 0.01, "ste_routing": True},
        "desc": "Straight-through estimator routing (beta=0.01, grl=0.01). "
                "Denser gradient coverage through routing masks. Second best z_L→SID (0.186) "
                "but worse than weakgrl on both axes.",
        "category": "routing_variant",
    },
    "ste_ub": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "main/ste_ub_29981425.out",
        "probe_log":  HIST_PROBE_DIR / "probe_ste_ub_29981426.out",
        "hparams":    {"beta": 0.01, "grl": 0.01, "ste_routing": True, "ub_weight": 0.01},
        "desc": "STE routing + undecided bucket. Combines STE's gradient density with "
                "UB pressure valve. Suspicious z_t→SID=0.592 suggests SAE information "
                "loss or routing collapse.",
        "category": "routing_variant",
    },
    "combined": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "main/combined_29981427.out",
        "probe_log":  HIST_PROBE_DIR / "probe_combined_29981428.out",
        "hparams":    {"beta": 0.03, "grl": 0.01, "grl_phoneme": 0.01,
                       "ste_routing": True, "ub_weight": 0.01},
        "desc": "Kitchen-sink combination: beta=0.03, dual-GRL, STE, UB. "
                "Over-constrained — worst overall result. z_L→SID=0.750, z_P→PR=0.160. "
                "Adding all interventions simultaneously causes conflicting gradients.",
        "category": "routing_variant",
    },
    "dual_weak_ub": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "main/dual_weak_ub_29989276.out",
        "probe_log":  HIST_PROBE_DIR / "probe_dual_weak_ub_29989277.out",
        "hparams":    {"beta": 0.01, "grl": 0.01, "grl_phoneme": 0.01, "ub_weight": 0.01},
        "desc": "Dual GRL with weak beta=0.01 + UB. Best z_P→SID (0.942) of all runs — "
                "the phoneme adversary on z_P concentrates speaker info in P. "
                "But z_L→SID degrades (0.330) vs weakgrl (0.104).",
        "category": "routing_variant",
    },
    "no_routing": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "ablations/stage2_no_routing_29909640.out",
        "probe_log":  None,
        "hparams":    {"beta": 0.01, "grl": 0.01, "no_routing": True},
        "desc": "No routing module — all features shared between tasks. GRL still applied. "
                "Ablation: what does the GRL achieve without any route separation?",
        "category": "ablation",
    },
    "fixed_70_30": {
        "stage1_log": None,
        "stage2_log": STAGE2_LOG_DIR / "ablations/stage2_fixed_70_30_29909641.out",
        "probe_log":  None,
        "hparams":    {"beta": 0.01, "grl": 0.01, "fixed_routing": True, "split": 0.7},
        "desc": "Fixed 70/30 routing split (not learned). 70% features go to L, 30% to P. "
                "Ablation: does learned routing help over a fixed random assignment?",
        "category": "ablation",
    },
}

# Probe results from log files
PROBE_RESULTS = {
    "sid1_weakgrl": {"zL_pr": 0.056, "zL_sid": 0.104, "zP_pr": 0.573, "zP_sid": 0.866,
                     "ht_pr": 0.056, "ht_sid": 1.000, "zt_pr": 0.077, "zt_sid": 1.000},
    "sid1_nogrl":   {"zL_pr": 0.055, "zL_sid": 0.784, "zP_pr": 0.394, "zP_sid": 0.926,
                     "ht_pr": 0.054, "ht_sid": 1.000, "zt_pr": 0.079, "zt_sid": 1.000},
    "beta_002":     {"zL_pr": 0.059, "zL_sid": 0.516, "zP_pr": 0.298, "zP_sid": 0.920,
                     "ht_pr": 0.056, "ht_sid": 1.000, "zt_pr": 0.077, "zt_sid": 1.000},
    "beta_003":     {"zL_pr": 0.061, "zL_sid": 0.200, "zP_pr": 0.184, "zP_sid": 0.938,
                     "ht_pr": 0.056, "ht_sid": 1.000, "zt_pr": 0.074, "zt_sid": 1.000},
    "dual_grl_03":  {"zL_pr": 0.059, "zL_sid": 0.844, "zP_pr": 0.425, "zP_sid": 0.936,
                     "ht_pr": 0.059, "ht_sid": 1.000, "zt_pr": 0.074, "zt_sid": 0.998},
    "ub":           {"zL_pr": 0.057, "zL_sid": 0.292, "zP_pr": 0.293, "zP_sid": 0.900,
                     "ht_pr": 0.056, "ht_sid": 1.000, "zt_pr": 0.074, "zt_sid": 1.000},
    "ste":          {"zL_pr": 0.055, "zL_sid": 0.186, "zP_pr": 0.297, "zP_sid": 0.902,
                     "ht_pr": 0.054, "ht_sid": 1.000, "zt_pr": 0.077, "zt_sid": 1.000},
    "decor_only":   {"zL_pr": 0.058, "zL_sid": 0.566, "zP_pr": 0.263, "zP_sid": 0.796,
                     "ht_pr": 0.078, "ht_sid": 1.000, "zt_pr": 0.056, "zt_sid": 0.740},
    "ste_ub":       {"zL_pr": 0.055, "zL_sid": 0.568, "zP_pr": 0.317, "zP_sid": 0.890,
                     "ht_pr": 0.075, "ht_sid": 0.998, "zt_pr": 0.053, "zt_sid": 0.592},
    "combined":     {"zL_pr": 0.063, "zL_sid": 0.750, "zP_pr": 0.160, "zP_sid": 0.956,
                     "ht_pr": 0.074, "ht_sid": 1.000, "zt_pr": 0.058, "zt_sid": 0.836},
    "dual_weak_ub": {"zL_pr": 0.059, "zL_sid": 0.330, "zP_pr": 0.382, "zP_sid": 0.942,
                     "ht_pr": 0.077, "ht_sid": 1.000, "zt_pr": 0.056, "zt_sid": 0.804},
}

# Data-driven radar normalization — actual observed min/max across all experiments
# (want_up=True → higher raw = better; False → lower raw = better)
_pv = {k: [p[k] for p in PROBE_RESULTS.values() if k in p]
       for k in ["zL_sid", "zP_pr", "zP_sid", "zL_pr"]}
RADAR_RANGES = {
    "zL_sid": (min(_pv["zL_sid"]), max(_pv["zL_sid"]), False),
    "zP_pr":  (min(_pv["zP_pr"]),  max(_pv["zP_pr"]),  True),
    "zP_sid": (min(_pv["zP_sid"]), max(_pv["zP_sid"]), True),
    "zL_pr":  (min(_pv["zL_pr"]),  max(_pv["zL_pr"]),  False),
}

CATEGORY_COLORS = {
    "baseline":        "#555555",
    "sae_variant":     "#9467bd",
    "grl_sweep":       "#1f77b4",
    "beta_sweep":      "#ff7f0e",
    "routing_variant": "#2ca02c",
    "ablation":        "#d62728",
}


# ─────────────────────────────────────────── log parsers

def _f(line: str, key: str) -> Optional[float]:
    m = re.search(rf'\b{re.escape(key)}=([0-9eE+\-.]+)', line)
    return float(m.group(1)) if m else None


def parse_stage1(path) -> Dict:
    d = {k: [] for k in ("step", "recon", "decor", "total", "lr",
                          "val_step", "val_recon",
                          "gs", "gr", "gd")}
    if not path or not Path(path).exists():
        return d
    for line in Path(path).read_text().splitlines():
        m = re.match(r'\s+step\s+(\d+)/\d+', line)
        if m:
            d["step"].append(int(m.group(1)))
            for k in ("recon", "decor", "total", "lr"):
                d[k].append(_f(line, k))
            continue
        m = re.search(r'\[val\].*step=(\d+).*val_recon=([0-9eE+\-.]+)', line)
        if m:
            d["val_step"].append(int(m.group(1)))
            d["val_recon"].append(float(m.group(2)))
            continue
        m = re.search(r'\[grad_norm @(\d+)\].*recon=([0-9eE+\-.]+)', line)
        if m:
            d["gs"].append(int(m.group(1)))
            d["gr"].append(float(m.group(2)))
            d["gd"].append(_f(line, "decor"))
    return d


def parse_stage2(path) -> Dict:
    d = {k: [] for k in ("step", "recon", "pr", "sid", "grl", "grl_p", "ub", "lr",
                          "L", "P", "U", "H",
                          "val_step", "val_recon", "val_per", "val_sid")}
    if not path or not Path(path).exists():
        return d
    for line in Path(path).read_text().splitlines():
        m = re.match(r'\s+step\s+(\d+)/\d+', line)
        if m:
            d["step"].append(int(m.group(1)))
            for k in ("recon", "pr", "sid", "grl", "grl_p", "ub", "lr"):
                d[k].append(_f(line, k))
            lpu = re.search(r'L/P/U=(\d+)/(\d+)/(\d+)', line)
            if lpu:
                tot = max(int(lpu.group(1))+int(lpu.group(2))+int(lpu.group(3)), 1)
                d["L"].append(int(lpu.group(1))/tot)
                d["P"].append(int(lpu.group(2))/tot)
                d["U"].append(int(lpu.group(3))/tot)
            else:
                d["L"].append(None); d["P"].append(None); d["U"].append(None)
            h = re.search(r'\bH=([0-9eE+\-.]+)', line)
            d["H"].append(float(h.group(1)) if h else None)
            continue
        m = re.search(r'\[val\].*step=(\d+).*recon=([0-9eE+\-.]+).*PER=([0-9eE+\-.]+).*sid_acc=([0-9eE+\-.]+)', line)
        if m:
            d["val_step"].append(int(m.group(1)))
            d["val_recon"].append(float(m.group(2)))
            d["val_per"].append(float(m.group(3)))
            d["val_sid"].append(float(m.group(4)))
    return d


def _nz(lst):
    return [(s, v) for s, v in zip(*lst) if v is not None] if len(lst) == 2 else []


def _smooth(y, w=7):
    y = np.array([v if v is not None else np.nan for v in y], dtype=float)
    out = np.convolve(np.where(np.isnan(y), 0, y), np.ones(w)/w, "same")
    cnt = np.convolve((~np.isnan(y)).astype(float), np.ones(w)/w, "same")
    return np.where(cnt > 0, out / cnt, np.nan)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {Path(path).relative_to(DIS_DIR)}")


# ─────────────────────────────────────────── per-experiment plots

def plot_training(exp, s1, s2, out):
    has1, has2 = bool(s1["step"]), bool(s2["step"])
    if not has1 and not has2:
        return
    n = (1 if has1 else 0) + (1 if has2 else 0)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 4.5))
    if n == 1: axes = [axes]
    col = 0

    if has1:
        ax = axes[col]; col += 1
        ax.plot(s1["step"], [v or 0 for v in s1["recon"]], color="#1f77b4", lw=1.8, label="recon MSE")
        if any(v for v in s1["decor"] if v):
            ax.plot(s1["step"], [v or 0 for v in s1["decor"]], color="#ff7f0e", lw=1.8, label="decor loss (raw)")
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("Stage-1 Training Losses"); ax.legend(fontsize=8)

    if has2:
        ax = axes[col]
        # Recon excluded — near zero and invisible at this scale
        CLRS = {"pr":"#2ca02c","sid":"#d62728",
                "grl":"#ff7f0e","grl_p":"#9467bd","ub":"#8c564b"}
        LBLS = {"pr":"PR CTC","sid":"SID CE (⚠ adversarially suppressed)",
                "grl":"GRL","grl_p":"GRL phoneme","ub":"UB loss"}
        for k, col_c in CLRS.items():
            pairs = [(s, v) for s, v in zip(s2["step"], s2[k]) if v is not None]
            if not pairs: continue
            ss, vv = zip(*pairs)
            ls = "--" if k == "sid" else "-"
            ax.plot(ss, _smooth(list(vv)), color=col_c, lw=1.5, ls=ls,
                    label=LBLS[k], alpha=0.9)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("Stage-2 Training Losses\n(recon excluded — near zero at this scale)")
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"{exp} — Training Curves", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "training.png")


def plot_validation(exp, s1, s2, out):
    has1v = bool(s1["val_step"])
    has2v = bool(s2["val_step"])
    if not has1v and not has2v:
        return
    n = (1 if has1v else 0) + (1 if has2v else 0)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4.5))
    if n == 1: axes = [axes]
    col = 0

    if has1v:
        ax = axes[col]; col += 1
        ax.plot(s1["val_step"], s1["val_recon"], "o-", color="#1f77b4", lw=2, ms=5)
        ax.set_title("Stage-1 Val Recon MSE"); ax.set_xlabel("Step"); ax.set_ylabel("MSE ↓")

    if has2v:
        ax = axes[col]
        ax2 = ax.twinx()
        ax.plot(s2["val_step"], s2["val_recon"], "o-", color="#1f77b4", lw=2, ms=5, label="val recon (MSE)")
        ax2.plot(s2["val_step"], s2["val_per"],  "s-",  color="#2ca02c", lw=2, ms=5, label="val PER ↓")
        ax2.plot(s2["val_step"], s2["val_sid"],  "^--", color="#d62728", lw=2, ms=5,
                 label="val SID acc ↑\n(⚠ suppressed by GRL)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Recon MSE ↓", color="#1f77b4")
        ax2.set_ylabel("PER / SID acc", color="#555555")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, fontsize=7.5, loc="upper right")
        ax.set_title("Stage-2 Validation\n(val recon + val PER + val SID)")

    fig.suptitle(f"{exp} — Validation Metrics", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "validation.png")


def plot_grad_norms(exp, s1, out):
    if not s1["gs"]:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(s1["gs"], s1["gr"], "o-", color="#1f77b4", lw=2, ms=6, label="|g| recon")
    gd_pairs = [(s, v) for s, v in zip(s1["gs"], s1["gd"]) if v is not None]
    if gd_pairs:
        gd_s, gd_v = zip(*gd_pairs)
        ax.semilogy(gd_s, gd_v, "s-", color="#ff7f0e", lw=2, ms=6, label="|g| decor (weighted)")
        med = float(np.median(gd_v))
        idx = int(np.argmax(gd_v))
        if gd_v[idx] > 5 * med:
            ax.annotate(f"spike ×{gd_v[idx]/med:.0f}",
                        xy=(gd_s[idx], gd_v[idx]),
                        xytext=(gd_s[idx]-700, gd_v[idx]*0.6),
                        arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
                        fontsize=8, color="#ff7f0e")
    ax.set_xlabel("Step"); ax.set_ylabel("|gradient| log scale")
    ax.legend(fontsize=9)
    ax.set_title(f"{exp} — Stage-1 Gradient Norms", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "grad_norms.png")


def plot_routing(exp, s2, out):
    pairs_L = [(s, v) for s, v in zip(s2["step"], s2["L"]) if v is not None]
    if not pairs_L:
        return
    ss, L = zip(*pairs_L)
    P = [v for v in s2["P"] if v is not None]
    U = [v for v in s2["U"] if v is not None]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(ss, _smooth(list(L)), _smooth(list(P)), _smooth(list(U)),
                 labels=["L (linguistic)", "P (paralinguistic)", "U (undecided)"],
                 colors=["#2196F3", "#FF9800", "#9E9E9E"], alpha=0.8)
    ax.axhline(1/3, ls="--", color="black", alpha=0.5, lw=1.5)
    ax.text(ss[-1]*1.005, 1/3, "uniform\n1/3", fontsize=7, va="center", color="gray")
    ax.set_ylabel("Feature fraction"); ax.set_ylim(0, 1)
    ax.set_xlabel("Step")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(f"{exp} — Routing Feature Fractions over Training", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "routing.png")


def plot_probe(exp, probe, out):
    if not probe:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, task, keys, labels, want_down in [
        (axes[0], "Phoneme Recognition (PER)",
         ["zL_pr", "zP_pr", "ht_pr", "zt_pr"],
         ["z_L→PR", "z_P→PR", "h_t→PR (ref)", "z_t→PR (ref)"],
         [True, False, None, None]),
        (axes[1], "Speaker ID (Accuracy)",
         ["zL_sid", "zP_sid", "ht_sid", "zt_sid"],
         ["z_L→SID", "z_P→SID", "h_t→SID (ref)", "z_t→SID (ref)"],
         [True, False, None, None]),
    ]:
        vals   = [probe.get(k, 0) for k in keys]
        colors = ["#2196F3", "#FF9800", "#aaaaaa", "#777777"]
        bars   = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white", lw=1.5)
        ax.set_ylim(0, 1.12)
        ax.set_title(task)
        ylabel = "PER (↓ z_L,  ↑ z_P)" if "PR" in task else "Accuracy (↓ z_L,  ↑ z_P)"
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=15)
        for bar, val, wd in zip(bars, vals, want_down):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
            if wd is True:
                ax.text(bar.get_x() + bar.get_width()/2, 0.02, "want ↓",
                        ha="center", fontsize=7, color="navy")
            elif wd is False:
                ax.text(bar.get_x() + bar.get_width()/2, 0.02, "want ↑",
                        ha="center", fontsize=7, color="darkgreen")

    fig.suptitle(f"{exp} — Probe Results", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "probe.png")


def plot_radar(exp, probe, out):
    if not probe:
        return
    keys   = ["zL_sid", "zP_pr", "zP_sid", "zL_pr"]
    labels = ["z_L→SID\n(↓ lower=better)", "z_P→PR\n(↑ higher=better)",
              "z_P→SID\n(↑ higher=better)", "z_L→PR\n(↓ lower=better)"]

    raw_vals = [probe.get(k, 0) for k in keys]
    closed   = raw_vals + raw_vals[:1]
    N        = len(keys)
    angles   = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig = plt.figure(figsize=(6.5, 7.5))
    # Place polar axes in the centre of the figure, leaving explicit room
    # at top (for title) and bottom (for bottom axis label)
    ax = fig.add_axes([0.15, 0.14, 0.70, 0.62], polar=True)

    ax.fill(angles, closed, alpha=0.2, color="#2196F3")
    ax.plot(angles, closed, "o-", color="#2196F3", lw=2, ms=7)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0)

    # Axis labels just outside the circle — label_r=1.28 stays within the axes box
    label_r = 1.28
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, label_r, label, ha="center", va="center",
                fontsize=8.5, linespacing=1.4)

    # Raw value annotations offset from each vertex
    for angle, val in zip(angles[:-1], raw_vals):
        offset = 0.13 if val < 0.85 else -0.13
        ax.text(angle, val + offset, f"{val:.3f}",
                ha="center", va="center", fontsize=10,
                fontweight="bold", color="#1565C0")

    # Title and subtitle at fixed figure positions — well above the polar axes
    fig.text(0.5, 0.96, f"{exp} — Disentanglement Radar",
             ha="center", va="top", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.92, "further from centre = higher raw value",
             ha="center", va="top", fontsize=8.5, color="#555555")

    _save(fig, out / "radar.png")


def plot_decor_analysis(s1, out):
    """Decorr-specific: loss reduction curve + gradient spike analysis."""
    if not any(v for v in s1["decor"] if v):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Loss curves
    ax = axes[0]
    recon = [v or 0 for v in s1["recon"]]
    decor = [v or 0 for v in s1["decor"]]
    ax.plot(s1["step"], recon, color="#1f77b4", lw=2, label="recon MSE")
    ax2 = ax.twinx()
    ax2.plot(s1["step"], decor, color="#ff7f0e", lw=2, label="decor loss (raw)")
    ax2.fill_between(s1["step"], decor, alpha=0.1, color="#ff7f0e")
    ax.set_xlabel("Step")
    ax.set_ylabel("Recon MSE", color="#1f77b4")
    ax2.set_ylabel("Decor loss (MEAN off-diag corr²)", color="#ff7f0e")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=8)
    ax.set_title("Decorrelation Loss Reduction over Training\n"
                 f"Initial: {decor[0]:.4f}  →  Final: {decor[-1]:.4f}  "
                 f"({100*(1-decor[-1]/decor[0]):.0f}% reduction)")

    # Gradient norms
    if s1["gs"]:
        ax = axes[1]
        ax.semilogy(s1["gs"], s1["gr"], "o-", color="#1f77b4", lw=2, ms=7, label="|g| recon")
        gd_pairs = [(s, v) for s, v in zip(s1["gs"], s1["gd"]) if v is not None]
        if gd_pairs:
            gds, gdv = zip(*gd_pairs)
            ax.semilogy(gds, gdv, "s-", color="#ff7f0e", lw=2, ms=7, label="|g| decor")
            med = float(np.median(gdv))
            idx = int(np.argmax(gdv))
            if gdv[idx] > 5 * med:
                ax.annotate(f"instability spike\n×{gdv[idx]/med:.0f} median\n"
                            f"(random frame subsample)",
                            xy=(gds[idx], gdv[idx]),
                            xytext=(gds[idx]-1000, gdv[idx]*0.3),
                            arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
                            fontsize=8, color="#ff7f0e")
            # ratio annotation
            for s, r, d in zip(s1["gs"], s1["gr"], s1["gd"]):
                if d:
                    ax.text(s, d * 1.3, f"{d/r:.1f}×", fontsize=6.5,
                            ha="center", color="#ff7f0e", alpha=0.7)
        ax.set_xlabel("Step")
        ax.set_ylabel("|gradient| log scale")
        ax.legend(fontsize=9)
        ax.set_title("Gradient Norms: decor vs recon (log scale)\n"
                     "Ratio labels show decor/recon at each checkpoint")

    fig.suptitle("decor_only — Decorrelation Analysis", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "decor_analysis.png")


def write_summary(exp, cfg, s1, s2, probe, out):
    lines = [f"{'='*60}", f"  EXPERIMENT: {exp}", f"{'='*60}",
             f"\nDESCRIPTION:\n  {cfg['desc']}",
             f"\nCATEGORY: {cfg['category']}",
             f"\nHYPERPARAMETERS:"]
    for k, v in cfg["hparams"].items():
        lines.append(f"  {k:22s} = {v}")

    if s1["step"]:
        r = [v for v in s1["recon"] if v]
        dec = [v for v in s1["decor"] if v]
        lines += ["\nSTAGE-1 TRAINING:",
                  f"  recon : {r[0]:.5f} → {r[-1]:.5f}"]
        if dec:
            lines.append(f"  decor : {dec[0]:.5f} → {min(dec):.5f}  "
                         f"(peak={max(dec):.5f}, "
                         f"reduction={100*(1-min(dec)/dec[0]):.0f}%)")
        if s1["val_step"]:
            lines.append(f"  best val_recon: {min(s1['val_recon']):.5f} "
                         f"@ step {s1['val_step'][int(np.argmin(s1['val_recon']))]}")
        if s1["gs"]:
            lines.append("\nGRADIENT NORMS (stage-1):")
            for step, gr, gd in zip(s1["gs"], s1["gr"], s1["gd"]):
                gd_str = f"  decor={gd:.5f}  ratio={gd/gr:.2f}×" if gd else ""
                lines.append(f"  @{step:5d}: recon={gr:.5f}{gd_str}")

    if s2["step"]:
        r = [v for v in s2["recon"] if v]
        lines += ["\nSTAGE-2 TRAINING:",
                  f"  recon : {r[0]:.5f} → {r[-1]:.5f}"]
        if s2["val_step"]:
            lines += [f"  best val_recon : {min(s2['val_recon']):.5f}",
                      f"  best val_PER   : {min(s2['val_per']):.4f}"]
        H = [v for v in s2["H"] if v]
        if H:
            lines += [f"  routing entropy (final): {H[-1]:.4f}  "
                      f"[max=ln(3)={np.log(3):.4f}]",
                      f"  NOTE: entropy ≈ ln(3) throughout all steps — "
                      f"routing logits barely trained (|logit|~0.003)."]

    if probe:
        lines += ["\nPROBE RESULTS:  (* = primary disentanglement axes)",
                  f"  {'Source':<22s}  {'Value':>8s}  Note"]
        items = [
            ("z_L → PR  (PER)",  "zL_pr",  "↓ lower = L preserves phonemes"),
            ("z_L → SID (acc) *","zL_sid", "↓ AXIS 1: speaker removed from L"),
            ("z_P → PR  (PER) *","zP_pr",  "↑ AXIS 2: phonemes removed from P"),
            ("z_P → SID (acc)", "zP_sid",  "↑ speaker concentrated in P"),
            ("h_t → PR  (ref)",  "ht_pr",  "frozen SPEAR reference"),
            ("h_t → SID (ref)",  "ht_sid", "frozen SPEAR reference"),
            ("z_t → PR  (ref)",  "zt_pr",  "full SAE latent reference"),
            ("z_t → SID (ref)",  "zt_sid", "full SAE latent reference"),
        ]
        for lbl, k, note in items:
            v = probe.get(k, float("nan"))
            lines.append(f"  {lbl:<22s}  {v:>8.3f}  {note}")

    lines.append(f"\n{'='*60}")
    p = out / "summary.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines))
    print(f"  → {p.relative_to(DIS_DIR)}")


# ─────────────────────────────────────────── comparison plots

def plot_scatter_disentanglement(probe_all, out):
    fig, ax = plt.subplots(figsize=(10, 8))
    for exp, p in probe_all.items():
        cat = EXPERIMENTS[exp]["category"]
        col = CATEGORY_COLORS.get(cat, "#333333")
        x, y = p["zL_sid"], p["zP_pr"]
        ax.scatter(x, y, color=col, s=130, zorder=5, edgecolors="white", lw=1.5)
        ax.annotate(exp, (x, y), textcoords="offset points", xytext=(5, 4),
                    fontsize=8.5, color=col, fontweight="bold")
    ax.fill_between([0, 0.25], [0.45, 0.45], [0.65, 0.65],
                    alpha=0.07, color="green")
    ax.text(0.02, 0.6, "Ideal region", fontsize=8, color="green", alpha=0.8)
    ax.axvline(0.5, ls=":", color="gray", alpha=0.4, lw=1)
    ax.axhline(0.3, ls=":", color="gray", alpha=0.4, lw=1)
    ax.set_xlabel("z_L → SID accuracy  (↓ better: less speaker info in L)", fontsize=11)
    ax.set_ylabel("z_P → PR   PER  (↑ better: less phoneme info in P)", fontsize=11)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 0.72)
    ax.set_title("Primary Disentanglement Trade-off\n"
                 "Axis 1: speaker removal from z_L    Axis 2: phoneme removal from z_P",
                 fontweight="bold", fontsize=12)
    handles = [mpatches.Patch(color=c, label=cat.replace("_", " "))
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, out / "scatter_disentanglement.png")


def plot_scatter_speaker(probe_all, out):
    fig, ax = plt.subplots(figsize=(10, 7))
    for exp, p in probe_all.items():
        cat = EXPERIMENTS[exp]["category"]
        col = CATEGORY_COLORS.get(cat, "#333333")
        x, y = p["zL_sid"], p["zP_sid"]
        ax.scatter(x, y, color=col, s=130, zorder=5, edgecolors="white", lw=1.5)
        ax.annotate(exp, (x, y), textcoords="offset points", xytext=(5, 4),
                    fontsize=8.5, color=col, fontweight="bold")
    ax.fill_between([0, 0.2], [0.93, 0.93], [1.0, 1.0], alpha=0.07, color="green")
    ax.text(0.02, 0.96, "Ideal", fontsize=8, color="green", alpha=0.8)
    ax.set_xlabel("z_L → SID accuracy  (↓ better: speaker rejected from L)", fontsize=11)
    ax.set_ylabel("z_P → SID accuracy  (↑ better: speaker concentrated in P)", fontsize=11)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.5, 1.05)
    ax.set_title("Speaker Information Distribution\n"
                 "Do L and P routes successfully push/pull speaker identity?",
                 fontweight="bold", fontsize=12)
    handles = [mpatches.Patch(color=c, label=cat.replace("_", " "))
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    _save(fig, out / "scatter_speaker.png")


def plot_probe_heatmap(probe_all, out):
    exps  = list(probe_all.keys())
    cols  = ["zL_sid", "zP_pr", "zP_sid", "zL_pr"]
    clbls = ["z_L→SID\n(↓ lower=better)", "z_P→PR\n(↑ higher=better)",
             "z_P→SID\n(↑ higher=better)", "z_L→PR\n(↓ lower=better)"]

    # Absolute normalization ranges — colour reflects true performance, not relative rank.
    # (lo, hi, want_up): normalised = (raw-lo)/(hi-lo) if up, else 1-(raw-lo)/(hi-lo)
    # z_L→PR uses 0–0.15 so all observed values (0.055–0.063) map to solid green (~0.6+)
    ABS_RANGES = {
        "zL_sid": (0.0,  1.0,  False),
        "zP_pr":  (0.0,  0.70, True),
        "zP_sid": (0.50, 1.0,  True),
        "zL_pr":  (0.0,  0.15, False),
    }

    mat   = np.array([[probe_all[e].get(c, np.nan) for c in cols] for e in exps])
    mat_n = np.full_like(mat, np.nan)
    for j, c in enumerate(cols):
        lo, hi, up = ABS_RANGES[c]
        raw = (mat[:, j] - lo) / max(hi - lo, 1e-6)
        mat_n[:, j] = np.clip(raw if up else 1 - raw, 0, 1)

    fig, ax = plt.subplots(figsize=(10, max(5, len(exps) * 0.65 + 2)))
    im = ax.imshow(mat_n, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(clbls, fontsize=10)
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels(exps, fontsize=9)

    for i in range(len(exps)):
        for j in range(len(cols)):
            v = mat[i, j]
            if not np.isnan(v):
                score = mat_n[i, j]
                txt_col = "black" if 0.25 < score < 0.80 else "white"
                ax.text(j, i, f"{v*100:.1f}%",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color=txt_col)

    plt.colorbar(im, ax=ax, label="Score  (1 = best, 0 = worst on absolute scale)")
    ax.set_title("Probe Results Heatmap  —  all values shown as percentages\n"
                 "green = better   |   z_L→PR normalised on [0, 15%] so ~6% = solid green",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "probe_heatmap.png")


def plot_val_recon_groups(all_s2, out):
    groups = {
        "GRL sweep":        ["sid1_weakgrl","sid1_nogrl","sid1_delayedgrl","sid1_highrho"],
        "β scaling":        ["sid1_weakgrl","beta_002","beta_003"],
        "Routing variants": ["sid1_weakgrl","dual_grl_03","ub","ste","ste_ub",
                             "combined","dual_weak_ub"],
        "SAE variants":     ["baseline","decor_only","K10240_t128"],
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (gname, exps) in zip(axes.flatten(), groups.items()):
        for exp in exps:
            s2 = all_s2.get(exp, {})
            if not s2.get("val_step"): continue
            col = CATEGORY_COLORS.get(EXPERIMENTS.get(exp,{}).get("category",""), "#333")
            ls  = "-" if exp == "sid1_weakgrl" else "--"
            lw  = 2.5 if exp == "sid1_weakgrl" else 1.5
            ax.plot(s2["val_step"], s2["val_recon"], ls=ls, lw=lw,
                    label=exp, color=col, alpha=0.9)
        ax.set_title(f"Val Recon — {gname}")
        ax.set_xlabel("Step"); ax.set_ylabel("val_recon MSE ↓")
        ax.legend(fontsize=7)
    fig.suptitle("Stage-2 Validation Reconstruction Convergence", fontweight="bold")
    fig.tight_layout()
    _save(fig, out / "val_recon_groups.png")


def plot_routing_entropy_all(all_s2, out):
    fig, ax = plt.subplots(figsize=(12, 5))
    ln3 = np.log(3)
    for exp, s2 in all_s2.items():
        H = [(s, v) for s, v in zip(s2.get("step",[]), s2.get("H",[])) if v is not None]
        if not H: continue
        hs, hv = zip(*H)
        col = CATEGORY_COLORS.get(EXPERIMENTS.get(exp,{}).get("category",""), "#333")
        ax.plot(hs, hv, lw=1.5, label=exp, color=col, alpha=0.75)
    ax.axhline(ln3, ls="--", color="black", lw=2, alpha=0.7,
               label=f"Max entropy = ln(3) = {ln3:.3f}")
    ax.set_xlabel("Step"); ax.set_ylabel("Routing entropy (nats)")
    ax.set_title("Routing Entropy — All Runs\n"
                 "All runs stay near ln(3) throughout: routing logits barely trained",
                 fontweight="bold")
    ax.legend(fontsize=7, ncol=2); ax.set_ylim(0.9, 1.2)
    fig.tight_layout()
    _save(fig, out / "routing_entropy_all.png")


def plot_reconstruction_quality(all_s2, out):
    data = {e: min(s2["val_recon"]) for e, s2 in all_s2.items() if s2.get("val_recon")}
    if not data: return
    exps = sorted(data, key=lambda e: data[e])
    vals = [data[e] for e in exps]
    cols = [CATEGORY_COLORS.get(EXPERIMENTS.get(e,{}).get("category",""), "#333") for e in exps]
    fig, ax = plt.subplots(figsize=(max(8, len(exps)*0.8+2), 5))
    bars = ax.bar(exps, vals, color=cols, alpha=0.85, edgecolor="white", lw=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8, rotation=45)
    ax.set_ylabel("Best val_recon MSE ↓"); ax.tick_params(axis="x", rotation=40)
    ax.set_title("Final SAE Reconstruction Quality per Run", fontweight="bold")
    handles = [mpatches.Patch(color=c, label=cat.replace("_"," "))
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, out / "reconstruction_quality.png")


# ─────────────────────────────────────────── single-image overview

def plot_overview(exp, cfg, s1, s2, probe, out):
    """One figure, four panels — everything about one experiment."""
    has_s2    = bool(s2["step"])
    has_probe = bool(probe)
    cat       = cfg["category"]
    col       = CATEGORY_COLORS.get(cat, "#333333")

    fig = plt.figure(figsize=(14, 9))

    # ── header strip ──────────────────────────────────────────────────────────
    fig.patch.set_facecolor("white")
    header_ax = fig.add_axes([0, 0.91, 1, 0.09])
    header_ax.set_facecolor(col)
    header_ax.axis("off")
    header_ax.text(0.02, 0.72, exp,
                   color="white", fontsize=16, fontweight="bold",
                   transform=header_ax.transAxes, va="top")
    header_ax.text(0.02, 0.30, f"  {cfg['hparams']}",
                   color="white", fontsize=9, alpha=0.9,
                   transform=header_ax.transAxes, va="top")
    # description — truncated to fit
    import textwrap as tw
    desc = tw.shorten(cfg["desc"], width=140)
    header_ax.text(0.02, 0.02, desc,
                   color="white", fontsize=8, alpha=0.85,
                   transform=header_ax.transAxes, va="bottom")

    # ── four quadrants — enough vertical gap so titles don't overlap ──────────
    ax_tl = fig.add_axes([0.03, 0.53, 0.44, 0.36])
    ax_tr = fig.add_axes([0.53, 0.53, 0.44, 0.36])
    ax_bl = fig.add_axes([0.03, 0.05, 0.44, 0.41])
    ax_br = fig.add_axes([0.53, 0.05, 0.44, 0.41])

    # ── TL: training ──────────────────────────────────────────────────────────
    if has_s2:
        CLRS = {"pr":"#2ca02c","sid":"#d62728","grl":"#ff7f0e",
                "grl_p":"#9467bd","ub":"#8c564b"}
        LBLS = {"pr":"PR CTC","sid":"SID CE ⚠","grl":"GRL",
                "grl_p":"GRL phoneme","ub":"UB"}
        ax_tl2 = ax_tl.twinx()  # secondary axis for recon
        plotted = False
        for k, c in CLRS.items():
            pairs = [(s, v) for s, v in zip(s2["step"], s2[k]) if v is not None]
            if not pairs: continue
            ss, vv = zip(*pairs)
            ls = "--" if k == "sid" else "-"
            ax_tl.plot(ss, _smooth(list(vv)), color=c, lw=1.8, ls=ls,
                       label=LBLS[k], alpha=0.9)
            plotted = True
        # recon on secondary axis (much smaller scale)
        recon_pairs = [(s, v) for s, v in zip(s2["step"], s2["recon"]) if v is not None]
        if recon_pairs:
            rs, rv = zip(*recon_pairs)
            ax_tl2.plot(rs, _smooth(list(rv)), color="#1f77b4", lw=1.5, ls=":",
                        label="recon (×1)", alpha=0.7)
            ax_tl2.set_ylabel("Recon MSE", color="#1f77b4", fontsize=7)
            ax_tl2.tick_params(axis="y", labelsize=7, colors="#1f77b4")
        if plotted:
            h1, l1 = ax_tl.get_legend_handles_labels()
            h2, l2 = ax_tl2.get_legend_handles_labels()
            ax_tl.legend(h1+h2, l1+l2, fontsize=7, ncol=2)
        ax_tl.set_xlabel("Step", fontsize=8); ax_tl.set_ylabel("Task Loss", fontsize=8)
        ax_tl.set_title("Stage-2 Training  (dotted = recon, right axis)",
                         fontsize=9)
    elif s1["step"]:
        has_decor = any(v for v in s1["decor"] if v)
        ax_tl.plot(s1["step"], [v or 0 for v in s1["recon"]],
                   color="#1f77b4", lw=2, label="recon MSE")
        if has_decor:
            ax_tl2 = ax_tl.twinx()
            # Show weighted decor (decor_weight × raw_decor)
            cfg_dw = cfg.get("hparams", "")
            dw = 1.0  # default weight shown in config
            weighted = [v * dw if v else 0 for v in s1["decor"]]
            ax_tl2.plot(s1["step"], weighted, color="#ff7f0e", lw=2,
                        label=f"decor (weighted ×{dw})", alpha=0.85)
            ax_tl2.set_ylabel("Decor loss (weighted)", color="#ff7f0e", fontsize=7)
            ax_tl2.tick_params(axis="y", labelsize=7, colors="#ff7f0e")
            h1, l1 = ax_tl.get_legend_handles_labels()
            h2, l2 = ax_tl2.get_legend_handles_labels()
            ax_tl.legend(h1+h2, l1+l2, fontsize=8)
        else:
            ax_tl.legend(fontsize=8)
        ax_tl.set_xlabel("Step", fontsize=8); ax_tl.set_ylabel("Recon MSE", fontsize=8)
        ax_tl.set_title("Stage-1 Training Losses", fontsize=9)
    else:
        ax_tl.axis("off")
        ax_tl.text(0.5, 0.5, "No training data", ha="center", va="center",
                   color="gray", transform=ax_tl.transAxes)

    # ── TR: validation ────────────────────────────────────────────────────────
    if s2["val_step"]:
        ax_tr2 = ax_tr.twinx()
        ax_tr.plot(s2["val_step"], s2["val_recon"], "o-", color="#1f77b4",
                   lw=2, ms=5, label="val recon")
        ax_tr2.plot(s2["val_step"], s2["val_per"],  "s-",  color="#2ca02c",
                    lw=2, ms=5, label="val PER ↓")
        ax_tr2.plot(s2["val_step"], s2["val_sid"],  "^--", color="#d62728",
                    lw=2, ms=5, label="val SID ⚠")
        ax_tr.set_ylabel("Recon MSE ↓", color="#1f77b4", fontsize=8)
        ax_tr2.set_ylabel("PER / SID", color="#555", fontsize=8)
        ax_tr.set_xlabel("Step", fontsize=8)
        h1, l1 = ax_tr.get_legend_handles_labels()
        h2, l2 = ax_tr2.get_legend_handles_labels()
        ax_tr.legend(h1+h2, l1+l2, fontsize=7, loc="upper right")
        ax_tr.set_title("Validation Metrics", fontsize=9)
    elif s1["val_step"]:
        ax_tr.plot(s1["val_step"], s1["val_recon"], "o-", color="#1f77b4", lw=2, ms=5)
        ax_tr.set_xlabel("Step", fontsize=8); ax_tr.set_ylabel("Val recon MSE ↓", fontsize=8)
        ax_tr.set_title("Stage-1 Validation", fontsize=9)
    else:
        ax_tr.axis("off")
        ax_tr.text(0.5, 0.5, "No validation data", ha="center", va="center",
                   color="gray", transform=ax_tr.transAxes)

    # ── BL: routing fractions ─────────────────────────────────────────────────
    pairs_L = [(s, v) for s, v in zip(s2["step"], s2["L"]) if v is not None]
    if pairs_L:
        ss, L = zip(*pairs_L)
        P = [v for v in s2["P"] if v is not None]
        U = [v for v in s2["U"] if v is not None]
        ax_bl.stackplot(ss, _smooth(list(L)), _smooth(list(P)), _smooth(list(U)),
                        labels=["L", "P", "U"],
                        colors=["#2196F3","#FF9800","#9E9E9E"], alpha=0.8)
        ax_bl.axhline(1/3, ls="--", color="black", alpha=0.4, lw=1)
        ax_bl.set_ylim(0, 1); ax_bl.set_xlabel("Step", fontsize=8)
        ax_bl.set_ylabel("Feature fraction", fontsize=8)
        ax_bl.legend(fontsize=8, loc="upper right")
        ax_bl.set_title("Routing Fractions  (dashed = uniform 1/3)", fontsize=9)
    else:
        ax_bl.axis("off")
        ax_bl.text(0.5, 0.5, "No routing data", ha="center", va="center",
                   color="gray", transform=ax_bl.transAxes)

    # ── BR: probe bars + radar combined ──────────────────────────────────────
    if has_probe:
        # Horizontal bar chart — cleaner for 4 metrics
        metrics  = ["z_L→SID\n(↓ better)", "z_P→PR\n(↑ better)",
                     "z_P→SID\n(↑ better)", "z_L→PR\n(↓ better)"]
        keys     = ["zL_sid", "zP_pr", "zP_sid", "zL_pr"]
        vals     = [probe.get(k, 0) * 100 for k in keys]
        colors   = ["#2196F3", "#FF9800", "#2ca02c", "#9467bd"]
        y_pos    = range(len(metrics))

        bars = ax_br.barh(list(y_pos), vals, color=colors, alpha=0.85,
                          edgecolor="white", lw=1.2, height=0.55)
        ax_br.set_yticks(list(y_pos))
        ax_br.set_yticklabels(metrics, fontsize=8.5)
        ax_br.set_xlim(0, 100)
        ax_br.set_xlabel("% (all metrics in [0, 100])", fontsize=8)
        ax_br.axvline(50, ls=":", color="gray", alpha=0.4, lw=1)

        # Direction arrows + value labels
        good_dir = [False, True, True, False]  # True = higher is better
        for bar, val, gd in zip(bars, vals, good_dir):
            x_lbl = val + 1.5
            ax_br.text(x_lbl, bar.get_y() + bar.get_height()/2,
                       f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
            arrow = "← want low" if not gd else "→ want high"
            ax_br.text(96, bar.get_y() + bar.get_height()/2,
                       arrow, va="center", ha="right", fontsize=7, color="gray")

        # z_t→SID reference line
        zt = probe.get("zt_sid", 1.0) * 100
        ax_br.set_title(f"Probe Results  (z_t→SID={zt:.1f}% — SAE info check)",
                         fontsize=9)
        if zt < 80:
            ax_br.set_title(f"Probe Results  (⚠ z_t→SID={zt:.1f}% — SAE info loss!)",
                             fontsize=9, color="#cc0000")
    else:
        ax_br.axis("off")
        ax_br.text(0.5, 0.5, "No probe data available\nfor this experiment.",
                   ha="center", va="center", fontsize=10, color="gray",
                   transform=ax_br.transAxes)

    fig.suptitle("", y=0)  # prevent suptitle overlap with header
    _save(fig, out / "overview.png")


# ─────────────────────────────────────────── main

def main():
    print("=== Disentanglement Analysis ===\n")
    all_s1, all_s2 = {}, {}

    for exp, cfg in EXPERIMENTS.items():
        print(f"\n── {exp}")
        od = OUT_DIR / exp
        od.mkdir(parents=True, exist_ok=True)

        s1 = parse_stage1(cfg.get("stage1_log"))
        s2 = parse_stage2(cfg.get("stage2_log"))
        probe = PROBE_RESULTS.get(exp, {})
        all_s1[exp] = s1; all_s2[exp] = s2

        plot_overview(exp, cfg, s1, s2, probe, od)
        plot_training(exp, s1, s2, od)
        plot_validation(exp, s1, s2, od)
        plot_grad_norms(exp, s1, od)
        plot_routing(exp, s2, od)
        plot_probe(exp, probe, od)
        plot_radar(exp, probe, od)
        write_summary(exp, cfg, s1, s2, probe, od)
        if exp == "decor_only":
            plot_decor_analysis(s1, od)

    print("\n── comparison")
    comp = OUT_DIR / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    probe_all = {e: p for e, p in PROBE_RESULTS.items() if e in EXPERIMENTS}
    plot_scatter_disentanglement(probe_all, comp)
    plot_scatter_speaker(probe_all, comp)
    plot_probe_heatmap(probe_all, comp)
    plot_val_recon_groups(all_s2, comp)
    plot_routing_entropy_all(all_s2, comp)
    plot_reconstruction_quality(all_s2, comp)

    print(f"\n=== Done. Output in analysis/ ===")


if __name__ == "__main__":
    main()
