#!/usr/bin/env python3
"""Generate a structured PDF report from pre-generated analysis PNGs.

Structure
---------
  Cover page
  1. Stage-1 Baseline SAE
  2. Stage-2 Experiments (per-experiment pages, grouped by category)
  3. Cross-experiment Comparison
"""

from __future__ import annotations
from pathlib import Path
import textwrap
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

DIS_DIR  = Path(__file__).resolve().parents[2]
ANA_DIR  = DIS_DIR / "analysis"
OUT_PDF  = DIS_DIR / "analysis" / "disentanglement_report.pdf"

# ── colour scheme ──────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "baseline":        "#555555",
    "sae_variant":     "#9467bd",
    "grl_sweep":       "#1f77b4",
    "beta_sweep":      "#ff7f0e",
    "routing_variant": "#2ca02c",
    "ablation":        "#d62728",
}

EXPERIMENTS = {
    "baseline": {
        "category": "baseline",
        "hparams": "K=5120  topk=256  stage=1 only",
        "desc": "Pure MSE reconstruction on frozen SPEAR-XL features (D=1280). "
                "Provides the shared encoder for all stage-2 runs. "
                "Achieves val_recon=0.0049 (R²≈0.80 given zero-predictor MSE≈0.024).",
    },
    "decor_only": {
        "category": "sae_variant",
        "hparams": "K=5120  topk=256  decor_weight=1.0  β=0.01  grl=0.01",
        "desc": "SAE trained with full K×K frame-level VICReg decorrelation loss. "
                "Reduces off-diagonal correlation ~60% but disentanglement is worse "
                "than weakgrl — decorrelation alone does not redistribute features "
                "without GRL guidance. z_t→SID drops to 0.740.",
    },
    "K10240_t128": {
        "category": "sae_variant",
        "hparams": "K=10240  topk=128  n_routes=2  β=0.01  grl=0.01",
        "desc": "Double-width SAE (1.25% sparsity vs 5% for K5120). "
                "Stage-1 achieves val_recon=0.0054. Probe results pending.",
    },
    "sid1_weakgrl": {
        "category": "grl_sweep",
        "hparams": "β=0.01  grl=0.01  ρ=0.001",
        "desc": "★ Best overall result. Simple GRL on z_L with mild weights. "
                "Dominates both primary axes simultaneously: z_L→SID=0.104 (best), "
                "z_P→PR=0.573 (best). No extra tricks required.",
    },
    "sid1_nogrl": {
        "category": "grl_sweep",
        "hparams": "β=0.01  grl=0.00  ρ=0.001",
        "desc": "No GRL — routing only. Without adversarial pressure, z_L retains "
                "substantial speaker information (z_L→SID=0.784). Shows routing "
                "alone cannot disentangle without explicit adversarial guidance.",
    },
    "sid1_delayedgrl": {
        "category": "grl_sweep",
        "hparams": "β=0.01  grl=0.01  grl_delay=2000 steps",
        "desc": "GRL applied after 2000-step warm-up. Intended to let routing "
                "stabilise first. No probe data available.",
    },
    "sid1_highrho": {
        "category": "grl_sweep",
        "hparams": "β=0.01  grl=0.01  ρ=0.1",
        "desc": "Stronger route entropy regularisation (ρ=0.1 vs 0.001). "
                "Tests anti-collapse pressure. No probe data available.",
    },
    "beta_002": {
        "category": "beta_sweep",
        "hparams": "β=0.02  grl=0.01",
        "desc": "Moderate SID adversary weight β=0.02. Mid-point between "
                "weakgrl (0.01) and beta_003 (0.03). z_L→SID=0.516, z_P→PR=0.298.",
    },
    "beta_003": {
        "category": "beta_sweep",
        "hparams": "β=0.03  grl=0.01",
        "desc": "Strong SID adversary β=0.03. Better z_P→SID (0.938) but "
                "z_P→PR collapses to 0.184 — phonemes leak into P as β increases.",
    },
    "dual_grl_03": {
        "category": "routing_variant",
        "hparams": "β=0.03  grl=0.01  grl_phoneme=0.01",
        "desc": "Dual GRL: speaker adversary on z_L + phoneme adversary on z_P. "
                "Best z_P→PR after weakgrl (0.425) but catastrophic z_L→SID=0.844 "
                "— phoneme adversary paradoxically drives speaker info into z_L.",
    },
    "ub": {
        "category": "routing_variant",
        "hparams": "β=0.01  grl=0.01  ub_weight=0.01",
        "desc": "Undecided-bucket bottleneck. Forces features toward uncommitted U "
                "route. Balanced but mediocre: z_L→SID=0.292, z_P→PR=0.293.",
    },
    "ste": {
        "category": "routing_variant",
        "hparams": "β=0.01  grl=0.01  ste_routing=True",
        "desc": "Straight-through estimator routing. Dense gradient through routing "
                "masks. Second best z_L→SID (0.186) but worse than weakgrl overall.",
    },
    "ste_ub": {
        "category": "routing_variant",
        "hparams": "β=0.01  grl=0.01  ste_routing=True  ub_weight=0.01",
        "desc": "STE + undecided bucket. Suspicious z_t→SID=0.592 (normally ≈1.0) "
                "suggests information loss or routing collapse.",
    },
    "combined": {
        "category": "routing_variant",
        "hparams": "β=0.03  grl=0.01  grl_phoneme=0.01  ste=True  ub=0.01",
        "desc": "Kitchen-sink combination. Worst overall result — conflicting "
                "gradients from all interventions simultaneously. "
                "z_L→SID=0.750, z_P→PR=0.160.",
    },
    "dual_weak_ub": {
        "category": "routing_variant",
        "hparams": "β=0.01  grl=0.01  grl_phoneme=0.01  ub_weight=0.01",
        "desc": "Dual GRL with weak β=0.01 + UB. Best z_P→SID (0.942) — "
                "phoneme adversary on z_P concentrates speaker info in P. "
                "But z_L→SID degrades to 0.330 vs weakgrl's 0.104.",
    },
    "no_routing": {
        "category": "ablation",
        "hparams": "β=0.01  grl=0.01  no_routing=True",
        "desc": "No routing module — all features shared. GRL still applied. "
                "Ablation: what does GRL achieve without route separation?",
    },
    "fixed_70_30": {
        "category": "ablation",
        "hparams": "β=0.01  grl=0.01  fixed_routing=True  split=0.7/0.3",
        "desc": "Fixed 70/30 routing (not learned). Ablation: does learned "
                "routing help over a random fixed assignment?",
    },
}

PROBE_RESULTS = {
    "sid1_weakgrl": {"zL_pr": 0.056, "zL_sid": 0.104, "zP_pr": 0.573, "zP_sid": 0.866,
                     "zt_sid": 1.000},
    "sid1_nogrl":   {"zL_pr": 0.055, "zL_sid": 0.784, "zP_pr": 0.394, "zP_sid": 0.926,
                     "zt_sid": 1.000},
    "beta_002":     {"zL_pr": 0.059, "zL_sid": 0.516, "zP_pr": 0.298, "zP_sid": 0.920,
                     "zt_sid": 1.000},
    "beta_003":     {"zL_pr": 0.061, "zL_sid": 0.200, "zP_pr": 0.184, "zP_sid": 0.938,
                     "zt_sid": 1.000},
    "dual_grl_03":  {"zL_pr": 0.059, "zL_sid": 0.844, "zP_pr": 0.425, "zP_sid": 0.936,
                     "zt_sid": 0.998},
    "ub":           {"zL_pr": 0.057, "zL_sid": 0.292, "zP_pr": 0.293, "zP_sid": 0.900,
                     "zt_sid": 1.000},
    "ste":          {"zL_pr": 0.055, "zL_sid": 0.186, "zP_pr": 0.297, "zP_sid": 0.902,
                     "zt_sid": 1.000},
    "decor_only":   {"zL_pr": 0.058, "zL_sid": 0.566, "zP_pr": 0.263, "zP_sid": 0.796,
                     "zt_sid": 0.740},
    "ste_ub":       {"zL_pr": 0.055, "zL_sid": 0.568, "zP_pr": 0.317, "zP_sid": 0.890,
                     "zt_sid": 0.592},
    "combined":     {"zL_pr": 0.063, "zL_sid": 0.750, "zP_pr": 0.160, "zP_sid": 0.956,
                     "zt_sid": 0.836},
    "dual_weak_ub": {"zL_pr": 0.059, "zL_sid": 0.330, "zP_pr": 0.382, "zP_sid": 0.942,
                     "zt_sid": 0.804},
}

# ── helpers ────────────────────────────────────────────────────────────────────

def _img(path):
    p = Path(path)
    return mpimg.imread(str(p)) if p.exists() else None


def _add_img(ax, path, title=None):
    img = _img(path)
    ax.axis("off")
    if img is not None:
        ax.imshow(img, aspect="auto")
    else:
        ax.text(0.5, 0.5, f"[not generated]\n{Path(path).name}",
                ha="center", va="center", fontsize=8, color="gray",
                transform=ax.transAxes)
    if title:
        ax.set_title(title, fontsize=9, pad=3)


def _section_header(fig, y, text, color="#333333"):
    fig.text(0.05, y, text, fontsize=13, fontweight="bold", color=color,
             transform=fig.transFigure)
    fig.add_artist(plt.matplotlib.lines.Line2D(
        [0.05, 0.95], [y - 0.012, y - 0.012],
        transform=fig.transFigure, color=color, lw=1.2, alpha=0.5))


def _wrap(text, width=110):
    return "\n".join(textwrap.wrap(text, width))


# ── cover page ────────────────────────────────────────────────────────────────

def page_cover(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#F8F9FA")

    # Title block
    fig.text(0.5, 0.72, "Disentanglement Experiments",
             ha="center", fontsize=28, fontweight="bold", color="#1a1a2e")
    fig.text(0.5, 0.65, "SAE-based Feature Routing for Speech Disentanglement",
             ha="center", fontsize=16, color="#444444", style="italic")
    fig.text(0.5, 0.60, f"Generated {datetime.date.today().isoformat()}",
             ha="center", fontsize=11, color="#888888")

    # Horizontal rule
    fig.add_artist(plt.matplotlib.lines.Line2D(
        [0.1, 0.9], [0.56, 0.56], transform=fig.transFigure,
        color="#1a1a2e", lw=2))

    # Summary stats
    stats = [
        ("Architecture", "SPEAR-XL (600M, D=1280) + SAE (K=5120, topk=256)"),
        ("Stage-1",      "SAE reconstruction — best val_recon=0.0049  (R²≈0.80)"),
        ("Stage-2 runs", "14 experiments across GRL sweep, β scaling, routing variants, SAE variants, ablations"),
        ("Probe metrics","z_L→SID ↓  (speaker removal from L)   |   z_P→PR ↑  (phoneme removal from P)"),
        ("Best result",  "sid1_weakgrl  (β=0.01, grl=0.01)  →  z_L→SID=0.104, z_P→PR=0.573"),
    ]
    y = 0.50
    for label, value in stats:
        fig.text(0.12, y, f"{label}:", fontsize=11, fontweight="bold", color="#333333")
        fig.text(0.32, y, value,        fontsize=11, color="#444444")
        y -= 0.065

    # Category legend
    y -= 0.02
    fig.text(0.12, y, "Experiment categories:", fontsize=10, fontweight="bold", color="#333333")
    y -= 0.045
    cat_labels = {
        "baseline": "Stage-1 Baseline",
        "sae_variant": "SAE Variants",
        "grl_sweep": "GRL Sweep",
        "beta_sweep": "β Scaling",
        "routing_variant": "Routing Variants",
        "ablation": "Ablations",
    }
    x = 0.12
    for cat, lbl in cat_labels.items():
        col = CATEGORY_COLORS[cat]
        rect = FancyBboxPatch((x, y - 0.015), 0.02, 0.022,
                              boxstyle="round,pad=0.002",
                              facecolor=col, edgecolor="none",
                              transform=fig.transFigure, clip_on=False)
        fig.add_artist(rect)
        fig.text(x + 0.025, y, lbl, fontsize=9, color="#333333", va="center")
        x += 0.145
        if x > 0.88:
            x = 0.12; y -= 0.04

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  cover page")


# ── stage-1 page ──────────────────────────────────────────────────────────────

def page_stage1(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    _section_header(fig, 0.94, "1.  Stage-1 Baseline SAE", CATEGORY_COLORS["baseline"])

    # Description block
    desc = ("Pure reconstruction SAE (K=5120 features, topk=256 active per frame = 5% sparsity) "
            "trained on frozen SPEAR-XL layer-averaged features (D=1280). No disentanglement objective. "
            "Provides the shared encoder checkpoint used as starting point for all stage-2 experiments. "
            "Key statistics: val_recon=0.0049, zero-predictor MSE≈0.024, R²≈0.80.")
    fig.text(0.05, 0.88, _wrap(desc, 115), fontsize=9, color="#444444", va="top")

    # Plots: training | validation
    ax1 = fig.add_axes([0.05, 0.45, 0.42, 0.38])
    ax2 = fig.add_axes([0.53, 0.45, 0.42, 0.38])
    _add_img(ax1, ANA_DIR / "baseline/training.png",   "Stage-1 Training Curve")
    _add_img(ax2, ANA_DIR / "baseline/validation.png", "Stage-1 Validation Curve")

    # Feature stats table
    fig.text(0.05, 0.40, "Feature Statistics (SPEAR-XL, val set):", fontsize=10,
             fontweight="bold", color="#333333")
    rows = [
        ("Feature dim D",          "1280"),
        ("SAE latent size K",       "5120   (4 × D)"),
        ("Active features / frame", "256    (topk, 5% of K)"),
        ("Untrained MSE (≈zero pred.)", "0.0246"),
        ("Trained val_recon",       "0.0049"),
        ("R²  (= 1 − 0.0049/0.0246)", "≈ 0.80"),
        ("Grad norm at convergence", "~0.006  (stable)"),
    ]
    y = 0.365
    for lbl, val in rows:
        fig.text(0.07, y, lbl,  fontsize=9, color="#555555")
        fig.text(0.38, y, val,  fontsize=9, color="#222222", fontweight="bold")
        y -= 0.038

    # Decor_only stage-1 side panel
    fig.text(0.53, 0.40, "Decorrelation SAE (decor_only stage-1):", fontsize=10,
             fontweight="bold", color=CATEGORY_COLORS["sae_variant"])
    ax3 = fig.add_axes([0.53, 0.05, 0.44, 0.32])
    _add_img(ax3, ANA_DIR / "decor_only/decor_analysis.png",
             "Decorrelation Loss & Gradient Norms")

    rows2 = [
        ("decor_weight", "1.0   (delta × decor ≈ recon at init)"),
        ("decor loss reduction", "0.0026 → 0.0010  (−62%)"),
        ("Grad spike at step 4500", "×512 median  (random subsample variance)"),
        ("Best val_recon (stage-1)", "0.0069  (vs baseline 0.0049)"),
    ]
    y = 0.365
    for lbl, val in rows2:
        fig.text(0.55, y, lbl,  fontsize=9, color="#555555")
        fig.text(0.75, y, val,  fontsize=9, color="#222222", fontweight="bold")
        y -= 0.038

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  stage-1 page")


# ── per-experiment page ────────────────────────────────────────────────────────

def page_experiment(pdf, exp, section_num, section_label=None):
    cfg   = EXPERIMENTS[exp]
    cat   = cfg["category"]
    col   = CATEGORY_COLORS[cat]
    probe = PROBE_RESULTS.get(exp, {})
    label = section_label or exp

    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    header = f"{section_num}.  {label}"
    _section_header(fig, 0.945, header, col)

    # Hparams + description
    fig.text(0.05, 0.910, f"Hyperparameters:  {cfg['hparams']}",
             fontsize=9, color="#555555", style="italic")
    fig.text(0.05, 0.875, _wrap(cfg["desc"], 115), fontsize=9, color="#444444", va="top")

    has_probe = bool(probe)
    has_routing = (ANA_DIR / exp / "routing.png").exists()
    has_training = (ANA_DIR / exp / "training.png").exists()
    has_val = (ANA_DIR / exp / "validation.png").exists()

    if has_probe:
        # Layout: training | validation (top row), routing | probe | radar (bottom row)
        ax_tr  = fig.add_axes([0.04, 0.49, 0.44, 0.35])
        ax_val = fig.add_axes([0.52, 0.49, 0.44, 0.35])
        ax_rt  = fig.add_axes([0.04, 0.05, 0.27, 0.38])
        ax_pb  = fig.add_axes([0.36, 0.05, 0.35, 0.38])
        ax_rd  = fig.add_axes([0.74, 0.05, 0.23, 0.38])

        _add_img(ax_tr,  ANA_DIR / exp / "training.png",   "Training Losses")
        _add_img(ax_val, ANA_DIR / exp / "validation.png", "Validation Metrics")
        if has_routing:
            _add_img(ax_rt, ANA_DIR / exp / "routing.png", "Routing Fractions")
        else:
            ax_rt.axis("off")
        _add_img(ax_pb, ANA_DIR / exp / "probe.png",   "Probe Results")
        _add_img(ax_rd, ANA_DIR / exp / "radar.png",   "Radar")

        # Probe summary text — percentages
        if probe:
            fig.text(0.04, 0.455, "Probe summary:", fontsize=9, fontweight="bold", color=col)
            items = [
                (f"z_L→SID = {probe.get('zL_sid',0)*100:.1f}%", "↓ Axis 1: speaker in L"),
                (f"z_P→PR  = {probe.get('zP_pr', 0)*100:.1f}%", "↑ Axis 2: phonemes in P"),
                (f"z_P→SID = {probe.get('zP_sid',0)*100:.1f}%", "↑ speaker in P"),
                (f"z_t→SID = {probe.get('zt_sid',0)*100:.1f}%", "SAE info check"),
            ]
            x = 0.04
            for val_str, note in items:
                fig.text(x, 0.428, val_str, fontsize=9, fontweight="bold", color="#1a1a1a")
                fig.text(x, 0.410, note,    fontsize=7.5, color="#666666")
                x += 0.23
    else:
        # No probe — training + validation + routing
        ax_tr  = fig.add_axes([0.04, 0.10, 0.44, 0.72])
        ax_val = fig.add_axes([0.52, 0.10, 0.44, 0.72])
        _add_img(ax_tr,  ANA_DIR / exp / "training.png",   "Training Losses")
        _add_img(ax_val, ANA_DIR / exp / "validation.png", "Validation Metrics")
        if has_routing:
            # Inset routing below val
            ax_rt = fig.add_axes([0.52, 0.05, 0.44, 0.20])
            _add_img(ax_rt, ANA_DIR / exp / "routing.png", "Routing Fractions")
        fig.text(0.5, 0.05, "No probe data for this experiment.",
                 ha="center", fontsize=9, color="#aaaaaa", style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print(f"  {exp}")


# ── comparison page ────────────────────────────────────────────────────────────

def page_comparison_scatter(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    _section_header(fig, 0.945, "Comparison  —  Disentanglement Scatter Plots", "#222222")

    ax1 = fig.add_axes([0.04, 0.08, 0.44, 0.80])
    ax2 = fig.add_axes([0.53, 0.08, 0.44, 0.80])
    _add_img(ax1, ANA_DIR / "comparison/scatter_disentanglement.png",
             "Primary axes: z_L→SID vs z_P→PR")
    _add_img(ax2, ANA_DIR / "comparison/scatter_speaker.png",
             "Speaker push/pull: z_L→SID vs z_P→SID")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  comparison scatter")


def page_comparison_heatmap(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    _section_header(fig, 0.945, "Comparison  —  Probe Results Heatmap", "#222222")

    ax = fig.add_axes([0.08, 0.05, 0.84, 0.85])
    _add_img(ax, ANA_DIR / "comparison/probe_heatmap.png")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  comparison heatmap")


def page_comparison_val(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    _section_header(fig, 0.945, "Comparison  —  Validation Convergence & Reconstruction Quality", "#222222")

    ax1 = fig.add_axes([0.04, 0.10, 0.54, 0.78])
    ax2 = fig.add_axes([0.62, 0.10, 0.35, 0.78])
    _add_img(ax1, ANA_DIR / "comparison/val_recon_groups.png",
             "Val Recon Convergence (grouped)")
    _add_img(ax2, ANA_DIR / "comparison/reconstruction_quality.png",
             "Best Val Recon per Run")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  comparison val + recon quality")


def page_routing_entropy(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    _section_header(fig, 0.945, "Comparison  —  Routing Entropy (All Runs)", "#222222")

    fig.text(0.05, 0.895, _wrap(
        "All runs maintain routing entropy ≈ ln(3) = 1.099 nats throughout training. "
        "This indicates the Gumbel-softmax routing logits barely move from their "
        "zero-initialisation — |logit| ≈ 0.003 at convergence vs maximum possible spread. "
        "The routing is effectively near-uniform in all experiments. "
        "This is caused by the route entropy loss (anti-collapse regularisation) with "
        "lr_routing=5×10⁻⁶ keeping logits near zero. Despite this, task gradients still "
        "flow through the soft masks and produce meaningful disentanglement.", 115),
        fontsize=9, color="#444444", va="top")

    ax = fig.add_axes([0.08, 0.12, 0.84, 0.71])
    _add_img(ax, ANA_DIR / "comparison/routing_entropy_all.png")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  routing entropy")


def page_summary_table(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    _section_header(fig, 0.945, "Summary  —  All Probe Results", "#222222")

    fig.text(0.05, 0.895, _wrap(
        "Primary disentanglement axes: Axis 1 = z_L→SID ↓ (speaker removal from linguistic route), "
        "Axis 2 = z_P→PR ↑ (phoneme removal from paralinguistic route, measured as PER — higher = better). "
        "z_P→SID ↑ measures speaker concentration in P. z_t→SID ≈ 1.0 expected; lower values indicate "
        "SAE information loss.", 115),
        fontsize=9, color="#444444", va="top")

    # Table data
    headers = ["Experiment", "Category", "z_L→SID ↓", "z_P→PR ↑", "z_P→SID ↑", "z_t→SID", "Notes"]
    rows_data = []
    for exp, p in sorted(PROBE_RESULTS.items(),
                         key=lambda x: (x[1].get("zL_sid", 99) - x[1].get("zP_pr", 0))):
        cat = EXPERIMENTS[exp]["category"].replace("_", " ")
        note = ""
        if exp == "sid1_weakgrl":
            note = "★ Best"
        elif p.get("zt_sid", 1.0) < 0.8:
            note = "⚠ SAE loss"
        rows_data.append([
            exp, cat,
            f"{p.get('zL_sid',0)*100:.1f}%",
            f"{p.get('zP_pr', 0)*100:.1f}%",
            f"{p.get('zP_sid',0)*100:.1f}%",
            f"{p.get('zt_sid',0)*100:.1f}%",
            note,
        ])

    ax = fig.add_axes([0.02, 0.08, 0.96, 0.76])
    ax.axis("off")

    col_widths = [0.20, 0.14, 0.10, 0.10, 0.10, 0.10, 0.20]
    col_x = [0.01]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    # Header row
    y = 0.96
    for i, (h, x) in enumerate(zip(headers, col_x)):
        ax.text(x, y, h, fontsize=9, fontweight="bold", color="white",
                transform=ax.transAxes, va="center")
    ax.add_patch(plt.Rectangle((0, y - 0.025), 1.0, 0.055,
                               facecolor="#333333", transform=ax.transAxes,
                               clip_on=False))
    for i, (h, x) in enumerate(zip(headers, col_x)):
        ax.text(x + 0.005, y, h, fontsize=9, fontweight="bold", color="white",
                transform=ax.transAxes, va="center")
    y -= 0.065

    for ri, row in enumerate(rows_data):
        bg = "#F0F4F8" if ri % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y - 0.022), 1.0, 0.050,
                                   facecolor=bg, transform=ax.transAxes,
                                   clip_on=False))
        exp_name = row[0]
        cat_col  = CATEGORY_COLORS.get(EXPERIMENTS[exp_name]["category"], "#333")
        for i, (val, x) in enumerate(zip(row, col_x)):
            color = "#1a1a1a"
            fw = "normal"
            if i == 0:
                color = cat_col; fw = "bold"
            elif i == 2 and "%" in str(val):  # z_L→SID % — lower = better
                try:
                    v = float(val.replace("%",""))
                    color = "#006400" if v < 30 else ("#cc0000" if v > 70 else "#333333")
                except: pass
            elif i == 3 and "%" in str(val):  # z_P→PR % — higher = better
                try:
                    v = float(val.replace("%",""))
                    color = "#006400" if v > 40 else ("#cc0000" if v < 25 else "#333333")
                except: pass
            ax.text(x + 0.005, y, val, fontsize=8.5, color=color,
                    fontweight=fw, transform=ax.transAxes, va="center")
        y -= 0.065

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  summary table")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Generating report → {OUT_PDF.relative_to(DIS_DIR)}\n")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(OUT_PDF)) as pdf:

        # Cover
        page_cover(pdf)

        # ── Section 1: Stage-1 Baseline
        page_stage1(pdf)

        # ── Section 2: GRL Sweep
        for i, exp in enumerate(["sid1_weakgrl", "sid1_nogrl",
                                  "sid1_delayedgrl", "sid1_highrho"], start=1):
            page_experiment(pdf, exp,
                            section_num=f"2.{i}",
                            section_label=f"GRL Sweep — {exp}")

        # ── Section 3: Beta Scaling
        for i, exp in enumerate(["beta_002", "beta_003"], start=1):
            page_experiment(pdf, exp,
                            section_num=f"3.{i}",
                            section_label=f"β Scaling — {exp}")

        # ── Section 4: Routing Variants
        for i, exp in enumerate(["dual_grl_03", "ub", "ste",
                                  "ste_ub", "combined", "dual_weak_ub"], start=1):
            page_experiment(pdf, exp,
                            section_num=f"4.{i}",
                            section_label=f"Routing Variant — {exp}")

        # ── Section 5: SAE Variants
        for i, exp in enumerate(["decor_only", "K10240_t128"], start=1):
            page_experiment(pdf, exp,
                            section_num=f"5.{i}",
                            section_label=f"SAE Variant — {exp}")

        # ── Section 6: Ablations
        for i, exp in enumerate(["no_routing", "fixed_70_30"], start=1):
            page_experiment(pdf, exp,
                            section_num=f"6.{i}",
                            section_label=f"Ablation — {exp}")

        # ── Section 7: Comparison
        page_comparison_scatter(pdf)
        page_comparison_heatmap(pdf)
        page_comparison_val(pdf)
        page_routing_entropy(pdf)
        page_summary_table(pdf)

        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "Disentanglement Experiments Report"
        d["Author"]  = "bbg25"
        d["Subject"] = "SAE-based feature routing for speech disentanglement"
        d["CreationDate"] = datetime.datetime.now()

    print(f"\nDone → {OUT_PDF}")


if __name__ == "__main__":
    main()
