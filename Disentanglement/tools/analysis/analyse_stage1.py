#!/usr/bin/env python3
"""Stage-1 analysis: training curves + checkpoint inspection.

Outputs
-------
  analysis/stage1/stage1_curves.png        — loss + routing + tau curves
  analysis/stage1/stage1_layer_weights.png — fixed/uniform SPEAR layer mix
  analysis/stage1/stage1_sparsity.png      — SAE activation statistics on val
  analysis/stage1/stage1_summary.txt       — key numbers
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

DIS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DIS_DIR))

OUT  = DIS_DIR / "analysis" / "stage1"
OUT.mkdir(parents=True, exist_ok=True)

CKPT_CANDIDATES = [
    DIS_DIR / "checkpoints" / "stage1_best.pt",
    DIS_DIR / "checkpoints" / "best.pt",
]
CKPT = next((p for p in CKPT_CANDIDATES if p.exists()), CKPT_CANDIDATES[0])
LOG_DIR = DIS_DIR / "logs" / "train" / "stage1"
LOG  = sorted(LOG_DIR.glob("*.out"))

# ---------------------------------------------------------------- 1. Parse log
def parse_log(paths: list[Path]) -> dict:
    steps, recon, n_L, n_P, n_U, tau = [], [], [], [], [], []
    pat = re.compile(
        r"step\s+(\d+)/\d+\s+recon=([\d.]+).*?L/P/U=(\d+)/(\d+)/(\d+)\s+tau=([\d.]+)"
    )
    for path in paths:
        for line in path.read_text().splitlines():
            m = pat.search(line)
            if m:
                steps.append(int(m[1]))
                recon.append(float(m[2]))
                n_L.append(int(m[3]))
                n_P.append(int(m[4]))
                n_U.append(int(m[5]))
                tau.append(float(m[6]))
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    return {
        "steps": [steps[i] for i in order],
        "recon": [recon[i] for i in order],
        "n_L":   [n_L[i]   for i in order],
        "n_P":   [n_P[i]   for i in order],
        "n_U":   [n_U[i]   for i in order],
        "tau":   [tau[i]    for i in order],
    }

# ---------------------------------------------------------------- 2. Training curves
def plot_curves(d: dict) -> None:
    steps = d["steps"]
    K = d["n_L"][0] + d["n_P"][0] + d["n_U"][0]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

    # Reconstruction loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, d["recon"], color="#2196F3", linewidth=1.5)
    ax1.set_title("Reconstruction Loss (MSE)", fontsize=11)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Routing counts
    ax2 = fig.add_subplot(gs[0, 1])
    frac_L = [x/K*100 for x in d["n_L"]]
    frac_P = [x/K*100 for x in d["n_P"]]
    frac_U = [x/K*100 for x in d["n_U"]]
    ax2.plot(steps, frac_L, label="Linguistic (L)", color="#4CAF50", linewidth=1.5)
    ax2.plot(steps, frac_P, label="Paralinguistic (P)", color="#FF9800", linewidth=1.5)
    ax2.plot(steps, frac_U, label="Residual (U)", color="#9C27B0", linewidth=1.5)
    ax2.axhline(100/3, color="gray", linestyle="--", linewidth=0.8, label="Uniform (33.3%)")
    ax2.set_title(f"Routing Group Fractions  (K={K})", fontsize=11)
    ax2.set_xlabel("Step"); ax2.set_ylabel("% of units")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Gumbel tau
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, d["tau"], color="#F44336", linewidth=1.5)
    ax3.set_title("Gumbel Temperature (τ)", fontsize=11)
    ax3.set_xlabel("Step"); ax3.set_ylabel("τ")
    ax3.grid(True, alpha=0.3)

    # Routing absolute counts
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.stackplot(
        steps,
        d["n_L"], d["n_P"], d["n_U"],
        labels=["L", "P", "U"],
        colors=["#4CAF50", "#FF9800", "#9C27B0"],
        alpha=0.75,
    )
    ax4.set_title("Routing Group Counts (stacked)", fontsize=11)
    ax4.set_xlabel("Step"); ax4.set_ylabel("# units")
    ax4.legend(fontsize=8, loc="upper right"); ax4.grid(True, alpha=0.3)

    fig.suptitle("Stage 1 Training — Reconstruction + Routing", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "stage1_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {OUT}/stage1_curves.png")

# ---------------------------------------------------------------- 3. Load checkpoint + val pass
def load_and_analyse() -> None:
    from unittest.mock import patch, MagicMock
    real_load = torch.load

    from config import DISConfig
    cfg = DISConfig()
    cfg.device = "cpu"
    cfg.bf16   = False

    # Load checkpoint to read cfg overrides
    ckpt = real_load(CKPT, map_location="cpu", weights_only=False)
    cfg.num_speakers = ckpt.get("num_speakers", cfg.num_speakers)
    cfg.vocab_size   = ckpt.get("vocab_size",   cfg.vocab_size)
    print(f"  num_speakers={cfg.num_speakers}  vocab_size={cfg.vocab_size}")

    # Build model with real SPEAR (loaded from HuggingFace cache)
    print("  loading SPEAR from cache …")
    from model import build_dis_model
    model = build_dis_model(cfg)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # --- Layer weights. Current SPEAR uses fixed uniform averaging; older
    # checkpoints may still carry a trainable mix_logits parameter.
    if hasattr(model.encoder, "mix_logits"):
        weights = torch.softmax(model.encoder.mix_logits, dim=0).detach().cpu().numpy()
        layer_title = "Learned Layer-Weighted Sum of SPEAR Hidden States"
    else:
        n_layers = getattr(model.encoder, "n_layers", 13)
        weights = np.full(n_layers, 1.0 / n_layers)
        layer_title = "Fixed Uniform Average of SPEAR Hidden States"
    plot_layer_weights(weights, layer_title)

    # --- Routing hard assignment
    with torch.no_grad():
        m_L, m_P, m_U = model.routing()
    hard = model.routing.hard_counts
    print(f"  final routing: L={hard[0]}  P={hard[1]}  U={hard[2]}")

    # --- Val sparsity (no data needed — just examine the checkpoint routing)
    analyse_routing(model)

    return model, cfg, ckpt


def plot_layer_weights(weights: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(range(len(weights)), weights, color="#2196F3", alpha=0.8)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels([f"L{i}" for i in range(len(weights))], fontsize=9)
    ax.set_xlabel("SPEAR Transformer Layer")
    ax.set_ylabel("Softmax Weight")
    ax.set_title(title, fontsize=11)
    # annotate top-3
    top3 = np.argsort(weights)[-3:]
    for i in top3:
        ax.bar(i, weights[i], color="#F44336", alpha=0.9)
        ax.text(i, weights[i] + 0.002, f"{weights[i]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "stage1_layer_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {OUT}/stage1_layer_weights.png")


def analyse_routing(model) -> None:
    K = model.cfg.K
    with torch.no_grad():
        logits = model.routing.logits          # (K, 3)
        probs  = torch.softmax(logits, dim=-1) # (K, 3)
        hard   = probs.argmax(-1)              # (K,)  0=L 1=P 2=U

    labels = ["L", "P", "U"]
    colors = ["#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Soft probability distributions per group
    for g, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.hist(probs[:, g].numpy(), bins=40, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(probs[:, g].mean().item(), color="black", linestyle="--",
                   linewidth=1.2, label=f"mean={probs[:,g].mean():.3f}")
        ax.set_title(f"Soft P(group={label})", fontsize=10)
        ax.set_xlabel("Routing probability"); ax.set_ylabel("# units")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Routing Soft Probabilities  (K={K}  |  final hard: "
        f"L={int((hard==0).sum())} P={int((hard==1).sum())} U={int((hard==2).sum())})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "stage1_routing_probs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {OUT}/stage1_routing_probs.png")


# ---------------------------------------------------------------- 4. Summary text
def write_summary(d: dict, model) -> None:
    K   = d["n_L"][0] + d["n_P"][0] + d["n_U"][0]
    r0  = d["recon"][0];  r_final = d["recon"][-1]
    r_reduction = (r0 - r_final) / r0 * 100

    # Find step where recon started consistently falling (first step where
    # recon < 95% of initial)
    threshold = r0 * 0.95
    breakout  = next((s for s, r in zip(d["steps"], d["recon"]) if r < threshold), None)

    if hasattr(model.encoder, "mix_logits"):
        weights = torch.softmax(model.encoder.mix_logits, dim=0).detach().cpu().numpy()
    else:
        n_layers = getattr(model.encoder, "n_layers", 13)
        weights = np.full(n_layers, 1.0 / n_layers)
    top_layers = np.argsort(weights)[-3:][::-1]

    hard = model.routing.hard_counts
    n_L, n_P, n_U = hard

    lines = [
        "=" * 60,
        "Stage 1 Analysis Summary",
        "=" * 60,
        "",
        "--- Reconstruction ---",
        f"  Initial recon loss   : {r0:.4f}",
        f"  Final recon loss     : {r_final:.4f}",
        f"  Reduction            : {r_reduction:.1f}%",
        f"  Loss breakout step   : {breakout}  (< {threshold:.4f})",
        "",
        "--- Routing (final hard assignment) ---",
        f"  K = {K}",
        f"  L (linguistic)       : {n_L}  ({n_L/K*100:.1f}%)",
        f"  P (paralinguistic)   : {n_P}  ({n_P/K*100:.1f}%)",
        f"  U (residual)         : {n_U}  ({n_U/K*100:.1f}%)",
        f"  Entropy at end       : {model.routing.routing_entropy:.4f} nats  (max=1.0986)",
        "",
        "--- SPEAR Layer Weights ---",
        f"  Top-3 layers         : {', '.join(f'L{i} ({weights[i]:.3f})' for i in top_layers)}",
        f"  Weight concentration : top-3 sum = {weights[top_layers].sum():.3f} / 1.0",
        "",
        "--- Key Observations ---",
    ]

    # Observations
    if n_P / K < 0.25:
        lines.append(f"  [!] P group is under-represented ({n_P/K*100:.1f}%) — "
                     "paralinguistic features are being compressed. Stage 2 SID "
                     "objective will either grow P or concentrate it further.")
    if abs(n_L - n_U) / K < 0.02:
        lines.append(f"  [✓] L and U groups are roughly equal ({n_L/K*100:.1f}% / "
                     f"{n_U/K*100:.1f}%) — residual channel absorbing similar "
                     "capacity to linguistic.")
    if r_final < 0.20:
        lines.append(f"  [✓] Recon loss < 0.20 — SAE is reconstructing SPEAR "
                     "features well; decoder has learned the mapping.")
    if breakout and breakout > 2000:
        lines.append(f"  [!] Loss plateau for first {breakout} steps — warmup + "
                     "routing randomness. Normal for stage 1 with no task signal.")
    if weights[top_layers[0]] > 0.15:
        lines.append(f"  [✓] Layer L{top_layers[0]} dominant ({weights[top_layers[0]]:.3f}) "
                     "— model is specialising to a preferred SPEAR layer.")

    lines += ["", "=" * 60]

    txt = "\n".join(lines)
    (OUT / "stage1_summary.txt").write_text(txt)
    print(txt)


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    print("\n=== Stage 1 Analysis ===\n")

    if not LOG:
        print(f"No stage1 log files found in {LOG_DIR}/"); sys.exit(1)
    if not CKPT.exists():
        print(f"Checkpoint not found: {CKPT}"); sys.exit(1)

    print("[1/4] Parsing training log …")
    d = parse_log(LOG)
    print(f"      {len(d['steps'])} log points  "
          f"(steps {d['steps'][0]}–{d['steps'][-1]})")

    print("[2/4] Plotting training curves …")
    plot_curves(d)

    print("[3/4] Loading checkpoint + analysing model …")
    model, cfg, ckpt = load_and_analyse()

    print("[4/4] Writing summary …")
    write_summary(d, model)

    print(f"\nAll outputs → {OUT}/")
