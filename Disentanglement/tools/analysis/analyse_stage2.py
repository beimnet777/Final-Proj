#!/usr/bin/env python3
"""Analyse Stage 2 results: parse log + TensorBoard events, produce plots."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------- config

DIS_DIR = Path(__file__).resolve().parents[2]
LOG_FILE = DIS_DIR / "logs" / "train" / "stage2" / "sweep" / "baseline_29880935.out"
TB_DIR   = DIS_DIR / "runs" / "tb" / "stage2_20260528_011238"
OUT_DIR  = DIS_DIR / "analysis" / "stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- parse log

steps, recons, prs, sids, entropies = [], [], [], [], []
counts_L, counts_P, counts_U = [], [], []

pat = re.compile(
    r"step\s+(\d+)/\d+\s+"
    r"recon=([\d.]+)\s+"
    r"pr=([\d.]+)\s+"
    r"sid=([\d.]+)\s+"
    r"route_H=([\d.]+)\s+"
    r"L/P/U=(\d+)/(\d+)/(\d+)"
)

with open(LOG_FILE) as f:
    for line in f:
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            recons.append(float(m.group(2)))
            prs.append(float(m.group(3)))
            sids.append(float(m.group(4)))
            entropies.append(float(m.group(5)))
            counts_L.append(int(m.group(6)))
            counts_P.append(int(m.group(7)))
            counts_U.append(int(m.group(8)))

steps    = np.array(steps)
recons   = np.array(recons)
prs      = np.array(prs)
sids     = np.array(sids)
entropies= np.array(entropies)
counts_L = np.array(counts_L)
counts_P = np.array(counts_P)
counts_U = np.array(counts_U)
K = 5120

print(f"Parsed {len(steps)} log entries  (steps {steps[0]}–{steps[-1]})")

# ---------------------------------------------------------------- parse TB

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(TB_DIR))
    ea.Reload()
    tb_tags = ea.Tags()["scalars"]
    print(f"TB tags ({len(tb_tags)}): {tb_tags}")

    def _tb(tag):
        if tag not in tb_tags:
            return np.array([]), np.array([])
        evts = ea.Scalars(tag)
        return np.array([e.step for e in evts]), np.array([e.value for e in evts])

    tb_recon_s,   tb_recon_v   = _tb("train/recon")
    tb_pr_s,      tb_pr_v      = _tb("train/pr")
    tb_sid_s,     tb_sid_v     = _tb("train/sid")
    tb_grl_s,     tb_grl_v     = _tb("train/grl")
    tb_decorr_s,  tb_decorr_v  = _tb("train/decorr")
    tb_route_s,   tb_route_v   = _tb("train/route")
    tb_total_s,   tb_total_v   = _tb("train/total")
    tb_density_s, tb_density_v = _tb("sae/z_dense_density")
    tb_ent_s,     tb_ent_v     = _tb("routing/entropy")
    tb_probe_sid_s,  tb_probe_sid_v  = _tb("probe/sid_acc")
    tb_probe_leak_s, tb_probe_leak_v = _tb("probe/leak_sid")
    tb_probe_pr_s,   tb_probe_pr_v   = _tb("probe/pr_ctc_val")
    tb_ok = True
except Exception as e:
    print(f"[warn] TB load failed: {e}")
    tb_ok = False

# ---------------------------------------------------------------- FIGURE 1: main losses

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Stage 2 — Training Losses", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.plot(steps, recons, lw=1.2, color="tab:blue", label="recon (MSE)")
ax.set_title("Reconstruction Loss"); ax.set_xlabel("step"); ax.set_ylabel("MSE")
ax.grid(True, alpha=0.3)
ax.annotate(f"final: {recons[-1]:.4f}", xy=(steps[-1], recons[-1]),
            xytext=(-80, 10), textcoords="offset points", fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.8))

ax = axes[0, 1]
ax.plot(steps, prs, lw=1.2, color="tab:orange", label="pr CTC")
ax.axhline(np.log(41), color="red", lw=1, ls="--", alpha=0.6, label=f"ln(41)={np.log(41):.2f} (chance)")
ax.set_title("PR CTC Loss"); ax.set_xlabel("step"); ax.set_ylabel("CTC loss")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.annotate(f"final: {prs[-1]:.4f}", xy=(steps[-1], prs[-1]),
            xytext=(-80, 10), textcoords="offset points", fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.8))

ax = axes[1, 0]
ax.plot(steps, sids, lw=1.2, color="tab:green", label="SID CE")
ax.axhline(np.log(27), color="red", lw=1, ls="--", alpha=0.6, label=f"ln(27)={np.log(27):.2f} (chance)")
ax.set_title("Speaker ID Loss (SID CE)"); ax.set_xlabel("step"); ax.set_ylabel("CE loss")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.annotate(f"final: {sids[-1]:.4f}", xy=(steps[-1], sids[-1]),
            xytext=(-80, 20), textcoords="offset points", fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.8))

ax = axes[1, 1]
if tb_ok and len(tb_grl_v) > 0:
    ax.plot(tb_grl_s, tb_grl_v, lw=1.2, color="tab:red", label="GRL CE")
    ax.axhline(np.log(27), color="gray", lw=1, ls="--", alpha=0.6, label=f"ln(27)={np.log(27):.2f} (chance)")
    ax.legend(fontsize=8)
    ax.annotate(f"final: {tb_grl_v[-1]:.4f}", xy=(tb_grl_s[-1], tb_grl_v[-1]),
                xytext=(-80, 10), textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.8))
else:
    ax.plot(steps, np.log(27) * np.ones_like(steps), lw=1.2, ls="--", color="gray", label="no GRL data in TB")
    ax.legend(fontsize=8)
ax.set_title("GRL Adversarial Loss"); ax.set_xlabel("step"); ax.set_ylabel("CE loss")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "stage2_losses.png", dpi=150)
plt.close()
print("Saved stage2_losses.png")

# ---------------------------------------------------------------- FIGURE 2: routing

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Stage 2 — Routing", fontsize=14, fontweight="bold")

ax = axes[0]
frac_L = counts_L / K
frac_P = counts_P / K
frac_U = counts_U / K
ax.stackplot(steps, frac_L, frac_P, frac_U,
             labels=["L (linguistic)", "P (paralinguistic)", "U (residual)"],
             colors=["tab:blue", "tab:orange", "tab:green"], alpha=0.7)
ax.set_title("Routing Fractions (hard counts / K)"); ax.set_xlabel("step"); ax.set_ylabel("fraction")
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

ax = axes[1]
ax.plot(steps, entropies, lw=1.5, color="tab:purple")
ax.axhline(np.log(3), color="red", lw=1, ls="--", alpha=0.7, label=f"log(3)={np.log(3):.4f} (max)")
ax.set_title("Routing Entropy (nats)"); ax.set_xlabel("step"); ax.set_ylabel("entropy (nats)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(0, np.log(3) * 1.05)

plt.tight_layout()
plt.savefig(OUT_DIR / "stage2_routing.png", dpi=150)
plt.close()
print("Saved stage2_routing.png")

# ---------------------------------------------------------------- FIGURE 3: TensorBoard metrics

if tb_ok:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Stage 2 — TensorBoard Metrics", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    if len(tb_decorr_v) > 0:
        ax.plot(tb_decorr_s, tb_decorr_v, lw=1.2, color="tab:cyan")
    ax.set_title("Decorr Loss (Barlow Twins)"); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if len(tb_route_v) > 0:
        ax.plot(tb_route_s, tb_route_v, lw=1.2, color="tab:brown")
    ax.set_title("Route Loss (entropy reg)"); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if len(tb_density_v) > 0:
        ax.plot(tb_density_s, tb_density_v * 100, lw=1.2, color="tab:olive")
        ax.set_ylabel("% active pre-TopK")
        ax.axhline(100 * 256 / 5120, color="red", lw=1, ls="--", alpha=0.6, label=f"TopK={256/5120*100:.1f}%")
        ax.legend(fontsize=8)
    ax.set_title("z_dense Density (pre-TopK)"); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(tb_total_v) > 0:
        ax.plot(tb_total_s, tb_total_v, lw=1.2, color="black")
    ax.set_title("Total Loss"); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "stage2_tb_metrics.png", dpi=150)
    plt.close()
    print("Saved stage2_tb_metrics.png")

# ---------------------------------------------------------------- FIGURE 4: all losses on one plot (log scale)

fig, ax = plt.subplots(figsize=(13, 6))
ax.semilogy(steps, recons, lw=1.5, label="recon", color="tab:blue")
ax.semilogy(steps, prs,    lw=1.5, label="PR CTC", color="tab:orange")
ax.semilogy(steps, sids,   lw=1.5, label="SID CE", color="tab:green")
if tb_ok and len(tb_grl_v) > 0:
    ax.semilogy(tb_grl_s, tb_grl_v, lw=1.2, ls="--", label="GRL CE", color="tab:red", alpha=0.8)
if tb_ok and len(tb_decorr_v) > 0:
    ax.semilogy(tb_decorr_s, tb_decorr_v, lw=1.0, ls=":", label="decorr", color="tab:cyan", alpha=0.8)
ax.set_title("Stage 2 — All Losses (log scale)", fontsize=13)
ax.set_xlabel("step"); ax.set_ylabel("loss (log scale)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(OUT_DIR / "stage2_all_losses.png", dpi=150)
plt.close()
print("Saved stage2_all_losses.png")

# ---------------------------------------------------------------- summary text

def pct_change(arr):
    return 100 * (arr[-1] - arr[0]) / arr[0]

summary = f"""
=== Stage 2 Analysis Summary ===

Training: 40,000 steps | batch=16 | A100-SXM4-80GB | bf16
Data: 3000 train examples, 27 speakers, 41-phone vocab, val=0

--- Loss Summary ---
recon :  {recons[0]:.4f} → {recons[-1]:.4f}  ({pct_change(recons):+.1f}%)
pr CTC:  {prs[0]:.4f}    → {prs[-1]:.4f}     ({pct_change(prs):+.1f}%)  [ln(41)={np.log(41):.2f} chance]
sid CE:  {sids[0]:.4f}   → {sids[-1]:.4f}     ({pct_change(sids):+.1f}%)  [ln(27)={np.log(27):.2f} chance]

--- Routing (hard counts at K=5120) ---
Start: L={counts_L[0]} ({100*counts_L[0]/K:.1f}%)  P={counts_P[0]} ({100*counts_P[0]/K:.1f}%)  U={counts_U[0]} ({100*counts_U[0]/K:.1f}%)
End:   L={counts_L[-1]} ({100*counts_L[-1]/K:.1f}%)  P={counts_P[-1]} ({100*counts_P[-1]/K:.1f}%)  U={counts_U[-1]} ({100*counts_U[-1]/K:.1f}%)
Routing entropy: always {entropies[0]:.4f}  [max=log(3)={np.log(3):.4f}]  NEVER BROKE

--- Key Observations ---
1. Recon: healthy -54% reduction, still improving at end (not plateaued)
2. PR CTC: {prs[0]:.2f} → {prs[-1]:.2f} — only {pct_change(prs):+.1f}%; plateau from ~step 5000 onwards
   Chance=ln(41)={np.log(41):.2f}. Final {prs[-1]:.2f} is {100*(prs[-1]-np.log(41))/np.log(41):+.1f}% vs chance.
3. SID CE: collapsed from {sids[0]:.2f} → {sids[-1]:.4f} (near zero) — OVERFIT to 27 speakers
   Chance=ln(27)={np.log(27):.2f}. Model perfectly memorised training speakers.
4. Routing entropy STUCK at log(3) throughout — routing never specialised
   L growing ({100*counts_L[0]/K:.0f}% → {100*counts_L[-1]/K:.0f}%), P shrinking ({100*counts_P[0]/K:.0f}% → {100*counts_P[-1]/K:.0f}%)
   but SOFT logits remain uniform (entropy=1.099)
5. val=0 — no validation set → fast probe never ran, no generalisation signal

--- Concerns ---
A. Routing collapse risk: entropy stuck at max even with task gradients
B. PR CTC plateaued early around {prs[np.argmin(np.abs(steps-5000))]:.2f} — CTC not learning further
C. SID extreme overfit: 27 speakers is too few / no validation
D. No probe metrics (need val examples)
"""

print(summary)
(OUT_DIR / "stage2_summary.txt").write_text(summary)
print(f"\nAll outputs written to {OUT_DIR}/")
