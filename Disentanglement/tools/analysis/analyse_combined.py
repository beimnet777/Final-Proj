#!/usr/bin/env python3
"""Deep combined analysis of Stage 1 + Stage 2."""

from __future__ import annotations
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import numpy as np

DIS_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = DIS_DIR / "analysis" / "combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────── parse logs

def _parse_stage1(path):
    pat = re.compile(
        r"step\s+(\d+)/\d+\s+recon=([\d.]+)\s+route_H=([\d.]+)\s+"
        r"L/P/U=(\d+)/(\d+)/(\d+)"
    )
    steps, recons, ents, cL, cP, cU = [], [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)));    recons.append(float(m.group(2)))
            ents.append(float(m.group(3)));   cL.append(int(m.group(4)))
            cP.append(int(m.group(5)));       cU.append(int(m.group(6)))
    return (np.array(steps), np.array(recons), np.array(ents),
            np.array(cL), np.array(cP), np.array(cU))

def _parse_stage2(path):
    pat = re.compile(
        r"step\s+(\d+)/\d+\s+recon=([\d.]+)\s+pr=([\d.]+)\s+sid=([\d.]+)\s+"
        r"route_H=([\d.]+)\s+L/P/U=(\d+)/(\d+)/(\d+)"
    )
    steps, recons, prs, sids, ents, cL, cP, cU = [], [], [], [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)));  recons.append(float(m.group(2)))
            prs.append(float(m.group(3)));  sids.append(float(m.group(4)))
            ents.append(float(m.group(5))); cL.append(int(m.group(6)))
            cP.append(int(m.group(7)));     cU.append(int(m.group(8)))
    return (np.array(steps), np.array(recons), np.array(prs), np.array(sids),
            np.array(ents), np.array(cL), np.array(cP), np.array(cU))

s1_steps, s1_recon, s1_ent, s1_cL, s1_cP, s1_cU = _parse_stage1(
    DIS_DIR / "logs" / "train" / "stage1" / "sae_29858328.out")
s2_steps, s2_recon, s2_pr, s2_sid, s2_ent, s2_cL, s2_cP, s2_cU = _parse_stage2(
    DIS_DIR / "logs" / "train" / "stage2" / "sweep" / "baseline_29880935.out")

K = 5120

# offset stage2 x-axis so it continues from stage1
s2_off = s1_steps[-1] + s2_steps   # global step

# ──────────────────────────────────────────── TB

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def _tb(run_dir, tag):
    ea = EventAccumulator(str(run_dir)); ea.Reload()
    tags = ea.Tags()["scalars"]
    if tag not in tags:
        return np.array([]), np.array([])
    evts = ea.Scalars(tag)
    return np.array([e.step for e in evts]), np.array([e.value for e in evts])

# stage 1 TB
S1_TB_DIR = DIS_DIR / "runs" / "tb" / "stage1_20260525_203848"
S2_TB_DIR = DIS_DIR / "runs" / "tb" / "stage2_20260528_011238"

s1_dec_s, s1_dec_v   = _tb(S1_TB_DIR, "train/decorr")
s1_tot_s, s1_tot_v   = _tb(S1_TB_DIR, "train/total")
s1_den_s, s1_den_v   = _tb(S1_TB_DIR, "sae/z_dense_density")

s1_lw = []
for i in range(13):
    _, v = _tb(S1_TB_DIR, f"layer_weights/layer_{i:02d}")
    s1_lw.append(v)

# stage 2 TB
s2_dec_s, s2_dec_v   = _tb(S2_TB_DIR, "train/decorr")
s2_grl_s, s2_grl_v   = _tb(S2_TB_DIR, "train/grl")
s2_den_s, s2_den_v   = _tb(S2_TB_DIR, "sae/z_dense_density")

s2_lw = []
for i in range(13):
    _, v = _tb(S2_TB_DIR, f"layer_weights/layer_{i:02d}")
    s2_lw.append(v)

# ──────────────────────────────────────────── FIGURE 1: recon full trajectory

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Reconstruction Loss — Full Training Trajectory", fontsize=13, fontweight="bold")

# left: both stages on continuous x-axis
ax = axes[0]
ax.plot(s1_steps, s1_recon, lw=1.8, color="tab:blue",   label="Stage 1 (recon only)")
ax.plot(s2_off,   s2_recon, lw=1.8, color="tab:orange", label="Stage 2 (full obj)")
ax.axvline(s1_steps[-1], color="gray", lw=1.2, ls="--", label="S1→S2 boundary")
ax.set_xlabel("global training step"); ax.set_ylabel("Reconstruction MSE")
ax.set_title("Continuous trajectory (S1+S2)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# right: log scale to see stage2 dynamics
ax = axes[1]
ax.semilogy(s1_steps, s1_recon, lw=1.8, color="tab:blue",   label="Stage 1")
ax.semilogy(s2_off,   s2_recon, lw=1.8, color="tab:orange", label="Stage 2")
ax.axvline(s1_steps[-1], color="gray", lw=1.2, ls="--")
ax.set_xlabel("global training step"); ax.set_ylabel("Reconstruction MSE (log)")
ax.set_title("Log scale — rate of improvement"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig1_recon_trajectory.png", dpi=150)
plt.close()
print("Saved fig1_recon_trajectory.png")

# ──────────────────────────────────────────── FIGURE 2: routing dynamics both stages

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("Routing Dynamics — Stage 1 & Stage 2", fontsize=13, fontweight="bold")

# top-left: entropy both stages
ax = axes[0, 0]
ax.plot(s1_steps, s1_ent,      lw=1.5, color="tab:blue",   label="Stage 1")
ax.plot(s2_off,   s2_ent,      lw=1.5, color="tab:orange", label="Stage 2")
ax.axhline(np.log(3), color="red", lw=1, ls="--", label=f"max = ln(3) = {np.log(3):.4f}")
ax.axvline(s1_steps[-1], color="gray", lw=1.2, ls="--")
ax.set_title("Routing Entropy (nats)"); ax.set_xlabel("global step")
ax.set_ylim(0, np.log(3)*1.08); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# top-right: P fraction — most sensitive group
ax = axes[0, 1]
ax.plot(s1_steps, s1_cP / K,   lw=1.5, color="tab:blue",   label="Stage 1")
ax.plot(s2_off,   s2_cP / K,   lw=1.5, color="tab:orange", label="Stage 2")
ax.axvline(s1_steps[-1], color="gray", lw=1.2, ls="--")
ax.set_title("P (paralinguistic) fraction of K"); ax.set_xlabel("global step")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# bottom-left: stack plot stage 1
ax = axes[1, 0]
ax.stackplot(s1_steps, s1_cL/K, s1_cP/K, s1_cU/K,
             labels=["L", "P", "U"], colors=["tab:blue","tab:orange","tab:green"], alpha=0.7)
ax.set_title("Stage 1 Routing Split"); ax.set_xlabel("step")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.2)

# bottom-right: stack plot stage 2
ax = axes[1, 1]
ax.stackplot(s2_steps, s2_cL/K, s2_cP/K, s2_cU/K,
             labels=["L", "P", "U"], colors=["tab:blue","tab:orange","tab:green"], alpha=0.7)
ax.set_title("Stage 2 Routing Split"); ax.set_xlabel("step (within stage 2)")
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig2_routing_dynamics.png", dpi=150)
plt.close()
print("Saved fig2_routing_dynamics.png")

# ──────────────────────────────────────────── FIGURE 3: Stage 2 disentanglement signal

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle("Stage 2 — Disentanglement Signal", fontsize=13, fontweight="bold")

ax = axes[0, 0]
ax.plot(s2_steps, s2_pr, lw=1.5, color="tab:orange")
ax.axhline(np.log(41), color="red", lw=1.2, ls="--", alpha=0.7, label=f"chance = ln(41) = {np.log(41):.2f}")
ax.set_title("PR CTC Loss (z_L → phones)"); ax.set_xlabel("step"); ax.set_ylabel("CTC loss")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
# annotate plateau region
ax.axvspan(5000, 40000, alpha=0.07, color="red", label="plateau zone")

ax = axes[0, 1]
ax.semilogy(s2_steps, s2_sid, lw=1.5, color="tab:green")
ax.axhline(np.log(27), color="red", lw=1.2, ls="--", alpha=0.7, label=f"chance = ln(27) = {np.log(27):.2f}")
ax.set_title("SID CE Loss (z̄_P → speaker)"); ax.set_xlabel("step"); ax.set_ylabel("CE (log)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
if len(s2_grl_v) > 0:
    ax.plot(s2_grl_s, s2_grl_v, lw=1.5, color="tab:red")
    ax.axhline(np.log(27), color="green", lw=1.2, ls="--", alpha=0.8,
               label=f"target (chance) = ln(27) = {np.log(27):.2f}")
    ax.set_ylim(0, 4)
    ax.legend(fontsize=9)
ax.set_title("GRL CE Loss (z_L ↛ speaker  = good if ≈ ln(27))"); ax.set_xlabel("step")
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
if len(s2_den_v) > 0:
    ax.plot(s2_den_s, s2_den_v * 100, lw=1.5, color="tab:purple")
    ax.axhline(100 * 256 / K, color="red", lw=1.2, ls="--", alpha=0.7,
               label=f"TopK budget = {100*256/K:.1f}%")
if len(s1_den_v) > 0:
    # show stage 1 density inline for comparison
    ax.axhline(s1_den_v[0]*100, color="gray", lw=1, ls=":", alpha=0.6, label=f"S1 start={s1_den_v[0]*100:.1f}%")
ax.set_title("z_dense Density (pre-TopK active %)"); ax.set_xlabel("step"); ax.set_ylabel("% active")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig3_stage2_disentanglement.png", dpi=150)
plt.close()
print("Saved fig3_stage2_disentanglement.png")

# ──────────────────────────────────────────── FIGURE 4: layer weights evolution

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("SPEAR Layer Weight Evolution (Softmax mix)", fontsize=13, fontweight="bold")

layer_labels = [f"L{i}" for i in range(13)]
colors = plt.cm.tab20(np.linspace(0, 1, 13))

for i in range(13):
    if len(s1_lw[i]):
        lw_steps = np.linspace(1, s1_steps[-1], len(s1_lw[i]))
        axes[0].plot(lw_steps, s1_lw[i], lw=1.2, color=colors[i], label=layer_labels[i])
axes[0].set_title("Stage 1 — Layer Weights over Time"); axes[0].set_xlabel("step")
axes[0].set_ylabel("softmax weight"); axes[0].legend(fontsize=7, ncol=2); axes[0].grid(True, alpha=0.3)

for i in range(13):
    if len(s2_lw[i]):
        lw_steps = np.linspace(1, s2_steps[-1], len(s2_lw[i]))
        axes[1].plot(lw_steps, s2_lw[i], lw=1.2, color=colors[i], label=layer_labels[i])
axes[1].set_title("Stage 2 — Layer Weights over Time"); axes[1].set_xlabel("step")
axes[1].set_ylabel("softmax weight"); axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig4_layer_weights.png", dpi=150)
plt.close()
print("Saved fig4_layer_weights.png")

# ──────────────────────────────────────────── FIGURE 5: decorr both stages

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Decorr (Barlow Twins) Loss — Both Stages", fontsize=13, fontweight="bold")

ax = axes[0]
if len(s1_dec_v):
    ax.plot(s1_dec_s, s1_dec_v, lw=1.5, color="tab:blue")
ax.set_title("Stage 1 Decorr"); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)

ax = axes[1]
if len(s2_dec_v):
    ax.plot(s2_dec_s, s2_dec_v, lw=1.5, color="tab:orange")
ax.set_title("Stage 2 Decorr"); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig5_decorr.png", dpi=150)
plt.close()
print("Saved fig5_decorr.png")

# ──────────────────────────────────────────── FIGURE 6: recon per-phase rate

# compute per-100-step improvement rate
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("Recon Improvement Rate (Δrecon per 100 steps)", fontsize=13)
dr1 = -np.diff(s1_recon)
dr2 = -np.diff(s2_recon)
ax.plot(s1_steps[1:], dr1, lw=1.2, color="tab:blue",   alpha=0.8, label="Stage 1")
ax.plot(s2_off[1:],   dr2, lw=1.2, color="tab:orange", alpha=0.8, label="Stage 2")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(s1_steps[-1], color="gray", lw=1.2, ls="--", label="S1→S2 boundary")
ax.set_xlabel("global step"); ax.set_ylabel("Δrecon (positive = improving)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "fig6_recon_rate.png", dpi=150)
plt.close()
print("Saved fig6_recon_rate.png")

# ──────────────────────────────────────────── final bar chart: end-state summary

fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("End-State Summary vs Baselines", fontsize=13, fontweight="bold")

metrics = ["recon\n(S1 end)", "recon\n(S2 end)", "PR CTC\n(S2 end)", "SID CE\n(S2 end)", "GRL CE\n(S2 end)"]
values  = [s1_recon[-1], s2_recon[-1], s2_pr[-1], s2_sid[-1],
           s2_grl_v[-1] if len(s2_grl_v) else np.log(27)]
baselines = [None, None, np.log(41), np.log(27), np.log(27)]
colors_bar = ["tab:blue","tab:orange","tab:orange","tab:green","tab:red"]
x = np.arange(len(metrics))
bars = ax.bar(x, values, color=colors_bar, alpha=0.75, width=0.5)
for xi, (v, b) in enumerate(zip(values, baselines)):
    if b is not None:
        ax.hlines(b, xi-0.3, xi+0.3, color="red", lw=2, ls="--")
        ax.text(xi+0.33, b, f"chance={b:.2f}", va="center", fontsize=8, color="red")
    ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel("loss value"); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig7_end_state_summary.png", dpi=150)
plt.close()
print("Saved fig7_end_state_summary.png")

# ──────────────────────────────────────────── print numerical summary

def pct(a, b): return 100*(b-a)/a

print("\n" + "="*60)
print("DEEP ANALYSIS — STAGE 1 + STAGE 2 COMBINED")
print("="*60)

print(f"""
STAGE 1  (10,000 steps, recon+decorr+route only)
  Recon:       {s1_recon[0]:.4f} → {s1_recon[-1]:.4f}   ({pct(s1_recon[0],s1_recon[-1]):+.1f}%)
  Route H:     {s1_ent[0]:.4f} → {s1_ent[-1]:.4f}   [max=log(3)={np.log(3):.4f}]
  L/P/U start: {s1_cL[0]}/{s1_cP[0]}/{s1_cU[0]}  ({100*s1_cL[0]/K:.0f}%/{100*s1_cP[0]/K:.0f}%/{100*s1_cU[0]/K:.0f}%)
  L/P/U end:   {s1_cL[-1]}/{s1_cP[-1]}/{s1_cU[-1]}  ({100*s1_cL[-1]/K:.0f}%/{100*s1_cP[-1]/K:.0f}%/{100*s1_cU[-1]/K:.0f}%)
  P lost:      {s1_cP[0]-s1_cP[-1]} features  →  P shrinking even without task signal

STAGE 2  (40,000 steps, full objective)
  Recon:       {s2_recon[0]:.4f} → {s2_recon[-1]:.4f}   ({pct(s2_recon[0],s2_recon[-1]):+.1f}%)
  PR CTC:      {s2_pr[0]:.4f} → {s2_pr[-1]:.4f}   (chance={np.log(41):.2f})
  SID CE:      {s2_sid[0]:.4f} → {s2_sid[-1]:.4f}   (chance={np.log(27):.2f})
  GRL CE:      {s2_grl_v[0] if len(s2_grl_v) else 'N/A':.4f} → {s2_grl_v[-1] if len(s2_grl_v) else 'N/A':.4f}  (want ≈ {np.log(27):.2f})
  Route H:     {s2_ent[0]:.4f} → {s2_ent[-1]:.4f}   NEVER BROKE
  z_dense:     start≈1.3%  end≈0.6%  [TopK budget={100*256/K:.1f}%]

ROUTING COLLAPSE DIAGNOSIS
  Both stages: entropy pinned at log(3) = {np.log(3):.4f}
  Hard counts shift (S1): uniform 33/33/34% → 38/23/38%
  Hard counts shift (S2): 38/23/38% → 48/18/34%
  P-group keeps losing features: {s1_cP[0]} (S1 start) → {s1_cP[-1]} (S1 end) → {s2_cP[-1]} (S2 end)
  Root cause: rho * route_loss = rho * (-H) holds logits at uniform;
              task gradients too weak to overcome entropy regulariser.

SPEAR LAYER PREFERENCE (both stages stable)
  Dominant layers: L01 ({0.140:.3f}), L10 ({0.158:.3f})
  Near-zero: L07 ({0.030:.3f}), L08 ({0.029:.3f})
  Pattern: early acoustic (L0-L1) + late contextual (L9-L10) dominant
  Middle layers (L3-L8) largely ignored — consistent across both stages

DISENTANGLEMENT QUALITY
  SID from z̄_P:  collapsed to 0.005 (perfect memorisation of 27 speakers)
  GRL from z_L:   stuck at ln(27) = 3.296 (z_L carries NO speaker info → good)
  PR from z_L:    3.20 vs chance 3.71 — only 13.8% below chance → poor
  Conclusion: speaker info IS separated (GRL good), but linguistic content
              in z_L is too sparse/noisy for CTC to decode (PR poor)
""")
