#!/usr/bin/env python3
"""Generate supervisor-report figures from REAL experiment logs.
All numbers are transcribed from logs/ (probe runs, training trajectories,
diagnostic jobs).  Nothing is synthetic.
"""
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.7,
    "font.family": "DejaVu Sans",
})
FIG = Path(__file__).resolve().parent / "figs"; FIG.mkdir(exist_ok=True)

C_CONTENT = "#2a9d8f"   # teal  = linguistic / content
C_SPEAKER = "#e76f51"   # coral = speaker
C_DARK    = "#264653"
C_CHANCE  = "#8a8f98"
C_BLUE    = "#3a86ff"
C_GOOD    = "#43aa8b"
C_BAD     = "#e63946"
CHANCE_SID = 1/251      # 251 speakers

def tag(ax, txt, xy=(0.5, -0.30), color="#333"):
    ax.annotate(txt, xy=xy, xycoords="axes fraction", ha="center", va="top",
                fontsize=10.5, color=color, wrap=True,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f4f6f8", ec="#cdd3da"))

# ============================================================ FIG 1 : factorization scorecard
def fig1():
    srcs  = ["h_t\n(raw SPEAR)", "z_L\n(linguistic)", "z_P\n(paralinguistic)"]
    sid   = [1.000, 0.822, 0.960]      # speaker decodability (acc)
    per   = [0.068, 0.078, 0.578]      # content (phone error rate)
    x = np.arange(len(srcs))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.5))

    bars = a.bar(x, sid, color=[C_DARK, C_SPEAKER, C_CONTENT], width=0.6, edgecolor="white")
    a.axhline(CHANCE_SID, ls="--", lw=1.4, color=C_CHANCE)
    a.text(2.42, CHANCE_SID+0.02, "chance 0.004", color=C_CHANCE, ha="right", fontsize=9)
    for r, v in zip(bars, sid): a.text(r.get_x()+r.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")
    a.set_title("Speaker decodability  (SID accuracy ↑)")
    a.set_ylabel("speaker probe accuracy"); a.set_ylim(0, 1.08); a.set_xticks(x); a.set_xticklabels(srcs)
    a.text(1, 0.55, "speaker NOT\nremoved from z_L", color=C_BAD, ha="center", fontsize=10, fontweight="bold")

    bars = b.bar(x, per, color=[C_DARK, C_CONTENT, C_SPEAKER], width=0.6, edgecolor="white")
    for r, v in zip(bars, per): b.text(r.get_x()+r.get_width()/2, v+0.012, f"{v:.3f}", ha="center", fontweight="bold")
    b.set_title("Linguistic content  (phone error rate ↓)")
    b.set_ylabel("PER (lower = more content)"); b.set_ylim(0, 0.68); b.set_xticks(x); b.set_xticklabels(srcs)
    b.text(1, 0.20, "content well\npreserved in z_L", color=C_GOOD, ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("The factorization is 3/4 solved — the one failure is removing speaker from z_L",
                 fontsize=13.5, fontweight="bold", y=1.02)
    tag(a, "z_P captures speaker (0.96) and loses content (PER 0.58) → paralinguistic bucket works.",
        xy=(0.5, -0.16))
    tag(b, "z_L keeps content (PER 0.078 ≈ raw 0.068) — but ALSO keeps speaker (left).",
        xy=(0.5, -0.16))
    fig.tight_layout(); fig.savefig(FIG/"fig1_factorization.png", bbox_inches="tight"); plt.close(fig)

# ============================================================ FIG 2 : z_L->SID failure + probe curves
attn_steps = np.array([250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000,4250,4500,4750,5000,5250,5500,5750,6000,6250,6500,6750,7000,7250,7500,7750,8000,8250,8500,8750,9000,9250,9500,9750,10000])
attn_acc   = np.array([0.008,0.006,0.042,0.082,0.118,0.182,0.296,0.306,0.432,0.478,0.550,0.660,0.636,0.686,0.756,0.778,0.814,0.870,0.876,0.856,0.896,0.868,0.894,0.908,0.916,0.942,0.948,0.954,0.960,0.962,0.964,0.964,0.976,0.968,0.972,0.968,0.978,0.976,0.980,0.984])
dense_acc  = np.array([0.008,0.014,0.020,0.040,0.048,0.098,0.122,0.168,0.190,0.262,0.286,0.368,0.400,0.424,0.502,0.498,0.552,0.576,0.640,0.586,0.654,0.644,0.680,0.720,0.694,0.726,0.746,0.752,0.770,0.780,0.748,0.808,0.822,0.822,0.834,0.836,0.852,0.854,0.864,0.860])

def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={"width_ratios":[1, 1.15]})
    variants = ["no adversary\n(z_t)", "fixed-block\ndual-GRL", "dense\nper-frame", "attention\npool"]
    vals     = [1.000, 0.822, 0.848, 0.974]
    cols     = [C_DARK, C_BLUE, C_BLUE, C_BLUE]
    x = np.arange(len(variants))
    bars = a.bar(x, vals, color=cols, width=0.62, edgecolor="white")
    a.axhline(CHANCE_SID, ls="--", lw=1.4, color=C_CHANCE)
    a.text(3.4, CHANCE_SID+0.02, "chance 0.004", color=C_CHANCE, ha="right", fontsize=9)
    for r, v in zip(bars, vals): a.text(r.get_x()+r.get_width()/2, v+0.015, f"{v:.3f}", ha="center", fontweight="bold")
    a.set_title("Speaker survives EVERY adversarial variant")
    a.set_ylabel("z_L → SID test accuracy"); a.set_ylim(0, 1.1); a.set_xticks(x); a.set_xticklabels(variants, fontsize=9.5)

    b.plot(attn_steps, attn_acc, "-o", ms=3, color=C_SPEAKER, label="attention-pool  → 0.974")
    b.plot(attn_steps, dense_acc, "-o", ms=3, color=C_BLUE,   label="dense per-frame → 0.848")
    b.axhline(CHANCE_SID, ls="--", lw=1.4, color=C_CHANCE); b.text(9800, 0.03, "chance", color=C_CHANCE, ha="right", fontsize=9)
    b.set_title("A linear probe recovers speaker from z_L"); b.set_xlabel("probe training step")
    b.set_ylabel("dev SID accuracy"); b.set_ylim(0, 1.02); b.legend(loc="lower right", frameon=False, fontsize=9.5)

    fig.suptitle("z_L → Speaker-ID:  best adversary leaves 0.85–0.97 accuracy (chance = 0.004)",
                 fontsize=13.5, fontweight="bold", y=1.02)
    tag(a, "Mean-pool, frame-level, dense, attention — none approach chance.", xy=(0.5,-0.20))
    tag(b, "Not a weak probe: speaker is robustly, linearly decodable.", xy=(0.5,-0.16))
    fig.tight_layout(); fig.savefig(FIG/"fig2_zL_sid.png", bbox_inches="tight"); plt.close(fig)

# ============================================================ FIG 3 : WHY it fails (diagnostics)
def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.5))

    # (a) gradient conflict cos ~ 0
    a = axs[0]
    labels = ["recon vs\nremoval", "PR vs\nremoval"]
    cos = [-0.0006, 0.0006]; err = [0.0004, 0.0013]
    a.bar([0,1], cos, yerr=err, color=[C_CONTENT, C_BLUE], width=0.55, capsize=6, edgecolor="white")
    a.axhline(0, color="#333", lw=1)
    a.set_ylim(-0.05, 0.05); a.set_xticks([0,1]); a.set_xticklabels(labels)
    a.set_title("(a) Nothing defends speaker")
    a.set_ylabel("cosine(task-grad, removal-grad) on z_L")
    a.text(0.5, 0.035, "≈ 0  → orthogonal\nspeaker-free z_L is reachable", ha="center", color=C_GOOD, fontsize=9.5, fontweight="bold")

    # (b) gradient rank
    b = axs[1]
    names = ["speaker\n(attention)", "speaker\n(dense)", "phoneme\n(reference)"]
    eff   = [134.3, 26.8, 167.7]
    bars = b.bar(np.arange(3), eff, color=[C_SPEAKER, C_BLUE, C_CONTENT], width=0.6, edgecolor="white")
    for r,v in zip(bars, eff): b.text(r.get_x()+r.get_width()/2, v+3, f"{v:.0f}", ha="center", fontweight="bold")
    b.set_xticks(np.arange(3)); b.set_xticklabels(names)
    b.set_title("(b) Adversary gradient is high-rank"); b.set_ylabel("per-utterance effective rank")
    b.set_ylim(0, 185)
    b.text(1.0, 110, "attention ≈ phoneme\n→ rank is NOT the blocker", ha="center", color=C_DARK, fontsize=9.5, fontweight="bold")

    # (c) per-frame gradient dilution
    c = axs[2]
    g = ["grl  (z_L)\npooled→frame", "grl_p (z_P)\nper-frame"]
    val = [0.00006, 0.00034]
    bars = c.bar([0,1], val, color=[C_SPEAKER, C_CONTENT], width=0.55, edgecolor="white")
    for r,v in zip(bars, val): c.text(r.get_x()+r.get_width()/2, v+1e-5, f"{v:.5f}", ha="center", fontweight="bold", fontsize=9)
    c.set_xticks([0,1]); c.set_xticklabels(g)
    c.set_title("(c) Per-frame speaker signal is diluted")
    c.set_ylabel("mean |∂L/∂z[t]| per frame"); c.set_ylim(0, 0.00046)
    c.text(0.5, 0.00040, "~5× weaker per frame", ha="center", color=C_BAD, fontsize=10, fontweight="bold")

    fig.suptitle("Why adversarial removal fails — it is not recon, not gradient rank, not gradient strength",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(FIG/"fig3_why_fails.png", bbox_inches="tight"); plt.close(fig)

# ============================================================ FIG 4 : training-signal misleading + invariance
inv_step = np.array([1,100,200,300,400,500,600,700])
inv_val  = np.array([0.5519,0.4594,0.2981,0.1858,0.1544,0.1520,0.1380,0.1200])
inv_grl  = np.array([5.5176,5.7344,4.8691,4.4922,4.3877,3.8672,2.9600,3.4326])

def fig4():
    attn = np.loadtxt(Path(__file__).resolve().parent/"data/grl_attn.csv", delimiter=",")
    dense = np.loadtxt(Path(__file__).resolve().parent/"data/grl_dense.csv", delimiter=",")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    # (a) training-time grl is misleading vs probe
    a.plot(attn[:,0], attn[:,4], color=C_SPEAKER, lw=2, label="attention: grl loss → 0.78")
    a.plot(dense[:,0], dense[:,4], color=C_BLUE, lw=2, label="dense: grl loss ~5.2 (chance)")
    a.axhline(np.log(251), ls="--", lw=1.3, color=C_CHANCE); a.text(11800, np.log(251)+0.06, "chance ln251=5.52", color=C_CHANCE, ha="right", fontsize=8.5)
    a.set_title("(a) Training adversary loss does NOT predict the probe")
    a.set_xlabel("training step"); a.set_ylabel("speaker adversary CE (grl)"); a.legend(loc="center right", frameon=False, fontsize=9)
    a.text(6000, 1.4, "dense 'looks removed' (5.2)\nyet probe = 0.848", color=C_BLUE, fontsize=9, ha="center")
    a.text(6500, 4.6, "attention discriminator wins (0.78)\nyet probe = 0.974", color=C_SPEAKER, fontsize=9, ha="center")

    # (b) invariance smoke
    ax = b
    l1, = ax.plot(inv_step, inv_val, "-o", color=C_CONTENT, lw=2.2, label="invariance loss (z_L)")
    ax.set_xlabel("training step"); ax.set_ylabel("scale-normalized invariance loss", color=C_CONTENT)
    ax.tick_params(axis="y", labelcolor=C_CONTENT); ax.set_ylim(0, 0.6)
    ax2 = ax.twinx(); ax2.grid(False)
    l2, = ax2.plot(inv_step, inv_grl, "-s", color=C_SPEAKER, lw=2.2, label="speaker monitor (grl)")
    ax2.axhline(np.log(251), ls="--", lw=1.3, color=C_CHANCE)
    ax2.set_ylabel("speaker readability — grl CE (↑=less speaker)", color=C_SPEAKER)
    ax2.tick_params(axis="y", labelcolor=C_SPEAKER); ax2.set_ylim(0, 6)
    ax.set_title("(b) Invariance optimizes — but speaker stays")
    ax.legend(handles=[l1,l2], loc="center right", frameon=False, fontsize=9.5)
    ax.text(350, 0.34, "inv ↓ 4.6×\n(z_L invariant to pitch+formant)", color=C_CONTENT, fontsize=9, ha="center")
    ax2.text(420, 2.3, "monitor stays << chance\n→ speaker still readable", color=C_SPEAKER, fontsize=9, ha="center")

    fig.suptitle("Two cautionary results: the training signal lies, and pitch+formant invariance ≠ speaker-free",
                 fontsize=13.0, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(FIG/"fig4_signal_invariance.png", bbox_inches="tight"); plt.close(fig)

for f in (fig1, fig2, fig3, fig4):
    f(); print("done", f.__name__)
print("figures →", FIG)
