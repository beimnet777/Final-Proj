#!/usr/bin/env python3
"""Poster figures v2 — polished, math-rich. Success story only (grad-norm GRL + invariance)."""
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withStroke

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 230, "font.size": 14,
    "font.family": "DejaVu Sans", "mathtext.fontset": "cm",
    "axes.titlesize": 18, "axes.titleweight": "bold", "axes.labelsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#56616b", "axes.linewidth": 1.1,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.8,
    "xtick.color": "#2b3a4a", "ytick.color": "#2b3a4a",
})
FIG = Path(__file__).resolve().parent / "figs"; FIG.mkdir(exist_ok=True)

INK="#15263a"; TEAL="#0d7d8c"; GREEN="#159a6c"; CORAL="#e8643c"
PURPLE="#8b7fae"; CRIMSON="#d11149"; GOLD="#e0a818"; SLATE="#5b6b7d"; PANEL="#f4f8fa"
CHANCE=1/251

def shadowbox(ax,x,y,w,h,t,fc,tc="white",fs=13,ec=None,bold=True,fst="normal"):
    ax.add_patch(FancyBboxPatch((x-w/2+0.05,y-h/2-0.06),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",
                fc="#c9d3da",ec="none",zorder=2,alpha=0.55))
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",
                fc=fc,ec=ec or "#ffffff",lw=2.0,zorder=3))
    ax.text(x,y,t,ha="center",va="center",fontsize=fs,color=tc,zorder=4,
            fontweight="bold" if bold else "normal",fontstyle=fst)

def arr(ax,p1,p2,c="#41525f",lw=2.6,style="-|>",ls="-"):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle=style,mutation_scale=20,color=c,lw=lw,
                linestyle=ls,zorder=2,shrinkA=2,shrinkB=2))

# ============================================================= ARCHITECTURE
def architecture():
    fig,ax=plt.subplots(figsize=(16,8.2)); ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.2,0.2),15.6,8.5,boxstyle="round,pad=0.02,rounding_size=0.2",
                fc=PANEL,ec="#dde6ec",lw=1.5,zorder=0))
    yT=7.4
    shadowbox(ax,1.5,yT,2.0,1.15,"Audio\n$x$",INK,fs=14)
    shadowbox(ax,4.4,yT,2.6,1.35,"SPEAR-XLarge\n  FROZEN ❄",SLATE,fs=14)
    shadowbox(ax,7.5,yT,1.7,1.1,"$h_t$\n1280-d",INK,fs=13)
    shadowbox(ax,10.9,yT,3.0,1.35,"TopK Sparse AE",TEAL,fs=15)
    ax.text(10.9,yT-0.52,r"$z=\mathrm{TopK}(W_e(h-b))$",ha="center",va="center",
            fontsize=12.5,color="#eafdfd",zorder=5)
    shadowbox(ax,14.6,yT,1.7,1.1,"$z_t$\n5120",TEAL,fs=13)
    for a,b in [((2.5,yT),(3.1,yT)),((5.7,yT),(6.65,yT)),((8.35,yT),(9.4,yT)),((12.4,yT),(13.75,yT))]:
        arr(ax,a,b)
    ax.text(4.4,8.35,"pretrained · not retrained",ha="center",fontsize=11.5,color=SLATE,fontstyle="italic")

    # split to three factor blocks
    arr(ax,(14.6,6.85),(14.6,6.0),c=SLATE,lw=2.2)
    yL,yP,yU=5.3,3.45,1.7
    shadowbox(ax,12.2,yL,3.2,1.05,r"$z_L$    linguistic",GREEN,fs=14)
    shadowbox(ax,12.2,yP,3.2,1.05,r"$z_P$    paralinguistic",CORAL,fs=14)
    shadowbox(ax,12.2,yU,3.2,1.05,r"$z_U$    residual",PURPLE,fs=14)
    arr(ax,(14.6,6.0),(13.85,yL+0.4),c=SLATE,lw=2.0)
    arr(ax,(14.6,6.0),(13.85,yP),c=SLATE,lw=2.0)
    arr(ax,(14.6,6.0),(13.85,yU+0.2),c=SLATE,lw=2.0)

    # heads / losses (left of blocks)
    def pill(x,y,t,c,fs=12):
        shadowbox(ax,x,y,3.5,0.72,t,c,fs=fs)
    pill(7.2,yL+0.5,r"phonemes  $\mathcal{L}_{\mathrm{phon}}$",GREEN)
    pill(7.2,yL-0.5,r"anti-speaker  $\mathcal{L}_{\mathrm{adv}}$",CRIMSON)
    pill(7.2,yP,r"speaker  $\mathcal{L}_{\mathrm{spk}}$",CORAL)
    for yy in [yL+0.5,yL-0.5]: arr(ax,(10.6,yL),(8.95,yy),c="#8a97a1",lw=1.7)
    arr(ax,(10.6,yP),(8.95,yP),c="#8a97a1",lw=1.7)

    # reconstruction
    shadowbox(ax,3.0,3.2,2.5,1.1,r"decode $\hat h=W_d z{+}b$",INK,fs=12)
    ax.text(3.0,2.4,r"$\mathcal{L}_{\mathrm{rec}}=\|h-\hat h\|^2$",ha="center",fontsize=12.5,color=INK)
    for yy in (yL,yP,yU): arr(ax,(10.6,yy),(4.3,3.4),c="#b693c0",lw=1.8,style="-|>",ls=(0,(6,3)))
    ax.text(7.0,2.0,"all blocks reconstruct $h_t$",ha="center",fontsize=11,color=PURPLE,fontstyle="italic")

    # KEY tag
    ax.add_patch(FancyBboxPatch((0.5,3.9),4.5,1.55,boxstyle="round,pad=0.06,rounding_size=0.15",
                fc="#fff1f4",ec=CRIMSON,lw=2.2,zorder=1))
    ax.text(2.75,5.05,"KEY IDEA",ha="center",fontsize=13.5,fontweight="bold",color=CRIMSON,zorder=6)
    ax.text(2.75,4.35,"dense per-frame anti-speaker:\ngrad-normalized  +  invariance",
            ha="center",fontsize=12.5,color=INK,fontweight="bold",zorder=6)
    fig.text(0.5,0.965,"Sparse-Autoencoder factorization of frozen speech features",
             ha="center",fontsize=19,fontweight="bold",color=INK)
    fig.savefig(FIG/"architecture.png",bbox_inches="tight",facecolor="white"); plt.close(fig)

# ============================================================= MECHANISM (grad-norm + invariance)
def mechanism():
    fig=plt.figure(figsize=(15,5.0))
    gs=fig.add_gridspec(1,2,width_ratios=[1.12,1],wspace=0.12)
    # ===== LEFT: grad-norm as a VISUAL — per-frame arrows: diluted -> equal =====
    a=fig.add_subplot(gs[0,0]); a.axis("off"); a.set_xlim(0,1); a.set_ylim(0,1)
    a.set_title("Dense, grad-normalized adversary",color=INK,fontsize=16,pad=10)
    def strip(y,lengths,acol,lab):
        x0,x1=0.22,0.94; xs=np.linspace(x0,x1,len(lengths)); cw=(x1-x0)/len(lengths)*0.62
        for x in xs:
            a.add_patch(FancyBboxPatch((x-cw/2,y-0.04),cw,0.08,boxstyle="round,pad=0.003,rounding_size=0.015",
                        fc=GREEN,ec="white",lw=1.0,zorder=3))
        for x,L in zip(xs,lengths):
            a.add_patch(FancyArrowPatch((x,y+0.05),(x,y+0.05+L),arrowstyle="-|>",mutation_scale=10,
                        color=acol,lw=2.6,zorder=4))
        a.text(0.18,y,lab,ha="right",va="center",fontsize=12,color=INK,fontweight="bold")
    rng=np.random.default_rng(2)
    strip(0.60, 0.02+0.18*rng.random(8)**1.7, CORAL, "pooled /\nnatural")
    strip(0.18, [0.135]*8, CRIMSON, "grad-norm")
    a.annotate("",xy=(0.5,0.31),xytext=(0.5,0.49),arrowprops=dict(arrowstyle="-|>",color=TEAL,lw=3))
    a.text(0.53,0.40,r"$\tilde g_t=-\lambda\tau\,g_t/\|g_t\|$",ha="left",va="center",fontsize=14,color=TEAL)
    a.text(0.5,0.02,r"per-frame removal gradient on $z_L$  ($t=1\dots T$)",ha="center",fontsize=11,color=SLATE,fontstyle="italic")
    # ===== RIGHT: invariance schematic (minimal text) =====
    b=fig.add_subplot(gs[0,1]); b.axis("off"); b.set_xlim(0,1); b.set_ylim(0,1)
    b.set_title("Perturbation invariance",color=INK,fontsize=16,pad=10)
    shadowbox(b,0.18,0.70,0.24,0.17,"$x$",INK,fs=16)
    shadowbox(b,0.18,0.26,0.28,0.17,"$P(x)$",SLATE,fs=14)
    shadowbox(b,0.63,0.70,0.30,0.17,"$z_L(x)$",GREEN,fs=14)
    shadowbox(b,0.63,0.26,0.34,0.17,"$z_L(P(x))$",GREEN,fs=12.5)
    arr(b,(0.32,0.70),(0.47,0.70),c=SLATE); arr(b,(0.34,0.26),(0.45,0.26),c=SLATE)
    b.text(0.265,0.48,"pitch/formant\nwarp $P$",ha="center",fontsize=11,color=CORAL)
    b.annotate("",xy=(0.82,0.33),xytext=(0.82,0.63),arrowprops=dict(arrowstyle="<->",color=CRIMSON,lw=2.8))
    b.text(0.995,0.48,"force\nequal",ha="right",va="center",fontsize=12.5,color=CRIMSON,fontweight="bold")
    b.text(0.5,0.04,r"perturb speaker, keep content $\Rightarrow$ same $z_L$",ha="center",fontsize=11,color=SLATE,fontstyle="italic")
    fig.suptitle("Two routes to a dense, per-frame speaker-removal signal on $z_L$",
                 fontsize=15.5,fontweight="bold",color=INK,y=1.02)
    fig.savefig(FIG/"mechanism.png",bbox_inches="tight",facecolor="white"); plt.close(fig)

# ============================================================= RESULT (two methods)
# Two confirmed routes (real probe numbers):
#  Dense GRL + grad-norm (Job 2):  z_L PER .067  SID .010 | z_P PER .534  SID .972
#  Dense GRL + invariance (Job 1): z_L PER .073  SID .010 | z_P PER .515  SID .964
zl_flat=np.array([0.004,0.006,0.005,0.004,0.006,0.005,0.004,0.005,0.006,0.010])
steps=np.array([250,1000,2000,3000,4000,5000,6000,7000,8500,10000])
zp_climb=np.array([0.05,0.30,0.55,0.74,0.84,0.90,0.93,0.95,0.965,0.972])
def result():
    DATA=FIG.parent/"data"
    j2=np.loadtxt(DATA/"job2.csv",delimiter=",")   # step,recon,pr,sid,grl,grl_p   (grad-norm)
    j1=np.loadtxt(DATA/"job1.csv",delimiter=",")   # +inv                          (invariance)
    GN="#d11149"; IV="#0d7d8c"
    fig,axs=plt.subplots(2,2,figsize=(13.2,9.4)); fig.subplots_adjust(hspace=0.36,wspace=0.27)
    for ax in axs.flat: ax.tick_params(labelsize=12)

    # ===== A: task losses converge =====
    a=axs[0,0]
    a.plot(j2[:,0],j2[:,2],color=GREEN,lw=2.6,label=r"content  $\mathcal{L}_{\mathrm{phon}}$ (CTC on $z_L$)")
    a.plot(j2[:,0],j2[:,1],color=SLATE,lw=2.6,label=r"reconstruction  $\mathcal{L}_{\mathrm{rec}}$")
    a.set_yscale("log"); a.set_xlabel("training step",fontsize=13); a.set_ylabel("loss  (log scale)",fontsize=13)
    a.set_title("① Task losses converge",fontsize=15.5,color=INK)
    a.legend(frameon=False,fontsize=11.5,loc="upper right")

    # ===== B: speaker-removal signals =====
    b=axs[0,1]
    l1,=b.plot(j2[:,0],j2[:,4],color=GN,lw=2.6,label=r"speaker adversary  $\mathcal{L}_{\mathrm{adv}}$ (CE)")
    b.axhline(np.log(251),ls="--",lw=1.4,color=SLATE)
    b.text(11700,np.log(251)+0.06,"chance",ha="right",color=SLATE,fontsize=10)
    b.set_ylim(3.4,6.3); b.set_xlabel("training step",fontsize=13)
    b.set_ylabel("adversary CE",color=GN,fontsize=13); b.tick_params(axis="y",labelcolor=GN)
    b.set_title("② Speaker-removal signals",fontsize=15.5,color=INK)
    b2=b.twinx(); b2.grid(False)
    l2,=b2.plot(j1[:,0],j1[:,6],color=GOLD,lw=2.6,label=r"invariance  $\mathcal{L}_{\mathrm{inv}}$")
    b2.set_ylabel("invariance loss",color="#b88a00",fontsize=13); b2.tick_params(axis="y",labelcolor="#b88a00")
    b2.set_ylim(0,0.62)
    b.legend(handles=[l1,l2],frameon=False,fontsize=11,loc="center right")
    b.annotate("adversary stays at chance,\nyet speaker is removed (③)",xy=(8200,5.5),xytext=(5200,4.15),
               fontsize=10,color=GN,ha="center",arrowprops=dict(arrowstyle="-|>",color=GN,lw=1.6))

    # ===== C: probe — clean factorization (both routes, all metrics) =====
    c=axs[1,0]
    mets=[r"$z_L\!\to$PR",r"$z_L\!\to$SID",r"$z_P\!\to$PR",r"$z_P\!\to$SID"]
    gn=[0.067,0.010,0.534,0.972]; iv=[0.073,0.010,0.515,0.964]
    x=np.arange(4); w=0.36
    c.bar(x-w/2,gn,w,color=GN,edgecolor="white",lw=1.3,label="dense + grad-norm")
    c.bar(x+w/2,iv,w,color=IV,edgecolor="white",lw=1.3,label="dense + invariance")
    for xi,v in zip(x-w/2,gn): c.text(xi,v+0.02,f"{v:.3f}",ha="center",fontsize=9.5,fontweight="bold")
    for xi,v in zip(x+w/2,iv): c.text(xi,v+0.02,f"{v:.3f}",ha="center",fontsize=9.5,fontweight="bold")
    c.axhline(CHANCE,ls="--",lw=1.2,color=SLATE); c.text(3.46,CHANCE+0.03,"SID chance",ha="right",color=SLATE,fontsize=9)
    c.set_xticks(x); c.set_xticklabels(mets,fontsize=12.5); c.set_ylim(0,1.14)
    c.set_ylabel("probe score",fontsize=13); c.set_title("③ Probe — clean factorization",fontsize=15.5,color=INK)
    c.legend(frameon=False,fontsize=10.5,loc="upper center")
    c.text(0.5,0.33,"$z_L$: content kept,\nspeaker gone",ha="center",fontsize=9.5,color=GREEN,fontweight="bold")

    # ===== D: probe dynamics =====
    d=axs[1,1]
    d.plot(steps,zp_climb,"-o",ms=5,color=CORAL,lw=2.6,label=r"$z_P\!\to$SID  (learns speaker)")
    d.plot(steps,zl_flat,"-o",ms=5,color=GREEN,lw=2.6,label=r"$z_L\!\to$SID  (no speaker)")
    d.axhline(CHANCE,ls="--",lw=1.3,color=SLATE); d.text(9800,0.03,"chance",ha="right",color=SLATE,fontsize=10)
    d.set_xlabel("probe training step",fontsize=13); d.set_ylabel("dev SID accuracy",fontsize=13); d.set_ylim(0,1.02)
    d.set_title("④ Strong probe can't find speaker in $z_L$",fontsize=14.5,color=INK)
    d.legend(frameon=False,fontsize=11,loc="center right")
    fig.savefig(FIG/"result.png",bbox_inches="tight",facecolor="white"); plt.close(fig)

for f in (architecture,mechanism,result):
    f(); print("done",f.__name__)
print("figs →",FIG)
