#!/usr/bin/env python3
"""Generate Week 6 report figures from experiment logs.

The figures are derived from the local .out logs used in week_6 report.md.
Outputs are written as high-resolution PNG and PDF files.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DIS = ROOT / "Disentanglement"
OUT = DIS / "report" / "figs" / "week6"

MULTITASK_LOGS = {
    "soft initial": DIS / "logs/train/stage2/inv_dense_prosody_emotion_8to1/invpem_31034976_0.out",
    "hard initial": DIS / "logs/train/stage2/inv_dense_prosody_emotion_8to1/invpem_31034976_1.out",
    "hard norm rerun": DIS / "logs/train/stage2/inv_dense_prosody_emotion_8to1/invpem_31043736_1.out",
    "soft norm rerun": DIS / "logs/train/stage2/inv_dense_prosody_emotion_8to1/invpem_31045826_0.out",
    "hard norm rerun 2": DIS / "logs/train/stage2/inv_dense_prosody_emotion_8to1/invpem_31045826_1.out",
}

DUAL_LOG = DIS / "logs/train/stage2/dual_inv_v1/dual_inv_v1_soft_nogrl_31006673.out"
ARCTIC_LOG = DIS / "logs/diag/v1_arctic_sid/probe_v1_arctic_31018841_0.out"

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
STEP_RE = re.compile(r"^\s*step\s+(\d+)/(\d+)\s+(.*)$")
VAL_RE = re.compile(
    rf"\[val\]\s+step=(\d+)\s+recon=({FLOAT})\s+pr=({FLOAT})\s+"
    rf"\|\s+z_L PR=({FLOAT}) SID=({FLOAT})\s+\|\s+z_P\s+(.*)$"
)
IEMO_RE = re.compile(
    rf"\[iemocap val\]\s+step=(\d+)\s+z_P emotion=({FLOAT})\s+z_L emotion=({FLOAT})"
)
GRAD_STEP_RE = re.compile(r"\[grad_norms @(\d+)\]")
GRAD_ROW_RE = re.compile(rf"^\s+([A-Za-z0-9_]+)\s+\|g\|=({FLOAT})\s+ratio=({FLOAT})x recon")
COS_STEP_RE = re.compile(r"\[grad_cos @(\d+)\]")
COS_ROW_RE = re.compile(rf"^\s+cos\(([A-Za-z0-9_]+)\s*\)\s*=\s*({FLOAT})")
ARCTIC_PROBE_RE = re.compile(rf"\[sid z_L\]\s+step\s+(\d+)/\d+\s+dev acc=({FLOAT})")


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        font_scale=1.05,
        rc={
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 12.5,
            "legend.frameon": False,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        },
    )


def fvalue(line: str, key: str) -> float | None:
    match = re.search(rf"\b{re.escape(key)}=({FLOAT})", line)
    return float(match.group(1)) if match else None


def count_triplet(line: str, label: str) -> tuple[int, int, int] | None:
    match = re.search(rf"{re.escape(label)}=(\d+)/(\d+)/(\d+)", line)
    if not match:
        return None
    return tuple(int(match.group(i)) for i in range(1, 4))


def parse_log(path: Path, run: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    emotion_rows: list[dict] = []
    grad_rows: list[dict] = []
    cos_rows: list[dict] = []
    current_grad_step: int | None = None
    current_cos_step: int | None = None

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()

        cos_step = COS_STEP_RE.search(line)
        if cos_step:
            current_cos_step = int(cos_step.group(1))
            continue

        cos_row = COS_ROW_RE.match(raw)
        if cos_row and current_cos_step is not None:
            cos_rows.append(
                {
                    "run": run,
                    "step": current_cos_step,
                    "pair": cos_row.group(1),
                    "cosine": float(cos_row.group(2)),
                }
            )
            continue

        grad_step = GRAD_STEP_RE.search(line)
        if grad_step:
            current_grad_step = int(grad_step.group(1))
            continue

        grad_row = GRAD_ROW_RE.match(raw)
        if grad_row and current_grad_step is not None:
            grad_rows.append(
                {
                    "run": run,
                    "step": current_grad_step,
                    "task": grad_row.group(1),
                    "grad_norm": float(grad_row.group(2)),
                    "ratio_to_recon": float(grad_row.group(3)),
                }
            )
            continue

        step_match = STEP_RE.match(raw)
        if step_match:
            step = int(step_match.group(1))
            row = {"run": run, "step": step}
            for key in ("recon", "pr", "sid", "grl", "grl_p", "pros", "inv", "inv_L", "inv_P", "var", "H", "Hu", "marg", "lstd", "lr"):
                value = fvalue(raw, key)
                if value is not None:
                    row[key] = value

            match = re.search(rf"grl=({FLOAT})\(acc=({FLOAT})\)", raw)
            if match:
                row["grl_acc"] = float(match.group(2))
            match = re.search(rf"grl_p=({FLOAT})\(per=({FLOAT})\)", raw)
            if match:
                row["grl_p_per"] = float(match.group(2))
            match = re.search(rf"grlPr=({FLOAT})/({FLOAT})", raw)
            if match:
                row["grlPr_loss"] = float(match.group(1))
                row["grlPr_aux"] = float(match.group(2))
            match = re.search(rf"emo=({FLOAT})\(acc=({FLOAT})\)", raw)
            if match:
                row["emo_loss"] = float(match.group(1))
                row["emo_acc"] = float(match.group(2))
            match = re.search(rf"grlE=({FLOAT})\(acc=({FLOAT})\)", raw)
            if match:
                row["grlE_loss"] = float(match.group(1))
                row["grlE_acc"] = float(match.group(2))
            match = re.search(rf"emoAux=({FLOAT})x({FLOAT})", raw)
            if match:
                row["emo_aux_raw"] = float(match.group(1))
                row["emo_aux_scale"] = float(match.group(2))

            counts = count_triplet(raw, "L/P/U")
            if counts:
                row["route_L"], row["route_P"], row["route_U"] = counts
            counts = count_triplet(raw, "actL/P/U")
            if counts:
                row["active_L"], row["active_P"], row["active_U"] = counts
            match = re.search(rf"spec<\.5=({FLOAT})", raw)
            if match:
                row["spec_lt_05"] = float(match.group(1))
            match = re.search(rf"mix\[arc/pert\]=({FLOAT})/({FLOAT})", raw)
            if match:
                row["mix_arctic"] = float(match.group(1))
                row["mix_pert"] = float(match.group(2))

            train_rows.append(row)
            continue

        val_match = VAL_RE.search(raw)
        if val_match:
            z_p_tail = val_match.group(6)
            z_p_pr = fvalue(z_p_tail, "PR")
            z_p_sid = fvalue(z_p_tail, "SID")
            val_rows.append(
                {
                    "run": run,
                    "step": int(val_match.group(1)),
                    "val_recon": float(val_match.group(2)),
                    "val_pr_loss": float(val_match.group(3)),
                    "zL_PR_PER": float(val_match.group(4)),
                    "zL_SID_acc_proxy": float(val_match.group(5)),
                    "zP_PR_PER": z_p_pr,
                    "zP_SID_acc": z_p_sid,
                }
            )
            continue

        iemo_match = IEMO_RE.search(raw)
        if iemo_match:
            emotion_rows.append(
                {
                    "run": run,
                    "step": int(iemo_match.group(1)),
                    "zP_emotion_acc": float(iemo_match.group(2)),
                    "zL_emotion_acc": float(iemo_match.group(3)),
                }
            )

    return (
        pd.DataFrame(train_rows),
        pd.DataFrame(val_rows),
        pd.DataFrame(emotion_rows),
        pd.DataFrame(grad_rows),
        pd.DataFrame(cos_rows),
    )


def parse_arctic_probe(path: Path) -> pd.DataFrame:
    rows = []
    for raw in path.read_text(errors="replace").splitlines():
        match = ARCTIC_PROBE_RE.search(raw)
        if match:
            rows.append({"step": int(match.group(1)), "dev_acc": float(match.group(2))})
    return pd.DataFrame(rows)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white", dpi=320)
    plt.close(fig)


def clear_outputs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for pattern in ("week6_*.png", "week6_*.pdf"):
        for path in OUT.glob(pattern):
            path.unlink()


def lineplot_safe(*, data: pd.DataFrame, x: str, y: str, ax, **kwargs) -> None:
    if data.empty or y not in data:
        ax.text(0.5, 0.5, "No parsed data", ha="center", va="center", transform=ax.transAxes)
        return
    sns.lineplot(data=data.dropna(subset=[y]), x=x, y=y, ax=ax, **kwargs)


def plot_multitask_validation(val_df: pd.DataFrame, emo_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), constrained_layout=True)
    palette = sns.color_palette("colorblind", n_colors=max(5, val_df["run"].nunique()))

    per_long = val_df.melt(
        id_vars=["run", "step"],
        value_vars=["zL_PR_PER", "zP_PR_PER"],
        var_name="metric",
        value_name="value",
    ).dropna()
    per_long["metric"] = per_long["metric"].map({"zL_PR_PER": "z_L PR PER", "zP_PR_PER": "z_P PR PER"})
    sns.lineplot(data=per_long, x="step", y="value", hue="run", style="metric", markers=True, dashes=True, ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title("Multitask validation: phone content and phone leakage")
    axes[0, 0].set_ylabel("PER (lower = more phone information)")
    axes[0, 0].set_xlabel("training step")
    axes[0, 0].set_ylim(bottom=0)

    sid_long = val_df.melt(
        id_vars=["run", "step"],
        value_vars=["zL_SID_acc_proxy", "zP_SID_acc"],
        var_name="metric",
        value_name="value",
    ).dropna()
    sid_long["metric"] = sid_long["metric"].map({"zL_SID_acc_proxy": "z_L SID proxy", "zP_SID_acc": "z_P SID"})
    sns.lineplot(data=sid_long, x="step", y="value", hue="run", style="metric", markers=True, dashes=True, ax=axes[0, 1], palette=palette, legend=False)
    axes[0, 1].set_title("Multitask validation: speaker placement proxy")
    axes[0, 1].set_ylabel("accuracy")
    axes[0, 1].set_xlabel("training step")
    axes[0, 1].set_ylim(-0.02, 1.05)

    if not emo_df.empty:
        emo_long = emo_df.melt(
            id_vars=["run", "step"],
            value_vars=["zP_emotion_acc", "zL_emotion_acc"],
            var_name="metric",
            value_name="value",
        )
        emo_long["metric"] = emo_long["metric"].map({"zP_emotion_acc": "z_P emotion", "zL_emotion_acc": "z_L emotion"})
        sns.lineplot(data=emo_long, x="step", y="value", hue="run", style="metric", markers=True, dashes=True, ax=axes[1, 0], palette=palette, legend=False)
    axes[1, 0].set_title("IEMOCAP validation: emotion is not isolated")
    axes[1, 0].set_ylabel("emotion accuracy")
    axes[1, 0].set_xlabel("training step")
    axes[1, 0].set_ylim(0.2, 0.8)
    axes[1, 0].axhline(0.25, color="0.45", linestyle="--", linewidth=1.1)
    axes[1, 0].text(axes[1, 0].get_xlim()[1], 0.255, "4-way chance", ha="right", va="bottom", color="0.35")

    last = val_df.groupby("run", as_index=False)["step"].max().sort_values("step")
    sns.barplot(data=last, x="step", y="run", ax=axes[1, 1], color="#7aa6c2")
    axes[1, 1].axvline(12000, color="0.35", linestyle="--", linewidth=1.1)
    axes[1, 1].set_title("Available validation coverage")
    axes[1, 1].set_xlabel("last validation step in log")
    axes[1, 1].set_ylabel("")
    axes[1, 1].text(12000, -0.45, "planned final step", ha="right", va="bottom", color="0.30")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, bbox_to_anchor=(1.02, 1.02), loc="upper left", title="")
    fig.suptitle("Week 6 multitask runs are partial and show objective competition", y=1.02, fontsize=16, fontweight="bold")
    save(fig, "week6_multitask_validation")


def plot_multitask_losses(train_df: pd.DataFrame) -> None:
    metrics = [
        "recon",
        "pr",
        "sid",
        "grl",
        "grl_p",
        "pros",
        "grlPr_loss",
        "emo_loss",
        "grlE_loss",
        "inv",
    ]
    long = train_df.melt(id_vars=["run", "step"], value_vars=[m for m in metrics if m in train_df], var_name="loss", value_name="value").dropna()
    long = long[long["value"] > 0]
    long["loss"] = long["loss"].map(
        {
            "recon": "recon",
            "pr": "PR",
            "sid": "SID",
            "grl": "speaker GRL",
            "grl_p": "phone GRL",
            "pros": "prosody cls",
            "grlPr_loss": "prosody GRL",
            "emo_loss": "emotion cls",
            "grlE_loss": "emotion GRL",
            "inv": "invariance",
        }
    )
    g = sns.relplot(
        data=long,
        x="step",
        y="value",
        hue="loss",
        col="run",
        col_wrap=2,
        kind="line",
        height=3.0,
        aspect=1.45,
        linewidth=1.8,
        facet_kws={"sharey": False, "sharex": True},
    )
    g.set_axis_labels("training step", "loss value (log scale)")
    g.set_titles("{col_name}")
    max_step = float(train_df["step"].max())
    xticks = list(range(0, int(math.ceil(max_step / 500.0) * 500) + 1, 500))
    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.set_xlim(0, max_step * 1.05)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelbottom=True)
    g.figure.suptitle("Multitask loss trajectories before truncation", y=1.03, fontsize=15, fontweight="bold")
    save(g.figure, "week6_multitask_losses")


def plot_multitask_routing(train_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8), constrained_layout=True)
    max_step = float(train_df["step"].max())
    xticks = list(range(0, int(math.ceil(max_step / 500.0) * 500) + 1, 500))

    pre_counts = train_df.melt(
        id_vars=["run", "step"],
        value_vars=["route_L", "route_P"],
        var_name="bucket",
        value_name="count",
    ).dropna()
    pre_counts["bucket"] = pre_counts["bucket"].map({"route_L": "z_L preactivation", "route_P": "z_P preactivation"})
    sns.lineplot(data=pre_counts, x="step", y="count", hue="run", style="bucket", ax=axes[0, 0], linewidth=2.0)
    axes[0, 0].set_title("Preactivation route allocation")
    axes[0, 0].set_xlabel("training step")
    axes[0, 0].set_ylabel("route count (of 5120)")
    axes[0, 0].axhline(2560, color="0.45", linestyle="--", linewidth=1.0)
    axes[0, 0].text(max_step, 2585, "balanced split", ha="right", va="bottom", color="0.35")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend_.remove()
    fig.legend(handles, labels, title="", bbox_to_anchor=(1.01, 0.98), loc="upper left")

    active_counts = train_df.melt(
        id_vars=["run", "step"],
        value_vars=["active_L", "active_P"],
        var_name="bucket",
        value_name="count",
    ).dropna()
    active_counts["bucket"] = active_counts["bucket"].map({"active_L": "z_L post-activation", "active_P": "z_P post-activation"})
    sns.lineplot(data=active_counts, x="step", y="count", hue="run", style="bucket", ax=axes[0, 1], linewidth=2.0, legend=False)
    axes[0, 1].set_title("Post-activation active route counts")
    axes[0, 1].set_xlabel("training step")
    axes[0, 1].set_ylabel("active count (of 256)")
    axes[0, 1].axhline(128, color="0.45", linestyle="--", linewidth=1.0)
    axes[0, 1].text(max_step, 130, "balanced active split", ha="right", va="bottom", color="0.35")

    share_rows = []
    for _, row in train_df.iterrows():
        pre_total = row.get("route_L", math.nan) + row.get("route_P", math.nan)
        active_total = row.get("active_L", math.nan) + row.get("active_P", math.nan)
        if pre_total and not math.isnan(pre_total):
            share_rows.append({"run": row["run"], "step": row["step"], "level": "preactivation", "zP_share": row.get("route_P", math.nan) / pre_total})
        if active_total and not math.isnan(active_total):
            share_rows.append({"run": row["run"], "step": row["step"], "level": "post-activation", "zP_share": row.get("active_P", math.nan) / active_total})
    share = pd.DataFrame(share_rows)
    sns.lineplot(data=share, x="step", y="zP_share", hue="run", style="level", ax=axes[1, 0], linewidth=2.0, legend=False)
    axes[1, 0].set_title("z_P share separates routing drift from active usage")
    axes[1, 0].set_xlabel("training step")
    axes[1, 0].set_ylabel("fraction assigned to z_P")
    axes[1, 0].set_ylim(0.15, 0.75)
    axes[1, 0].axhline(0.5, color="0.45", linestyle="--", linewidth=1.0)
    axes[1, 0].legend(
        handles=[
            Line2D([0], [0], color="0.25", linewidth=2.0, label="preactivation"),
            Line2D([0], [0], color="0.25", linewidth=2.0, linestyle="--", label="post-activation"),
        ],
        title="",
        loc="upper left",
    )

    diag_cols = ["Hu", "spec_lt_05", "marg", "lstd"]
    diag = train_df.melt(id_vars=["run", "step"], value_vars=[c for c in diag_cols if c in train_df], var_name="diagnostic", value_name="value").dropna()
    diag["diagnostic"] = diag["diagnostic"].map({"Hu": "unit entropy", "spec_lt_05": "specialized frac", "marg": "top1-top2 margin", "lstd": "logit std"})
    sns.lineplot(data=diag, x="step", y="value", hue="diagnostic", units="run", estimator=None, ax=axes[1, 1], linewidth=1.7, alpha=0.78)
    axes[1, 1].set_title("Routing specialization diagnostics")
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].set_ylabel("diagnostic value")
    axes[1, 1].legend(title="")

    for ax in axes.flat:
        ax.set_xlim(0, max_step * 1.05)
        ax.set_xticks(xticks)
    fig.suptitle("Multitask learned-routing dynamics", y=1.02, fontsize=16, fontweight="bold")
    save(fig, "week6_multitask_routing")


def plot_multitask_gradients(grad_df: pd.DataFrame) -> None:
    keep = ["pr_weighted", "sid_weighted", "grl", "grl_p", "recon"]
    data = grad_df[grad_df["task"].isin(keep)].copy()
    g = sns.relplot(
        data=data,
        x="step",
        y="ratio_to_recon",
        hue="task",
        col="run",
        col_wrap=2,
        kind="line",
        marker="o",
        height=3.0,
        aspect=1.45,
        facet_kws={"sharey": False},
    )
    g.set_axis_labels("training step", "gradient norm / recon gradient")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    g.figure.suptitle("Multitask gradient ratios show adversarial/task imbalance", y=1.03, fontsize=15, fontweight="bold")
    save(g.figure, "week6_multitask_gradients")


def plot_multitask_gradient_conflicts(cos_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.6), constrained_layout=True)
    if cos_df.empty:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No parsed cosine data", ha="center", va="center", transform=ax.transAxes)
        save(fig, "week6_multitask_gradient_conflicts")
        return

    data = cos_df.copy()
    data["conflict"] = data["cosine"] < 0
    data["negative_cosine"] = data["cosine"].where(data["cosine"] < 0, 0.0)
    summary = (
        data.groupby(["run", "step"], as_index=False)
        .agg(
            conflict_count=("conflict", "sum"),
            min_cosine=("cosine", "min"),
            mean_negative_cosine=("negative_cosine", "mean"),
        )
    )

    sns.lineplot(data=summary, x="step", y="conflict_count", hue="run", marker="o", ax=axes[0, 0], linewidth=2.0)
    axes[0, 0].set_title("Number of negative pairwise cosines")
    axes[0, 0].set_xlabel("training step")
    axes[0, 0].set_ylabel("conflicting task pairs")
    axes[0, 0].set_ylim(bottom=-0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend_.remove()
    fig.legend(handles, labels, title="run", bbox_to_anchor=(1.01, 0.98), loc="upper left")

    sns.lineplot(data=summary, x="step", y="min_cosine", hue="run", marker="o", ax=axes[0, 1], linewidth=2.0, legend=False)
    axes[0, 1].axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Strongest conflict at each diagnostic step")
    axes[0, 1].set_xlabel("training step")
    axes[0, 1].set_ylabel("minimum cosine")

    key_pairs = ["recon_vs_grl", "recon_vs_grl_p", "pr_vs_grl_p", "sid_vs_grl", "sid_vs_grl_p"]
    pair_labels = {
        "recon_vs_grl": "recon-grl",
        "recon_vs_grl_p": "recon-grl_p",
        "pr_vs_grl_p": "pr-grl_p",
        "sid_vs_grl": "sid-grl",
        "sid_vs_grl_p": "sid-grl_p",
    }
    heat = data[data["pair"].isin(key_pairs)].copy()
    heat["pair"] = heat["pair"].map(pair_labels)
    mean_cos = heat.pivot_table(index="run", columns="pair", values="cosine", aggfunc="mean")
    conflict_rate = heat.pivot_table(index="run", columns="pair", values="conflict", aggfunc="mean")

    sns.heatmap(mean_cos, center=0, cmap="vlag", annot=True, fmt=".3f", linewidths=0.5, ax=axes[1, 0], cbar_kws={"label": "mean cosine"})
    axes[1, 0].set_title("Mean cosine for key competing objectives")
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("")

    sns.heatmap(conflict_rate, vmin=0, vmax=1, cmap="rocket_r", annot=True, fmt=".0%", linewidths=0.5, ax=axes[1, 1], cbar_kws={"label": "cos < 0 rate"})
    axes[1, 1].set_title("How often each key pair is in conflict")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("")

    for ax in axes[1]:
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Multitask gradient conflicts on the shared SAE encoder", y=1.03, fontsize=15, fontweight="bold")
    save(fig, "week6_multitask_gradient_conflicts")


def plot_multitask_aux_clipping(train_df: pd.DataFrame) -> None:
    data = train_df.dropna(subset=["emo_aux_raw", "emo_aux_scale"], how="all")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    if not data.empty:
        sns.lineplot(data=data, x="step", y="emo_aux_raw", hue="run", marker="o", ax=axes[0])
        sns.lineplot(data=data, x="step", y="emo_aux_scale", hue="run", marker="o", ax=axes[1], legend=False)
    axes[0].set_title("Raw emotion/prosody auxiliary loss")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("raw auxiliary loss before cap")
    axes[0].set_yscale("log")
    axes[1].set_title("Auxiliary clipping scale")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("scale applied to auxiliary loss")
    axes[1].set_ylim(0, 1.08)
    axes[1].axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    fig.suptitle("Emotion/prosody auxiliary terms often required clipping", y=1.04, fontsize=15, fontweight="bold")
    save(fig, "week6_multitask_aux_clipping")


def plot_dual_losses(train_df: pd.DataFrame) -> None:
    loss_cols = ["inv_L", "inv_P", "var"]
    long = train_df.melt(id_vars=["step"], value_vars=loss_cols, var_name="loss", value_name="value").dropna()
    first = long.sort_values("step").groupby("loss")["value"].first().rename("start")
    norm = long.join(first, on="loss")
    norm["normalized"] = norm["value"] / norm["start"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    sns.lineplot(data=long, x="step", y="value", hue="loss", ax=axes[0], linewidth=2.0)
    axes[0].set_title("Dual-invariance losses")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("raw loss value")
    sns.lineplot(data=norm, x="step", y="normalized", hue="loss", ax=axes[1], linewidth=2.0, legend=False)
    axes[1].set_title("Losses normalized to step 1")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("fraction of initial value")
    axes[1].axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    fig.suptitle("Dual-invariance v1 optimizes its losses but does not remove speaker", y=1.04, fontsize=15, fontweight="bold")
    save(fig, "week6_dual_inv_losses")


def plot_dual_routing(train_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), constrained_layout=True)
    counts = train_df.melt(id_vars=["step"], value_vars=["route_L", "route_P", "active_L", "active_P"], var_name="bucket", value_name="count").dropna()
    counts["bucket"] = counts["bucket"].map(
        {
            "route_L": "route L",
            "route_P": "route P",
            "active_L": "active L",
            "active_P": "active P",
        }
    )
    sns.lineplot(data=counts, x="step", y="count", hue="bucket", ax=axes[0], linewidth=2.0)
    axes[0].set_title("Dual-invariance route and active counts")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("count")
    diag = train_df.melt(id_vars=["step"], value_vars=["H", "Hu", "spec_lt_05", "marg"], var_name="diagnostic", value_name="value").dropna()
    diag["diagnostic"] = diag["diagnostic"].map({"H": "balance entropy", "Hu": "unit entropy", "spec_lt_05": "specialized frac", "marg": "top1-top2 margin"})
    sns.lineplot(data=diag, x="step", y="value", hue="diagnostic", ax=axes[1], linewidth=2.0)
    axes[1].set_title("Dual-invariance routing specialization")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("diagnostic value")
    fig.suptitle("Dual-invariance learned routing becomes specialized and balanced", y=1.02, fontsize=15, fontweight="bold")
    save(fig, "week6_dual_inv_routing")


def plot_dual_proxy_probe(arctic_df: pd.DataFrame) -> None:
    arctic_last = float(arctic_df["dev_acc"].iloc[-1]) if not arctic_df.empty else math.nan
    data = pd.DataFrame(
        [
            {"signal": "validation proxy\nLibri z_L SID", "acc": 0.002, "status": "proxy"},
            {"signal": "held-out probe\nLibri z_L SID", "acc": 1.000, "status": "completed"},
            {"signal": "matched probe\nARCTIC z_L SID", "acc": arctic_last, "status": "partial"},
        ]
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    sns.barplot(data=data, x="signal", y="acc", hue="status", dodge=False, ax=ax, palette=["#8fbcd4", "#d95f5f", "#f2b35e"])
    ax.axhline(1 / 251, color="0.35", linestyle="--", linewidth=1.0)
    ax.text(2.45, 1 / 251 + 0.015, "Libri chance 1/251", ha="right", color="0.35")
    ax.axhline(1 / 18, color="0.55", linestyle=":", linewidth=1.0)
    ax.text(2.45, 1 / 18 + 0.015, "ARCTIC chance 1/18", ha="right", color="0.45")
    for patch in ax.patches:
        if patch.get_height() >= 0:
            ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height() + 0.025, f"{patch.get_height():.3f}", ha="center", fontweight="bold")
    ax.set_title("Dual-invariance proxy/probe contradiction")
    ax.set_ylabel("SID accuracy")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.12)
    ax.legend(title="")
    save(fig, "week6_dual_proxy_vs_probe")


def plot_dual_pair_mix(train_df: pd.DataFrame) -> None:
    data = train_df.dropna(subset=["mix_arctic", "mix_pert"])
    long = data.melt(id_vars=["step"], value_vars=["mix_arctic", "mix_pert"], var_name="source", value_name="fraction")
    long["source"] = long["source"].map({"mix_arctic": "ARCTIC pair-alpha", "mix_pert": "perturbed Libri pair-alpha"})
    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    sns.lineplot(data=long, x="step", y="fraction", hue="source", ax=ax, linewidth=2.0)
    ax.set_title("Dual-invariance pair-alpha source mix")
    ax.set_xlabel("training step")
    ax.set_ylabel("batch fraction")
    ax.set_ylim(-0.04, 1.04)
    ax.axhline(0.6, color="0.35", linestyle="--", linewidth=1.0)
    ax.text(train_df["step"].max(), 0.615, "target ARCTIC weight 0.6", ha="right", color="0.35")
    save(fig, "week6_dual_pair_mix")


def plot_dual_trajectories(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.8), constrained_layout=True)

    loss_cols = ["recon", "pr", "sid", "inv_L", "inv_P", "var"]
    losses = train_df.melt(id_vars=["step"], value_vars=[c for c in loss_cols if c in train_df], var_name="loss", value_name="value").dropna()
    first = losses.sort_values("step").groupby("loss")["value"].first().rename("start")
    losses = losses.join(first, on="loss")
    losses["normalized"] = losses["value"] / losses["start"]
    sns.lineplot(data=losses, x="step", y="normalized", hue="loss", ax=axes[0, 0], linewidth=2.0)
    axes[0, 0].axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("Training losses normalized to first logged value")
    axes[0, 0].set_xlabel("training step")
    axes[0, 0].set_ylabel("fraction of initial value")
    axes[0, 0].legend(title="")

    val_cols = ["zL_PR_PER", "zL_SID_acc_proxy", "zP_SID_acc"]
    val_long = val_df.melt(id_vars=["step"], value_vars=[c for c in val_cols if c in val_df], var_name="metric", value_name="value").dropna()
    val_long["metric"] = val_long["metric"].map(
        {
            "zL_PR_PER": "z_L PR PER",
            "zL_SID_acc_proxy": "z_L SID proxy",
            "zP_SID_acc": "z_P SID proxy",
        }
    )
    sns.lineplot(data=val_long, x="step", y="value", hue="metric", marker="o", ax=axes[0, 1], linewidth=2.0)
    axes[0, 1].set_title("Validation proxy trajectories")
    axes[0, 1].set_xlabel("training step")
    axes[0, 1].set_ylabel("proxy metric value")
    axes[0, 1].set_ylim(-0.04, 1.04)
    axes[0, 1].legend(title="")

    route_ax = axes[1, 0]
    active_ax = route_ax.twinx()
    route_palette = {"z_L preactivation": "#4C72B0", "z_P preactivation": "#DD8452"}
    active_palette = {"z_L post-activation": "#4C72B0", "z_P post-activation": "#DD8452"}
    route = train_df.melt(id_vars=["step"], value_vars=["route_L", "route_P"], var_name="bucket", value_name="count").dropna()
    route["bucket"] = route["bucket"].map({"route_L": "z_L preactivation", "route_P": "z_P preactivation"})
    active = train_df.melt(id_vars=["step"], value_vars=["active_L", "active_P"], var_name="bucket", value_name="count").dropna()
    active["bucket"] = active["bucket"].map({"active_L": "z_L post-activation", "active_P": "z_P post-activation"})
    sns.lineplot(data=route, x="step", y="count", hue="bucket", palette=route_palette, ax=route_ax, linewidth=2.0, legend=False)
    sns.lineplot(data=active, x="step", y="count", hue="bucket", palette=active_palette, ax=active_ax, linewidth=2.0, linestyle="--", legend=False)
    route_ax.axhline(2560, color="0.35", linestyle=":", linewidth=1.0)
    active_ax.axhline(128, color="0.55", linestyle=":", linewidth=1.0)
    route_ax.set_title("Learned route allocation before and after activation")
    route_ax.set_xlabel("training step")
    route_ax.set_ylabel("preactivation route count (of 5120)")
    active_ax.set_ylabel("post-activation active count (of 256)")
    active_ax.grid(False)
    route_ax.legend(
        handles=[
            Line2D([0], [0], color="#4C72B0", linewidth=2.0, label="z_L preactivation"),
            Line2D([0], [0], color="#DD8452", linewidth=2.0, label="z_P preactivation"),
            Line2D([0], [0], color="#4C72B0", linewidth=2.0, linestyle="--", label="z_L post-activation"),
            Line2D([0], [0], color="#DD8452", linewidth=2.0, linestyle="--", label="z_P post-activation"),
        ],
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=9,
    )

    mix = train_df.dropna(subset=["mix_arctic", "mix_pert"]).melt(
        id_vars=["step"],
        value_vars=["mix_arctic", "mix_pert"],
        var_name="source",
        value_name="fraction",
    )
    mix["source"] = mix["source"].map({"mix_arctic": "ARCTIC pair alpha", "mix_pert": "perturbed Libri pair alpha"})
    sns.lineplot(data=mix, x="step", y="fraction", hue="source", ax=axes[1, 1], linewidth=2.0)
    axes[1, 1].axhline(0.6, color="0.35", linestyle="--", linewidth=1.0)
    axes[1, 1].set_title("Pair-alpha source mix")
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].set_ylabel("batch fraction")
    axes[1, 1].set_ylim(-0.04, 1.04)
    axes[1, 1].legend(title="")

    fig.suptitle("Dual-invariance v1 no-GRL run trajectories", y=1.03, fontsize=15, fontweight="bold")
    save(fig, "week6_dual_inv_trajectories")


def plot_arctic_probe(arctic_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    if not arctic_df.empty:
        sns.lineplot(data=arctic_df, x="step", y="dev_acc", marker="o", ax=ax, linewidth=2.2, color="#cc6677")
    ax.axhline(1 / 18, color="0.35", linestyle="--", linewidth=1.0)
    ax.text(arctic_df["step"].max() if not arctic_df.empty else 1, 1 / 18 + 0.02, "chance 1/18", ha="right", color="0.35")
    ax.set_title("Partial ARCTIC z_L SID probe climbs rapidly")
    ax.set_xlabel("probe training step")
    ax.set_ylabel("dev SID accuracy")
    ax.set_ylim(0, 1.05)
    save(fig, "week6_dual_arctic_sid_partial")


def plot_sidpr_checkpoint_confounds() -> None:
    data = pd.DataFrame(
        [
            {"seed": "7", "measure": "final validation proxy", "zL_sid_acc": 0.008},
            {"seed": "7", "measure": "selected checkpoint probe", "zL_sid_acc": 0.704},
            {"seed": "21", "measure": "final validation proxy", "zL_sid_acc": 0.001},
            {"seed": "21", "measure": "selected checkpoint probe", "zL_sid_acc": 0.452},
            {"seed": "84", "measure": "final validation proxy", "zL_sid_acc": 0.006},
            {"seed": "84", "measure": "selected checkpoint probe", "zL_sid_acc": 0.006},
        ]
    )
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    sns.barplot(data=data, x="seed", y="zL_sid_acc", hue="measure", ax=ax, palette=["#8fbcd4", "#d95f5f"])
    ax.axhline(1 / 251, color="0.35", linestyle="--", linewidth=1.0)
    ax.text(2.45, 1 / 251 + 0.015, "chance 1/251", ha="right", color="0.35")
    ax.set_title("Job2 seed replications: proxy and selected-checkpoint probe disagree")
    ax.set_xlabel("training seed")
    ax.set_ylabel("z_L -> SID accuracy")
    ax.set_ylim(0, 0.8)
    save(fig, "week6_sidpr_checkpoint_confounds")


def plot_sidpr_probe_summary() -> None:
    data = pd.DataFrame(
        [
            {"run": "Job2 dense GradNorm", "metric": "z_L PR PER", "value": 0.067},
            {"run": "Job2 dense GradNorm", "metric": "z_L SID acc", "value": 0.010},
            {"run": "Job2 dense GradNorm", "metric": "z_P PR PER", "value": 0.534},
            {"run": "Job2 dense GradNorm", "metric": "z_P SID acc", "value": 0.972},
            {"run": "Invariance-only fixed blocks", "metric": "z_L PR PER", "value": 0.066},
            {"run": "Invariance-only fixed blocks", "metric": "z_L SID acc", "value": 0.010},
            {"run": "Invariance-only fixed blocks", "metric": "z_P PR PER", "value": 0.864},
            {"run": "Invariance-only fixed blocks", "metric": "z_P SID acc", "value": 1.000},
            {"run": "Dual-inv v1 no-GRL", "metric": "z_L PR PER", "value": 0.054},
            {"run": "Dual-inv v1 no-GRL", "metric": "z_L SID acc", "value": 1.000},
        ]
    )
    g = sns.catplot(
        data=data,
        x="run",
        y="value",
        hue="metric",
        kind="bar",
        height=5.0,
        aspect=1.8,
        palette=sns.color_palette("colorblind", 4),
    )
    g.set_axis_labels("", "held-out diagnostic value")
    g.set_xticklabels(rotation=15, ha="right")
    g.figure.suptitle("Held-out probes separate successful invariance from proxy failures", y=1.05, fontsize=15, fontweight="bold")
    save(g.figure, "week6_sidpr_probe_summary")


def concat_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    clear_outputs()

    train_frames = []
    val_frames = []
    emotion_frames = []
    grad_frames = []
    cos_frames = []
    for run, path in MULTITASK_LOGS.items():
        train, val, emotion, grad, cos = parse_log(path, run)
        train_frames.append(train)
        val_frames.append(val)
        emotion_frames.append(emotion)
        grad_frames.append(grad)
        cos_frames.append(cos)

    mt_train = concat_frames(train_frames)
    mt_val = concat_frames(val_frames)
    mt_emotion = concat_frames(emotion_frames)
    mt_grad = concat_frames(grad_frames)
    mt_cos = concat_frames(cos_frames)

    dual_train, dual_val, _, _, _ = parse_log(DUAL_LOG, "dual_inv_v1_soft_nogrl")

    plot_multitask_losses(mt_train)
    plot_multitask_routing(mt_train)
    plot_multitask_gradients(mt_grad)
    plot_multitask_gradient_conflicts(mt_cos)
    plot_dual_trajectories(dual_train, dual_val)

    print(f"Wrote figures to {OUT}")
    for path in sorted(OUT.glob("week6_*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
