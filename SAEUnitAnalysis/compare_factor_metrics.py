from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _label(path: Path) -> tuple[str, int | None, str]:
    name = path.name
    if "fixed_240L16P_step8000" in name:
        return "Fixed · 8k", 8, "fixed"
    if "fixed_240L16P_step12000" in name:
        return "Fixed · 12k final", 12, "fixed"
    if "learned_freeze4k_step20000" in name:
        return "Learned baseline · 20k final", 20, "learned_baseline"
    for step in (5000, 12000, 16000, 20000):
        if f"step{step}" in name and "postgp030" in name:
            suffix = " final" if step == 20000 else ""
            return f"Learned gp030 · {step // 1000}k{suffix}", step // 1000, "gp030"
    return name, None, "other"


def comparison_table(results: list[Path]) -> pd.DataFrame:
    rows = []
    for result in results:
        source = result / "tables" / "speech_factor_metrics.csv"
        if not source.exists():
            continue
        table = pd.read_csv(source)
        if "scope" not in table or not table.scope.eq("route_subspace").all():
            continue
        label, step, family = _label(result)
        selected = table[
            (table.component == "route_contrast")
            & (table.control == "observed")
            & (table.target.isin(["phone", "speaker_id"]))
        ]
        for _, item in selected.iterrows():
            rows.append({
                "result": result.name, "label": label, "family": family,
                "step_k": step, "metric": item.metric,
                "factor": "Phone" if item.target == "phone" else "Speaker",
                "contrast": item.value, "interval_low": item.ci95_low,
                "interval_high": item.ci95_high,
                "capacity_mode": item.capacity_mode,
            })
        structure = table[
            (table.metric == "DCI") & (table.component == "directional_alignment")
            & (table.target == "mean") & (table.capacity_mode == "all_observed")
            & (table.control == "observed")
        ]
        if len(structure):
            item = structure.iloc[0]
            rows.append({
                "result": result.name, "label": label, "family": family,
                "step_k": step, "metric": "DCI alignment", "factor": "Mean",
                "contrast": item.value, "interval_low": item.ci95_low,
                "interval_high": item.ci95_high, "capacity_mode": "all_observed",
            })
    return pd.DataFrame(rows)


def make_comparison(results: list[Path], output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    table = comparison_table(results)
    if table.empty:
        raise SystemExit("No grouped factor-metric results were found.")
    table.to_csv(output / "tables" / "checkpoint_route_metric_comparison.csv", index=False)

    primary = table[
        (table.capacity_mode == "all_observed") & (table.metric.isin(["MIG", "SAP", "DCI"]))
    ].copy()
    order = [
        label for label in (
            "Fixed · 8k", "Fixed · 12k final", "Learned baseline · 20k final",
            "Learned gp030 · 5k", "Learned gp030 · 12k",
            "Learned gp030 · 16k", "Learned gp030 · 20k final",
        ) if label in set(primary.label)
    ]
    colors = {"Phone": "#087f5b", "Speaker": "#d9480f"}
    markers = {"Phone": "o", "Speaker": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.0), sharey=True)
    for ax, metric in zip(axes, ("MIG", "SAP", "DCI")):
        subset = primary[primary.metric == metric]
        for y, label in enumerate(order):
            for factor, offset in (("Phone", -.13), ("Speaker", .13)):
                match = subset[(subset.label == label) & (subset.factor == factor)]
                if match.empty:
                    continue
                row = match.iloc[0]; value = float(row.contrast)
                ax.errorbar(
                    value, y + offset,
                    xerr=[[value - float(row.interval_low)], [float(row.interval_high) - value]],
                    fmt=markers[factor], ms=7, capsize=3, color=colors[factor],
                    label=factor if y == 0 else None,
                )
        ax.axvline(0, color="#344054", lw=1)
        ax.set(title=metric, xlabel="desired-route contrast", yticks=np.arange(len(order)),
               yticklabels=order, xlim=(-.03, max(.10, float(subset.interval_high.max()) + .05)))
        ax.invert_yaxis()
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Grouped route disentanglement across analysed checkpoints", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "plots" / "checkpoint_route_metric_forest.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "plots" / "checkpoint_route_metric_forest.pdf", bbox_inches="tight")
    plt.close(fig)

    trajectory = table[
        (table.family == "gp030") & (table.metric.isin(["MIG", "SAP", "DCI"]))
        & (table.factor.isin(["Phone", "Speaker"]))
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=True)
    for ax, metric in zip(axes, ("MIG", "SAP", "DCI")):
        subset = trajectory[trajectory.metric == metric]
        for factor in ("Phone", "Speaker"):
            for capacity, linestyle, alpha, label_suffix in (
                ("all_observed", "-", 1.0, ""),
                ("matched_active_units", "--", .72, " · capacity matched"),
            ):
                group = subset[(subset.factor == factor) & (subset.capacity_mode == capacity)].sort_values("step_k")
                if group.empty:
                    continue
                ax.plot(group.step_k, group.contrast, marker=markers[factor], linestyle=linestyle,
                        color=colors[factor], alpha=alpha, lw=2,
                        label=f"{factor}{label_suffix}")
                if capacity == "all_observed":
                    ax.fill_between(group.step_k, group.interval_low, group.interval_high,
                                    color=colors[factor], alpha=.12)
        ax.axhline(0, color="#344054", lw=1)
        ax.set(title=metric, xlabel="training checkpoint (k steps)",
               ylabel="desired-route contrast", xticks=[5, 12, 16, 20])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Learned-route trajectory after quota freeze", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "plots" / "learned_route_metric_trajectory.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "plots" / "learned_route_metric_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)

    page = """<!doctype html><html><head><meta charset='utf-8'><style>
    body{font-family:Inter,system-ui,sans-serif;background:#f5f7fb;color:#162033;margin:0;padding:32px}
    .panel{background:white;border-radius:12px;padding:20px;margin:18px 0;box-shadow:0 2px 10px #1d2b4b18}
    img{max-width:100%;height:auto}</style><title>Grouped Route Metric Comparison</title></head><body>
    <h1>Grouped Route Metric Comparison</h1>
    <p>Positive contrasts mean phone information favours zL and speaker information favours zP. These compare complete route vectors, not units inside a route.</p>
    <div class='panel'><h2>All checkpoints</h2><img src='../plots/checkpoint_route_metric_forest.png'></div>
    <div class='panel'><h2>Learned-route trajectory</h2><img src='../plots/learned_route_metric_trajectory.png'></div>
    </body></html>"""
    (output / "report").mkdir(exist_ok=True)
    (output / "report" / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare grouped MIG/SAP/DCI across SAE reports.")
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("SAEUnitAnalysis/results/route_factor_comparison"))
    args = parser.parse_args()
    results = args.results or sorted(Path("SAEUnitAnalysis/results").glob("*5k_mps"))
    make_comparison(results, args.output)
    print(args.output / "report" / "index.html")


if __name__ == "__main__":
    main()
