from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare historical trainer deadness with frozen-checkpoint replays."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("LABEL", "TRAIN_DEAD", "DEADNESS_JSON", "HEALTH_5K_JSON"),
        required=True,
        help="Repeat once per checkpoint; fractions are in [0,1].",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    rows: list[dict] = []
    for label, training, deadness_path, health_path in args.run:
        deadness = json.loads(Path(deadness_path).read_text(encoding="utf-8"))
        health = json.loads(Path(health_path).read_text(encoding="utf-8"))
        interval = deadness["train_like_dead_fraction_replay_interval"]
        routes = {item["route"]: item["train_like_dead_fraction"] for item in deadness["route_summary"]}
        rows.append({
            "model": label,
            "training_log_dead_fraction": float(training),
            "frozen_replay_dead_fraction": float(deadness["train_like_dead_fraction"]),
            "frozen_replay_ci_low": float(interval[0]),
            "frozen_replay_ci_high": float(interval[1]),
            "five_k_unobserved_fraction": float(health["unobserved_units"]) / float(health["K"]),
            "frozen_replay_L_dead_fraction": routes.get("L", np.nan),
            "frozen_replay_P_dead_fraction": routes.get("P", np.nan),
            "frozen_replay_unassigned_dead_fraction": routes.get("unassigned", np.nan),
            "utterances": int(deadness["utterances"]),
            "analysis_batches": int(deadness["deadness_analysis_batches"]),
            "threshold_batches": int(deadness["deadness_threshold_batches"]),
            "replays": int(deadness["deadness_replays"]),
        })

    frame = pd.DataFrame(rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "deadness_comparison.csv", index=False)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "#ffffff",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    labels = frame["model"].tolist()
    y = np.arange(len(frame))[::-1]

    ax = axes[0]
    for yi, (_, row) in zip(y, frame.iterrows()):
        train = 100 * row.training_log_dead_fraction
        replay = 100 * row.frozen_replay_dead_fraction
        ax.plot([train, replay], [yi, yi], color="#cbd5e1", lw=3, zorder=1)
        ax.scatter(train, yi, s=70, color="#e76f51", edgecolor="white", lw=1.2, zorder=3)
        lo = 100 * (row.frozen_replay_dead_fraction - row.frozen_replay_ci_low)
        hi = 100 * (row.frozen_replay_ci_high - row.frozen_replay_dead_fraction)
        ax.errorbar(replay, yi, xerr=np.asarray([[lo], [hi]]), fmt="D", ms=7,
                    color="#264653", ecolor="#264653", capsize=4, lw=1.8, zorder=4)
    ax.set_yticks(y, labels)
    ax.set_xlim(-3, 76)
    ax.set_xlabel("units (%)")
    ax.set_title("A. Historical log vs frozen replay", loc="left", fontweight="bold")
    ax.scatter([], [], s=70, color="#e76f51", label="training log (historical)")
    ax.scatter([], [], s=55, marker="D", color="#264653", label="12k frozen replay (95% order interval)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    ax = axes[1]
    for yi, (_, row) in zip(y, frame.iterrows()):
        unobserved = 100 * row.five_k_unobserved_fraction
        replay = 100 * row.frozen_replay_dead_fraction
        ax.plot([unobserved, replay], [yi, yi], color="#dbeafe", lw=4, zorder=1)
        ax.scatter(unobserved, yi, s=72, marker="s", color="#2a9d8f",
                   edgecolor="white", lw=1.2, zorder=3)
        ax.scatter(replay, yi, s=62, marker="D", color="#264653",
                   edgecolor="white", lw=1.2, zorder=4)
        ax.text(max(unobserved, replay) + 1.0, yi,
                f"{unobserved:.1f}% / {replay:.1f}%", va="center", fontsize=8, color="#334155")
    ax.set_yticks(y, labels)
    ax.set_xlim(-3, 76)
    ax.set_xlabel("units (%)")
    ax.set_title("B. More data does not recover frozen units", loc="left", fontweight="bold")
    ax.scatter([], [], s=72, marker="s", color="#2a9d8f", label="5k never observed")
    ax.scatter([], [], s=62, marker="D", color="#264653", label="12k frozen replay")
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    for ax in axes:
        ax.grid(axis="x", color="#e2e8f0", lw=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.savefig(output / "deadness_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "deadness_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    (output / "README.md").write_text(
        "# Deadness comparison\n\n"
        "Panel A deliberately compares, but does not equate, two measurements. "
        "The training log is the transient counter accumulated while weights changed "
        "and the legacy trainer included padded frames. The 12k replay freezes the final "
        "checkpoint, excludes padding, and reports the mean over ten shuffled orders. "
        "Panel B shows that the earlier 5k never-observed fractions already closely "
        "predict the frozen-checkpoint rolling result; sample length was not the main "
        "cause of the routed models' inactive capacity.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
