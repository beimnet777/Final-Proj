from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bundle import AnalysisBundle
from .extraction import FeatureCache, _read_audio
from .types import ResolvedModel
from .utils import jsonable, write_json


CSS = """
body{font-family:Inter,system-ui,sans-serif;margin:0;background:#f5f7fb;color:#162033}
header{background:#17233c;color:white;padding:24px 5vw} main{padding:24px 5vw}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.card,table,.panel{background:white;border-radius:10px;padding:16px;box-shadow:0 2px 10px #1d2b4b18}
table{border-collapse:collapse;width:100%;padding:0}th,td{padding:8px;border-bottom:1px solid #e5e9f2;text-align:left}
input,select{padding:8px;margin:8px;border:1px solid #bbc4d5;border-radius:6px}.L{color:#087f5b}.P{color:#d9480f}.U{color:#7048e8}
a{color:#275dad}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:18px}.plot{max-width:100%;height:auto}
code{background:#eef1f7;padding:2px 5px;border-radius:4px}.muted{color:#667085}.warn{background:#fff3bf;padding:10px;border-radius:8px}
.badge{display:inline-block;border-radius:999px;padding:2px 8px;margin:2px;background:#eef1f7;font-size:.85em}
.tight td,.tight th{font-size:.92em;padding:6px}.scroll{overflow-x:auto}
"""


ROUTE_COLORS = {"L": "#087f5b", "P": "#d9480f", "U": "#7048e8", "unassigned": "#667085"}

PLOT_CAPTIONS = {
    "route_activity": "Distribution of how often observed units enter the frame-level Top-K, separated by route.",
    "phone_score_vs_speaker_score": "Each point is one SAE unit. PhoneScore uses positive frame-activity AUROC; SpeakerScore uses positive mean-activation correlation. Shapes mark the top-decile associations; anti-associations receive zero.",
    "route_probe_accuracy": "Held-out frozen linear-probe balanced accuracy. Disentanglement predicts the crossover visible here: phone decoding is stronger from L, while speaker decoding is stronger from P. Dashed segments show chance.",
    "route_classifier_free_geometry": "Classifier-free full-space cosine geometry with controlled pairs. Phone pairs cross speakers and utterances; speaker pairs cross transcript/content and utterances. Bars show same-label minus different-label cosine similarity with cluster-bootstrap 95% intervals.",
    "route_phone_probe_confusion": "A true held-out classification confusion matrix. Rows are actual phones and columns are frozen-probe predictions. The expected result is a strong L diagonal and a diffuse P matrix.",
    "route_speaker_probe_confusion": "A true held-out classification confusion matrix. Rows are actual speakers and columns are frozen-probe predictions. The expected result is a diffuse L matrix and a strong P diagonal.",
    "phone_unit_alignment_ranked": "All 39 unique phone–unit assignments, grouped by phonetic family. Filled circles show held-out test specificity: P(unit in Top-K | target phone) minus the largest P(unit in Top-K | any other phone). Diamonds show the train+validation selection margin; connecting lines expose generalization shifts. This is an activity-specificity margin, not correlation.",
    "phone_selected_unit_confusion": "DEPRECATED diagnostic: this is raw P(unit active | phone), not a classifier confusion matrix. Broadly active units and related phones can create bright off-diagonals; use the held-out route-probe matrices for the main claim.",
    "route_phone_representation_embedding": "Centered PCA of the same held-out test frames and phone labels in L and P. Labels are selected route-neutrally on the probe-training partition; evaluation points are untouched.",
    "route_speaker_representation_embedding": "Centered PCA of the same held-out test utterances and speakers in L and P. The two panels use identical observations.",
    "route_phone_representation_umap": "Supplementary cosine-UMAP of exactly the same held-out phone observations used by PCA. UMAP preserves local neighbourhoods but can exaggerate visual gaps; quantitative claims use probes and full-space geometry.",
    "route_speaker_representation_umap": "Supplementary cosine-UMAP of exactly the same held-out speaker observations used by PCA. Parameters are n_neighbors=30, min_dist=0.1 and seed=42.",
    "latent_swap_outcomes": "Feature-level intervention, not generated audio. P-swap combines recipient L with donor P; L-swap is the complementary control. Bars report recipient-phone preservation and whether the reconstructed SPEAR features match donor or recipient speaker identity. Evaluators are calibrated only on unswapped SAE reconstructions. The shuffled-mask diagnostic can overlap true P units—especially in learned models with roughly half their capacity assigned to P—so it is not a clean negative control.",
    "route_selectivity_composition": "Fractions use observed-active units as the denominator; assigned-capacity fractions remain in the CSV tables.",
}

PLOT_TITLES = {
    "route_probe_accuracy": "Held-Out Probe Accuracy: Phone–Speaker Crossover",
    "route_classifier_free_geometry": "Classifier-Free Geometry: Phone–Speaker Crossover",
    "route_phone_probe_confusion": "Held-Out Phone Probe Confusion",
    "route_speaker_probe_confusion": "Held-Out Speaker Probe Confusion",
    "phone_unit_alignment_ranked": "39-Phone SAE Unit Alignment Atlas",
    "phone_selected_unit_confusion": "Deprecated: Raw 39-Phone Unit-Coverage Map",
    "route_phone_representation_umap": "Supplementary UMAP: Held-Out Phone Vectors",
    "route_speaker_representation_umap": "Supplementary UMAP: Held-Out Speaker Vectors",
    "latent_swap_outcomes": "Latent Swap: Content Retention and Speaker Transfer",
}


def _sns():
    try:
        import seaborn as sns
    except Exception:
        return None
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        },
    )
    return sns


def _plot(path: Path, draw, *, figsize: tuple[float, float] = (8.6, 5.2)) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    _sns()
    fig = plt.figure(figsize=figsize)
    draw(fig)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _save_plot_data(output: Path, name: str, data: pd.DataFrame, *, index: bool = False) -> Path:
    path = output / "tables" / "plot_data" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=index)
    return path


def _unit_type_series(frame: pd.DataFrame) -> pd.Series:
    phone = frame.get("phone_selective", False)
    speaker = frame.get("speaker_selective", False)
    prosody = frame.get("prosody_selective", False)
    metadata = frame.get("metadata_paralinguistic_selective", False)
    emotion = frame.get("emotion_selective", False)
    linguistic = frame.get("linguistic_selective", False)
    paraling = frame.get("paralinguistic_selective", False)

    phone = pd.Series(phone, index=frame.index, dtype=bool)
    speaker = pd.Series(speaker, index=frame.index, dtype=bool)
    prosody = pd.Series(prosody, index=frame.index, dtype=bool)
    metadata = pd.Series(metadata, index=frame.index, dtype=bool)
    emotion = pd.Series(emotion, index=frame.index, dtype=bool)
    linguistic = pd.Series(linguistic, index=frame.index, dtype=bool)
    paraling = pd.Series(paraling, index=frame.index, dtype=bool)

    labels = pd.Series("nonselective", index=frame.index, dtype="object")
    labels.loc[linguistic & paraling] = "mixed_linguistic_paralinguistic"
    labels.loc[phone & ~(speaker | prosody | metadata | emotion)] = "phone_only"
    labels.loc[speaker & ~phone] = "speaker_only"
    labels.loc[prosody & ~(phone | speaker)] = "prosody_only"
    labels.loc[metadata & ~(phone | speaker | prosody)] = "metadata_only"
    labels.loc[emotion & ~(phone | speaker | prosody | metadata)] = "emotion_only"
    labels.loc[linguistic & ~phone & ~paraling] = "linguistic_other"
    labels.loc[paraling & ~(speaker | prosody | metadata | emotion) & ~linguistic] = "paralinguistic_other"
    return labels


def make_plots(output: Path, tables: dict[str, pd.DataFrame]) -> list[Path]:
    made = []
    sns = _sns()
    focused = tables.get("phone_speaker_scores") is not None
    health = tables.get("health")
    if health is not None and len(health):
        path = output / "plots" / "route_activity"
        if "observed_active" in health.columns:
            active = health[health.observed_active.fillna(False)].copy()
        else:
            active = health[~health.dead].copy()
        active["log10_frame_frequency"] = np.log10(active.frame_frequency.clip(lower=1e-9))
        _save_plot_data(
            output,
            "route_activity",
            active[[c for c in (
                "unit", "route", "route_id", "route_probability", "frame_frequency",
                "utterance_frequency", "active_frames", "active_utterances",
                "log10_frame_frequency",
            ) if c in active.columns]],
        )
        def draw(fig):
            ax = fig.subplots()
            if sns is not None:
                sns.histplot(
                    data=active, x="log10_frame_frequency", hue="route",
                    hue_order=[x for x in ("L", "P", "U", "unassigned") if x in set(active.route)],
                    palette=ROUTE_COLORS, bins=45, element="step", stat="count",
                    common_norm=False, alpha=.35, ax=ax,
                )
            else:
                for route, group in active.groupby("route"):
                    ax.hist(group.log10_frame_frequency, bins=40, alpha=.55, label=route,
                            color=ROUTE_COLORS.get(route))
                ax.legend()
            ax.set(xlabel="log10 frame firing frequency", ylabel="units", title="Active-unit firing distribution")
        _plot(path, draw); made.append(path.with_suffix(".png"))
    profiles = tables.get("profiles")
    if profiles is not None and len(profiles) and not focused:
        path = output / "plots" / "unit_factor_alignment"
        _save_plot_data(output, "unit_factor_alignment", profiles.copy())
        def draw(fig):
            ax = fig.subplots()
            if sns is not None:
                sns.scatterplot(
                    data=profiles, x="linguistic_score", y="paralinguistic_score",
                    hue="route", palette=ROUTE_COLORS, s=16, alpha=.55,
                    linewidth=0, ax=ax,
                )
            else:
                for route, group in profiles.groupby("route"):
                    ax.scatter(group.linguistic_score, group.paralinguistic_score, s=8, alpha=.45,
                               label=route, color=ROUTE_COLORS.get(route))
                ax.legend()
            ax.set(xlabel="linguistic selectivity", ylabel="paralinguistic selectivity",
                   title="Unit factor alignment")
        _plot(path, draw); made.append(path.with_suffix(".png"))
    unit_scores = tables.get("phone_speaker_scores")
    if unit_scores is not None and len(unit_scores):
        path = output / "plots" / "phone_score_vs_speaker_score"
        _save_plot_data(output, "phone_score_vs_speaker_score", unit_scores.copy())
        def draw(fig):
            ax = fig.subplots()
            if sns is not None:
                sns.scatterplot(
                    data=unit_scores, x="PhoneScore", y="SpeakerScore",
                    hue="route", style="category", palette=ROUTE_COLORS,
                    s=22, alpha=.65, linewidth=0, ax=ax,
                )
            else:
                for route, group in unit_scores.groupby("route"):
                    ax.scatter(group.PhoneScore, group.SpeakerScore, s=12, alpha=.55,
                               label=route, color=ROUTE_COLORS.get(route))
                ax.legend()
            ax.set(
                xlabel="PhoneScore (positive active AUROC)",
                ylabel="SpeakerScore (positive mean-activation correlation)",
                title="Per-unit phone vs. speaker scores",
            )
        _plot(path, draw); made.append(path.with_suffix(".png"))
    separation = tables.get("representation_separation")
    if separation is not None and len(separation) and "linear_probe_balanced_accuracy" in separation:
        accuracy = separation[[
            c for c in ("target", "route", "labels", "linear_probe_balanced_accuracy")
            if c in separation.columns
        ]].copy()
        accuracy = accuracy.dropna(subset=["linear_probe_balanced_accuracy"])
        if len(accuracy):
            accuracy["target_label"] = accuracy["target"].map({
                "phone": "Phone", "speaker_id": "Speaker",
            }).fillna(accuracy["target"].astype(str))
            accuracy["chance"] = 1.0 / accuracy["labels"].clip(lower=1).astype(float)
            _save_plot_data(output, "route_probe_accuracy", accuracy)
            path = output / "plots" / "route_probe_accuracy"
            def draw(fig):
                ax = fig.subplots()
                targets = [
                    t for t in ("Phone", "Speaker")
                    if t in set(accuracy["target_label"])
                ]
                targets += sorted(set(accuracy["target_label"]) - set(targets))
                routes = [r for r in ("L", "P") if r in set(accuracy["route"])]
                x = np.arange(len(targets), dtype=float)
                width = 0.34
                for ri, route in enumerate(routes):
                    values = []
                    for target in targets:
                        match = accuracy[
                            (accuracy["target_label"] == target) &
                            (accuracy["route"] == route)
                        ]
                        values.append(float(match.iloc[0]["linear_probe_balanced_accuracy"]) if len(match) else np.nan)
                    offsets = x + (ri - (len(routes) - 1) / 2) * width
                    bars = ax.bar(
                        offsets, values, width=width, label=f"{route} route",
                        color=ROUTE_COLORS.get(route, "#667085"), alpha=.88,
                    )
                    for bar, value in zip(bars, values):
                        if np.isfinite(value):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2, value + .025,
                                f"{value:.1%}", ha="center", va="bottom", fontsize=9,
                            )
                target_chances = []
                for target in targets:
                    match = accuracy[accuracy["target_label"] == target]
                    target_chances.append(float(match.iloc[0]["chance"]) if len(match) else np.nan)
                finite_chances = [value for value in target_chances if np.isfinite(value)]
                if finite_chances and np.allclose(finite_chances, finite_chances[0]):
                    chance = finite_chances[0]
                    ax.axhline(chance, color="#475467", ls="--", lw=1.4)
                    ax.text(
                        .99, chance + .015, f"chance {chance:.1%}",
                        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                        fontsize=8, color="#475467",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1},
                    )
                else:
                    for xi, chance in enumerate(target_chances):
                        if np.isfinite(chance):
                            ax.hlines(chance, xi - .43, xi + .43, colors="#475467", linestyles="--", lw=1.4)
                            ax.text(xi, chance + .015, f"chance {chance:.1%}", ha="center", va="bottom", fontsize=7, color="#475467")
                ax.set(
                    xticks=x, xticklabels=targets, ylim=(0, 1.08),
                    ylabel="balanced accuracy", title="Held-out route-vector probe accuracy",
                )
                ax.legend(loc="upper center", ncol=max(1, len(routes)), frameon=False)
            _plot(path, draw, figsize=(8.8, 5.4)); made.append(path.with_suffix(".png"))

    classifier_free_geometry = tables.get("classifier_free_geometry_summary")
    if classifier_free_geometry is not None and len(classifier_free_geometry):
        geometry_plot = classifier_free_geometry.copy()
        geometry_plot["target_label"] = geometry_plot["target"].map({
            "phone": "Phone", "speaker_id": "Speaker",
        }).fillna(geometry_plot["target"].astype(str))
        _save_plot_data(output, "route_classifier_free_geometry", geometry_plot)
        path = output / "plots" / "route_classifier_free_geometry"
        def draw(fig):
            targets = [
                target for target in ("Phone", "Speaker")
                if target in set(geometry_plot["target_label"])
            ]
            targets += sorted(set(geometry_plot["target_label"]) - set(targets))
            routes = [route for route in ("L", "P") if route in set(geometry_plot["route"])]
            axes = fig.subplots(1, max(1, len(targets)), squeeze=False)[0]
            for target_index, (ax, target) in enumerate(zip(axes, targets)):
                values, low_errors, high_errors = [], [], []
                for route in routes:
                    match = geometry_plot[
                        (geometry_plot["target_label"] == target) &
                        (geometry_plot["route"] == route)
                    ]
                    if match.empty:
                        values.append(np.nan); low_errors.append(0.0); high_errors.append(0.0)
                        continue
                    row = match.iloc[0]
                    value = float(row["paired_cosine_difference"])
                    low = float(row["ci95_low"]) if pd.notna(row.get("ci95_low")) else value
                    high = float(row["ci95_high"]) if pd.notna(row.get("ci95_high")) else value
                    values.append(value)
                    low_errors.append(max(0.0, value - low))
                    high_errors.append(max(0.0, high - value))
                x = np.arange(len(routes), dtype=float)
                bars = ax.bar(
                    x, values, width=.62,
                    color=[ROUTE_COLORS.get(route, "#667085") for route in routes],
                    alpha=.88, yerr=np.asarray([low_errors, high_errors]), capsize=5,
                )
                finite = [abs(v) for v in values if np.isfinite(v)]
                vertical = max(finite, default=.01) * .055
                for bar, value in zip(bars, values):
                    if np.isfinite(value):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            value + (vertical if value >= 0 else -vertical),
                            f"{value:+.3f}", ha="center",
                            va="bottom" if value >= 0 else "top", fontsize=10,
                        )
                ax.axhline(0, color="#475467", lw=1.1)
                ax.set(
                    xticks=x, xticklabels=[f"{route} route" for route in routes],
                    title=f"{target} geometry",
                    ylabel="same-label − different-label cosine" if target_index == 0 else "",
                )
            for ax in axes[len(targets):]:
                ax.axis("off")
            fig.suptitle("Controlled full-space geometry (no classifier)", y=1.02, fontsize=13)
        _plot(path, draw, figsize=(10.4, 5.4)); made.append(path.with_suffix(".png"))

    probe_confusion = tables.get("probe_confusion")
    if probe_confusion is not None and len(probe_confusion):
        for target, plot_name, target_label in (
            ("phone", "route_phone_probe_confusion", "phone"),
            ("speaker_id", "route_speaker_probe_confusion", "speaker"),
        ):
            target_data = probe_confusion[probe_confusion["target"] == target].copy()
            if target_data.empty:
                continue
            _save_plot_data(output, plot_name, target_data)
            path = output / "plots" / plot_name
            def draw(fig, target_data=target_data, target=target, target_label=target_label):
                routes = [r for r in ("L", "P") if r in set(target_data["route"])]
                axes = fig.subplots(1, max(1, len(routes)), squeeze=False)[0]
                label_order = sorted(
                    set(target_data["true_label"].astype(str)) |
                    set(target_data["predicted_label"].astype(str))
                )
                for route_index, (ax, route) in enumerate(zip(axes, routes)):
                    route_data = target_data[target_data["route"] == route]
                    matrix = route_data.pivot_table(
                        index="true_label", columns="predicted_label",
                        values="row_fraction", aggfunc="first", fill_value=0.0,
                    ).reindex(index=label_order, columns=label_order, fill_value=0.0)
                    annotations = matrix.map(lambda value: f"{value:.0%}")
                    if sns is not None:
                        sns.heatmap(
                            matrix, cmap="Blues", vmin=0, vmax=1, square=True,
                            linewidths=.6, linecolor="white", annot=annotations,
                            fmt="", annot_kws={"fontsize": 6.5},
                            cbar=route_index == len(routes) - 1,
                            cbar_kws={"label": "fraction of actual class"}, ax=ax,
                        )
                    else:
                        image = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1)
                        if route_index == len(routes) - 1:
                            fig.colorbar(image, ax=ax, label="fraction of actual class")
                        ax.set_xticks(range(len(label_order)), label_order)
                        ax.set_yticks(range(len(label_order)), label_order)
                    score_match = separation[
                        (separation["target"] == target) & (separation["route"] == route)
                    ] if separation is not None and len(separation) else pd.DataFrame()
                    score = (
                        float(score_match.iloc[0]["linear_probe_balanced_accuracy"])
                        if len(score_match) else float("nan")
                    )
                    score_text = f" · balanced accuracy {score:.1%}" if np.isfinite(score) else ""
                    ax.set(
                        xlabel=f"predicted {target_label}", ylabel=f"actual {target_label}",
                        title=f"{route} route{score_text}",
                    )
                    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
                    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
                for ax in axes[len(routes):]:
                    ax.axis("off")
                fig.suptitle(f"Held-out frozen linear-probe {target_label} predictions", y=1.02, fontsize=13)
            _plot(path, draw, figsize=(13.4, 6.0)); made.append(path.with_suffix(".png"))
    selected_phone_units = tables.get("selected_phone_units")
    if selected_phone_units is not None and len(selected_phone_units):
        required = {"phone", "unit", "route", "selection_margin", "evaluation_margin"}
        if required.issubset(selected_phone_units.columns):
            atlas = selected_phone_units.copy()
            if "phone_family" not in atlas.columns:
                atlas["phone_family"] = "other"
            family_order = [
                "vowel", "stop", "fricative", "affricate",
                "nasal", "liquid", "glide", "other",
            ]
            atlas["phone_family"] = pd.Categorical(
                atlas["phone_family"].fillna("other").astype(str),
                categories=family_order, ordered=True,
            )
            atlas = atlas.sort_values(["phone_family", "phone"]).reset_index(drop=True)
            labels = []
            previous_family = None
            for _, row in atlas.iterrows():
                family = str(row["phone_family"])
                label = f"{row['phone']} · u{int(row['unit'])} · {row['route']}"
                if family != previous_family:
                    label = f"{family.upper()}  |  {label}"
                labels.append(label)
                previous_family = family
            atlas["unit_label"] = labels
            _save_plot_data(output, "phone_unit_alignment_ranked", atlas.copy())
            path = output / "plots" / "phone_unit_alignment_ranked"
            def draw(fig):
                from matplotlib.lines import Line2D
                ax = fig.subplots()
                y = np.arange(len(atlas), dtype=float)
                selection = atlas["selection_margin"].to_numpy(dtype=float)
                evaluation = atlas["evaluation_margin"].to_numpy(dtype=float)
                colors = [ROUTE_COLORS.get(route, "#667085") for route in atlas["route"]]
                # Alternating family bands keep all 39 rows legible without
                # turning the view back into a dense matrix.
                grouped = atlas.groupby("phone_family", observed=True, sort=False)
                for family_index, (_, group) in enumerate(grouped):
                    start, stop = int(group.index.min()), int(group.index.max())
                    if family_index % 2 == 0:
                        ax.axhspan(start - .5, stop + .5, color="#eef3f8", alpha=.55, zorder=0)
                    if start > 0:
                        ax.axhline(start - .5, color="#b8c2d1", lw=.8, zorder=1)
                for yi, start, stop, color in zip(y, selection, evaluation, colors):
                    ax.plot([start, stop], [yi, yi], color=color, lw=2.1, alpha=.58, zorder=2)
                ax.scatter(
                    selection, y, marker="D", s=30, facecolor="white",
                    edgecolor="#17233c", linewidth=1.25, zorder=4,
                )
                ax.scatter(
                    evaluation, y, marker="o", s=44, c=colors,
                    edgecolor="white", linewidth=.65, zorder=5,
                )
                ax.axvline(0, color="#475467", lw=1.0)
                span = max(
                    float(np.nanmax(np.abs(atlas[["selection_margin", "evaluation_margin"]].to_numpy()))),
                    .01,
                )
                for yi, value in zip(y, evaluation):
                    ax.text(
                        value + (.012 * span if value >= 0 else -.012 * span),
                        yi, f"{value:+.3f}", va="center",
                        ha="left" if value >= 0 else "right", fontsize=7.2,
                    )
                ax.set(
                    yticks=y, yticklabels=atlas["unit_label"],
                    xlabel="phone specificity margin",
                    ylabel="target phone · selected unit · route",
                    title=f"All {len(atlas)} unique phone–unit assignments · held-out specificity",
                )
                ax.invert_yaxis()
                ax.tick_params(axis="y", labelsize=7.6)
                route_handles = [
                    Line2D(
                        [0], [0], marker="o", color="none",
                        markerfacecolor=ROUTE_COLORS[route], markeredgecolor="white",
                        markersize=7, label=f"{route} route",
                    )
                    for route in ("L", "P") if route in set(atlas["route"].astype(str))
                ]
                metric_handles = [
                    Line2D(
                        [0], [0], marker="o", color="#667085",
                        markerfacecolor="#667085", markersize=6, lw=0,
                        label="held-out test",
                    ),
                    Line2D(
                        [0], [0], marker="D", color="#17233c",
                        markerfacecolor="white", markersize=5.5, lw=0,
                        label="train+validation selection",
                    ),
                ]
                ax.legend(
                    handles=metric_handles + route_handles,
                    loc="lower right", ncol=2, frameon=False,
                )
            height = max(10.5, .275 * len(atlas) + 2.0)
            _plot(path, draw, figsize=(11.2, height)); made.append(path.with_suffix(".png"))
    phone_confusion = tables.get("phone_unit_confusion")
    if phone_confusion is not None and len(phone_confusion):
        value_cols = [
            c for c in phone_confusion.columns
            if c not in {"selected_phone", "selected_unit", "route", "unit_baseline_active_probability"}
        ]
        if value_cols:
            plot_frame = phone_confusion.copy()
            labels = (
                plot_frame["selected_phone"].astype(str)
                + " · u"
                + plot_frame["selected_unit"].astype(str)
                + " · "
                + plot_frame["route"].astype(str)
            )
            matrix = plot_frame[value_cols].astype(float)
            matrix.index = labels
            _save_plot_data(output, "phone_selected_unit_confusion", phone_confusion.copy())
            path = output / "plots" / "phone_selected_unit_confusion"
            def draw(fig):
                ax = fig.subplots()
                if sns is not None:
                    sns.heatmap(
                        matrix, cmap="mako", vmin=0, vmax=min(1.0, max(0.05, float(matrix.max().max()))),
                        linewidths=.15, cbar_kws={"label": "P(unit active | actual phone)"}, ax=ax,
                    )
                else:
                    im = ax.imshow(matrix.to_numpy(), cmap="viridis", aspect="auto")
                    fig.colorbar(im, ax=ax, label="P(unit active | actual phone)")
                    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
                    ax.set_yticks(range(len(matrix.index)), matrix.index)
                ax.set(
                    xlabel="actual phone on frame",
                    ylabel="selected phone unit",
                    title="Deprecated raw coverage map · held-out test activity",
                )
                ax.tick_params(axis="x", labelrotation=90, labelsize=6)
                ax.tick_params(axis="y", labelsize=6)
            _plot(path, draw, figsize=(11.5, 9.5)); made.append(path.with_suffix(".png"))
    phone_embedding = tables.get("phone_embedding")
    if phone_embedding is not None and len(phone_embedding):
        _save_plot_data(output, "route_phone_representation_embedding", phone_embedding.copy())
        path = output / "plots" / "route_phone_representation_embedding"
        def draw(fig):
            routes = [r for r in ("L", "P") if r in set(phone_embedding.route)]
            axes = fig.subplots(1, max(1, len(routes)), squeeze=False)[0]
            for ax, route in zip(axes, routes):
                data = phone_embedding[phone_embedding.route == route]
                common_labels = sorted(phone_embedding["phone"].astype(str).unique().tolist())
                if sns is not None:
                    sns.scatterplot(
                        data=data, x="x", y="y", hue="phone", s=14, alpha=.65,
                        linewidth=0, hue_order=common_labels, ax=ax,
                    )
                else:
                    for phone, group in data.groupby("phone"):
                        ax.scatter(group.x, group.y, s=10, alpha=.6, label=phone)
                    ax.legend(fontsize=6)
                ax.set(title=f"{route} route · held-out phone vectors", xlabel="PCA component 1", ylabel="PCA component 2")
                if ax.get_legend() is not None:
                    ax.get_legend().set_title("phone")
                    for text in ax.get_legend().get_texts():
                        text.set_fontsize(6)
            for ax in axes[len(routes):]:
                ax.axis("off")
        _plot(path, draw, figsize=(12.5, 5.2)); made.append(path.with_suffix(".png"))
        if {"umap_x", "umap_y"}.issubset(phone_embedding.columns):
            _save_plot_data(output, "route_phone_representation_umap", phone_embedding.copy())
            path = output / "plots" / "route_phone_representation_umap"
            def draw(fig):
                routes = [r for r in ("L", "P") if r in set(phone_embedding.route)]
                axes = fig.subplots(1, max(1, len(routes)), squeeze=False)[0]
                for ax, route in zip(axes, routes):
                    data = phone_embedding[phone_embedding.route == route]
                    common_labels = sorted(phone_embedding["phone"].astype(str).unique().tolist())
                    if sns is not None:
                        sns.scatterplot(
                            data=data, x="umap_x", y="umap_y", hue="phone",
                            s=14, alpha=.65, linewidth=0, hue_order=common_labels, ax=ax,
                        )
                    else:
                        for phone, group in data.groupby("phone"):
                            ax.scatter(group.umap_x, group.umap_y, s=10, alpha=.6, label=phone)
                        ax.legend(fontsize=6)
                    ax.set(
                        title=f"{route} route · held-out phone neighbourhoods",
                        xlabel="UMAP component 1", ylabel="UMAP component 2",
                    )
                    if ax.get_legend() is not None:
                        ax.get_legend().set_title("phone")
                        for text in ax.get_legend().get_texts():
                            text.set_fontsize(6)
                for ax in axes[len(routes):]:
                    ax.axis("off")
            _plot(path, draw, figsize=(12.5, 5.2)); made.append(path.with_suffix(".png"))
    speaker_embedding = tables.get("speaker_embedding")
    if speaker_embedding is not None and len(speaker_embedding):
        _save_plot_data(output, "route_speaker_representation_embedding", speaker_embedding.copy())
        path = output / "plots" / "route_speaker_representation_embedding"
        def draw(fig):
            routes = [r for r in ("L", "P") if r in set(speaker_embedding.route)]
            axes = fig.subplots(1, max(1, len(routes)), squeeze=False)[0]
            for ax, route in zip(axes, routes):
                data = speaker_embedding[speaker_embedding.route == route]
                common_labels = sorted(speaker_embedding["speaker_id"].astype(str).unique().tolist())
                if sns is not None:
                    sns.scatterplot(
                        data=data, x="x", y="y", hue="speaker_id", s=30, alpha=.75,
                        linewidth=0, hue_order=common_labels, ax=ax,
                    )
                else:
                    for speaker, group in data.groupby("speaker_id"):
                        ax.scatter(group.x, group.y, s=16, alpha=.7, label=speaker)
                    ax.legend(fontsize=6)
                ax.set(title=f"{route} route · held-out speaker vectors", xlabel="PCA component 1", ylabel="PCA component 2")
                if ax.get_legend() is not None:
                    ax.get_legend().set_title("speaker")
                    for text in ax.get_legend().get_texts():
                        text.set_fontsize(6)
            for ax in axes[len(routes):]:
                ax.axis("off")
        _plot(path, draw, figsize=(12.5, 5.2)); made.append(path.with_suffix(".png"))
        if {"umap_x", "umap_y"}.issubset(speaker_embedding.columns):
            _save_plot_data(output, "route_speaker_representation_umap", speaker_embedding.copy())
            path = output / "plots" / "route_speaker_representation_umap"
            def draw(fig):
                routes = [r for r in ("L", "P") if r in set(speaker_embedding.route)]
                axes = fig.subplots(1, max(1, len(routes)), squeeze=False)[0]
                for ax, route in zip(axes, routes):
                    data = speaker_embedding[speaker_embedding.route == route]
                    common_labels = sorted(speaker_embedding["speaker_id"].astype(str).unique().tolist())
                    if sns is not None:
                        sns.scatterplot(
                            data=data, x="umap_x", y="umap_y", hue="speaker_id",
                            s=30, alpha=.75, linewidth=0, hue_order=common_labels, ax=ax,
                        )
                    else:
                        for speaker, group in data.groupby("speaker_id"):
                            ax.scatter(group.umap_x, group.umap_y, s=16, alpha=.7, label=speaker)
                        ax.legend(fontsize=6)
                    ax.set(
                        title=f"{route} route · held-out speaker neighbourhoods",
                        xlabel="UMAP component 1", ylabel="UMAP component 2",
                    )
                    if ax.get_legend() is not None:
                        ax.get_legend().set_title("speaker")
                        for text in ax.get_legend().get_texts():
                            text.set_fontsize(6)
                for ax in axes[len(routes):]:
                    ax.axis("off")
            _plot(path, draw, figsize=(12.5, 5.2)); made.append(path.with_suffix(".png"))
    disent = tables.get("disentanglement")
    if disent is not None and len(disent) and not focused:
        active = disent.copy()
        if "observed_active" in active:
            active = active[active.observed_active.fillna(False)]
        elif "dead" in active:
            active = active[~active.dead.fillna(False)]
        active["log_phone_like_score"] = np.log1p(active.get("phone_like_score", 0.0))
        active["log_speaker_score"] = np.log1p(active.get("speaker_score", 0.0))
        active["log_max_score"] = np.log1p(active.get("max_factor_score", 0.0))
        if "frame_frequency" in active:
            freq = active["frame_frequency"].fillna(0).clip(lower=1e-9)
        else:
            freq = pd.Series(np.full(len(active), 1e-9), index=active.index)
        active["log10_frame_frequency"] = np.log10(freq)
        active["unit_type"] = _unit_type_series(active)

        path = output / "plots" / "phone_speaker_quadrants"
        _save_plot_data(
            output,
            "phone_speaker_quadrants",
            active[[c for c in (
                "unit", "route", "unit_type", "phone_like_score", "speaker_score",
                "prosody_score", "metadata_paralinguistic_score", "paralinguistic_score",
                "linguistic_score", "mixed_phone_speaker", "mixed_linguistic_paralinguistic",
                "route_violation", "log_phone_like_score", "log_speaker_score",
            ) if c in active.columns]],
        )
        def draw(fig):
            ax = fig.subplots()
            if sns is not None:
                sns.scatterplot(
                    data=active, x="log_phone_like_score", y="log_speaker_score",
                    hue="route", style="mixed_phone_speaker",
                    palette=ROUTE_COLORS, s=22, alpha=.65, linewidth=0, ax=ax,
                )
            else:
                for route, group in active.groupby("route"):
                    ax.scatter(group.log_phone_like_score, group.log_speaker_score, s=12, alpha=.55,
                               label=route, color=ROUTE_COLORS.get(route))
                ax.legend()
            threshold = np.log1p(5.0)
            ax.axvline(threshold, color="#17233c", lw=1.0, ls="--", alpha=.6)
            ax.axhline(threshold, color="#17233c", lw=1.0, ls="--", alpha=.6)
            ax.text(threshold + .03, ax.get_ylim()[1] * .92, "phone-selective", fontsize=8, color="#17233c")
            ax.text(ax.get_xlim()[1] * .70, threshold + .03, "speaker-selective", fontsize=8, color="#17233c")
            ax.set(
                xlabel="log1p phone-like selectivity score",
                ylabel="log1p speaker selectivity score",
                title="Phone–speaker selectivity quadrants",
            )
        _plot(path, draw); made.append(path.with_suffix(".png"))

        path = output / "plots" / "frequency_vs_selectivity"
        _save_plot_data(
            output,
            "frequency_vs_selectivity",
            active[[c for c in (
                "unit", "route", "unit_type", "frame_frequency", "log10_frame_frequency",
                "max_factor_score", "log_max_score", "phone_like_score", "speaker_score",
                "paralinguistic_score", "linguistic_score", "route_violation",
            ) if c in active.columns]],
        )
        def draw(fig):
            ax = fig.subplots()
            if sns is not None:
                sns.scatterplot(
                    data=active, x="log10_frame_frequency", y="log_max_score",
                    hue="route", palette=ROUTE_COLORS, s=20, alpha=.62,
                    linewidth=0, ax=ax,
                )
            else:
                for route, group in active.groupby("route"):
                    ax.scatter(group.log10_frame_frequency, group.log_max_score, s=12, alpha=.55,
                               label=route, color=ROUTE_COLORS.get(route))
                ax.legend()
            ax.set(
                xlabel="log10 frame firing frequency",
                ylabel="log1p strongest factor score",
                title="Feature frequency vs. factor specificity",
            )
        _plot(path, draw); made.append(path.with_suffix(".png"))

        score_cols = [
            c for c in active.columns
            if (
                c.endswith("__score") or c in {
                    "phone_like_score", "speaker_score", "prosody_score",
                    "metadata_paralinguistic_score", "emotion_score",
                    "linguistic_score", "paralinguistic_score",
                    "delta_L_minus_P", "frame_frequency",
                }
            )
        ]
        factor_score_cols = [c for c in score_cols if c.endswith("__score")]
        score_frame = active[score_cols].apply(pd.to_numeric, errors="coerce")
        score_frame = score_frame.loc[:, score_frame.notna().any(axis=0)]
        score_frame = score_frame.loc[:, score_frame.std(axis=0, skipna=True).fillna(0) > 0]
        if len(factor_score_cols) > 2 and len(score_frame.columns) >= 2 and len(score_frame) >= 3:
            corr = score_frame.corr(method="pearson")
            _save_plot_data(output, "factor_score_pearson_correlation", corr.reset_index(names="factor"))
            path = output / "plots" / "factor_score_pearson_correlation"
            def draw(fig):
                ax = fig.subplots()
                if sns is not None:
                    sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap="vlag", square=True,
                                linewidths=.25, cbar_kws={"label": "Pearson r"}, ax=ax)
                else:
                    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
                    fig.colorbar(im, ax=ax, label="Pearson r")
                    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
                    ax.set_yticks(range(len(corr.index)), corr.index)
                ax.set(title="Pearson correlation between unit factor scores")
            _plot(path, draw); made.append(path.with_suffix(".png"))
        else:
            for stale in (
                output / "plots" / "factor_score_pearson_correlation.png",
                output / "plots" / "factor_score_pearson_correlation.pdf",
                output / "tables" / "plot_data" / "factor_score_pearson_correlation.csv",
            ):
                stale.unlink(missing_ok=True)

        order = [
            "phone_only", "speaker_only", "prosody_only", "metadata_only",
            "emotion_only", "mixed_linguistic_paralinguistic",
            "linguistic_other", "paralinguistic_other", "nonselective",
        ]
        confusion_counts = pd.crosstab(active["route"], active["unit_type"]).reindex(
            index=[r for r in ("L", "P", "U", "unassigned") if r in set(active["route"])],
            columns=order,
            fill_value=0,
        )
        confusion_counts = confusion_counts.loc[:, confusion_counts.sum(axis=0) > 0]
        if len(confusion_counts) and len(confusion_counts.columns):
            row_sums = confusion_counts.sum(axis=1).replace(0, np.nan)
            confusion_fraction = confusion_counts.div(row_sums, axis=0).fillna(0)
            confusion_long = confusion_counts.reset_index().melt(
                id_vars="route", var_name="unit_type", value_name="count")
            fraction_long = confusion_fraction.reset_index().melt(
                id_vars="route", var_name="unit_type", value_name="route_fraction")
            confusion_long = confusion_long.merge(fraction_long, on=["route", "unit_type"], how="left")
            _save_plot_data(output, "route_unit_type_confusion", confusion_long)
            path = output / "plots" / "route_unit_type_confusion"
            def draw(fig):
                ax = fig.subplots()
                if sns is not None:
                    sns.heatmap(confusion_fraction, annot=confusion_counts, fmt="d", cmap="Blues",
                                linewidths=.35, cbar_kws={"label": "row-normalized fraction"}, ax=ax)
                else:
                    im = ax.imshow(confusion_fraction.to_numpy(), cmap="Blues", vmin=0, vmax=1)
                    fig.colorbar(im, ax=ax, label="row-normalized fraction")
                    ax.set_xticks(range(len(confusion_fraction.columns)), confusion_fraction.columns, rotation=45, ha="right")
                    ax.set_yticks(range(len(confusion_fraction.index)), confusion_fraction.index)
                ax.set(xlabel="unit type", ylabel="route", title="Route × unit-type confusion matrix")
            _plot(path, draw); made.append(path.with_suffix(".png"))

    route_summary = tables.get("route_summary")
    if route_summary is not None and len(route_summary):
        cols = [
            "phone_selective_fraction", "speaker_selective_fraction",
            "mixed_phone_speaker_fraction", "route_violation_fraction",
        ]
        if not focused:
            cols[2:2] = ["prosody_selective_fraction", "metadata_paralinguistic_selective_fraction"]
        present = [c for c in cols if c in route_summary]
        if present:
            path = output / "plots" / "route_selectivity_composition"
            melted = route_summary.melt(id_vars=["route"], value_vars=present,
                                        var_name="category", value_name="fraction")
            melted["category"] = (
                melted["category"].str.replace("_fraction", "", regex=False)
                .str.replace("_", " ")
            )
            _save_plot_data(output, "route_selectivity_composition", melted)
            def draw(fig):
                ax = fig.subplots()
                if sns is not None:
                    sns.barplot(data=melted, x="route", y="fraction", hue="category", ax=ax)
                else:
                    pivot = melted.pivot(index="route", columns="category", values="fraction")
                    pivot.plot(kind="bar", ax=ax)
                ax.set(ylim=(0, min(1.0, max(.05, float(melted.fraction.max()) * 1.25))),
                       xlabel="route", ylabel="fraction of units",
                       title="Route composition by unit selectivity")
                ax.legend(fontsize=7, loc="upper right")
            _plot(path, draw); made.append(path.with_suffix(".png"))

    leaky = tables.get("leaky")
    if leaky is not None and len(leaky) and "leakage_score" in leaky and not focused:
        path = output / "plots" / "top_leaky_units"
        top = leaky.sort_values("leakage_score", ascending=False).head(25).copy()
        top["unit_label"] = top["unit"].astype(str) + " · " + top["route"].astype(str)
        _save_plot_data(output, "top_leaky_units", top)
        def draw(fig):
            ax = fig.subplots()
            top_plot = top.iloc[::-1]
            if sns is not None:
                sns.barplot(data=top_plot, x="leakage_score", y="unit_label", hue="route",
                            dodge=False, palette=ROUTE_COLORS, ax=ax)
            else:
                ax.barh(top_plot.unit_label, top_plot.leakage_score,
                        color=[ROUTE_COLORS.get(r, "#667085") for r in top_plot.route])
            ax.set(xlabel="route-violation score", ylabel="unit · route",
                   title="Highest-risk mixed/leaky units")
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        _plot(path, draw); made.append(path.with_suffix(".png"))

    causal = tables.get("causal")
    if causal is not None and len(causal) and "target_delta" in causal:
        path = output / "plots" / "causal_curves"
        data = causal.dropna(subset=["target_delta"]).copy()
        _save_plot_data(output, "causal_curves", data)
        def draw(fig):
            ax = fig.subplots()
            if sns is not None:
                sns.lineplot(data=data, x="budget", y="target_delta", hue="family",
                             style="mode", marker="o", ax=ax)
            else:
                for (family, mode), g in data.groupby(["family", "mode"]):
                    ax.plot(g.budget, g.target_delta, marker="o", label=f"{family}/{mode}")
                ax.legend(fontsize=8)
            ax.axhline(0, color="black", lw=.8); ax.set(xscale="log", xlabel="units", ylabel="target metric delta",
                                                        title="Necessity, sufficiency, and controls")
        _plot(path, draw); made.append(path.with_suffix(".png"))

    swap_summary = tables.get("swap_summary")
    if swap_summary is not None and len(swap_summary):
        metrics = [
            metric for metric in (
                "phone_recipient_accuracy", "donor_speaker_match", "recipient_speaker_match",
            ) if metric in swap_summary.columns
        ]
        if metrics:
            plot_frame = swap_summary[["mode", *metrics, *[
                column for metric in metrics
                for column in (f"{metric}_ci95_low", f"{metric}_ci95_high")
                if column in swap_summary.columns
            ]]].copy()
            _save_plot_data(output, "latent_swap_outcomes", plot_frame)
            path = output / "plots" / "latent_swap_outcomes"
            def draw(fig):
                ax = fig.subplots()
                mode_order = [
                    mode for mode in (
                        "baseline", "P_from_donor", "L_from_donor", "random_route_P_from_donor",
                    ) if mode in set(plot_frame["mode"])
                ]
                mode_labels = {
                    "baseline": "baseline",
                    "P_from_donor": "recipient L\n+ donor P",
                    "L_from_donor": "donor L\n+ recipient P",
                    "random_route_P_from_donor": "shuffled mask\n(overlaps P)",
                }
                metric_labels = {
                    "phone_recipient_accuracy": "recipient phone",
                    "donor_speaker_match": "donor speaker",
                    "recipient_speaker_match": "recipient speaker",
                }
                metric_colors = {
                    "phone_recipient_accuracy": ROUTE_COLORS["L"],
                    "donor_speaker_match": ROUTE_COLORS["P"],
                    "recipient_speaker_match": "#275dad",
                }
                x = np.arange(len(mode_order), dtype=float)
                width = .24
                for metric_index, metric in enumerate(metrics):
                    values, low_error, high_error = [], [], []
                    for mode in mode_order:
                        row = plot_frame[plot_frame["mode"] == mode].iloc[0]
                        value = float(row[metric])
                        low = float(row.get(f"{metric}_ci95_low", value))
                        high = float(row.get(f"{metric}_ci95_high", value))
                        values.append(value)
                        low_error.append(max(0.0, value - low))
                        high_error.append(max(0.0, high - value))
                    offsets = x + (metric_index - (len(metrics) - 1) / 2) * width
                    ax.bar(
                        offsets, values, width=width, label=metric_labels[metric],
                        color=metric_colors[metric], alpha=.88,
                        yerr=np.asarray([low_error, high_error]), capsize=3,
                    )
                ax.set(
                    xticks=x, xticklabels=[mode_labels[mode] for mode in mode_order],
                    ylim=(0, 1.06), ylabel="held-out rate",
                    title="Feature-level latent swap outcomes",
                )
                ax.legend(loc="upper center", ncol=max(1, len(metrics)), frameon=False)
            _plot(path, draw, figsize=(10.4, 5.6)); made.append(path.with_suffix(".png"))

    # Reports are regenerated in-place.  Remove plots and their backing CSVs
    # from older report layouts so a focused phone/speaker run does not retain
    # misleading, unlinked artifacts from a previous broad run.
    current = {path.stem for path in made}
    plot_dir = output / "plots"
    for pattern in ("*.png", "*.pdf"):
        for stale in plot_dir.glob(pattern):
            if stale.stem not in current:
                stale.unlink()
    plot_data_dir = output / "tables" / "plot_data"
    for stale in plot_data_dir.glob("*.csv"):
        if stale.stem not in current:
            stale.unlink()
    return made


def _spectrogram(bundle: AnalysisBundle, row: pd.Series, destination: Path) -> None:
    if destination.exists(): return
    import matplotlib.pyplot as plt
    audio = _read_audio(bundle.audio_path(row), bundle.spec.sample_rate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.specgram(audio, NFFT=512, Fs=bundle.spec.sample_rate, noverlap=384, cmap="magma")
    ax.set(xlabel="seconds", ylabel="Hz", ylim=(0, 8000))
    fig.tight_layout(); fig.savefig(destination, dpi=120); plt.close(fig)


def _activation_trace(cache: FeatureCache, unit: int, frame: int, destination: Path) -> None:
    import matplotlib.pyplot as plt
    ui = int(np.searchsorted(cache.offsets + cache.lengths, frame, side="right"))
    sl = cache.utterance_slice(ui)
    idx, val = cache.indices[sl], cache.values[sl].astype(np.float32)
    trace = np.where(idx == unit, val, 0).sum(1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data_path = destination.with_suffix(".csv")
    if not data_path.exists():
        pd.DataFrame({
            "utterance_frame": np.arange(len(trace), dtype=int),
            "absolute_frame": np.arange(int(cache.offsets[ui]), int(cache.offsets[ui]) + len(trace), dtype=int),
            "activation": trace.astype(float),
        }).to_csv(data_path, index=False)
    if destination.exists(): return
    fig, ax = plt.subplots(figsize=(9, 1.8))
    ax.plot(trace, color="#275dad", lw=1.5); ax.axhline(0, color="#667085", lw=.6)
    ax.set(xlabel="SPEAR frame", ylabel="activation", title=f"Unit {unit} activation trace")
    fig.tight_layout(); fig.savefig(destination, dpi=120); plt.close(fig)


def build_atlas(
    output: Path, cache: FeatureCache, bundle: AnalysisBundle, health: pd.DataFrame,
    examples: pd.DataFrame, scores: pd.DataFrame | None = None,
    causal: pd.DataFrame | None = None,
    *,
    include_spectrograms: bool = False,
    include_audio: bool = False,
    include_traces: bool = False,
) -> None:
    atlas = output / "report" / "units"
    assets = output / "report" / "assets" / "spectrograms"
    trace_assets = output / "report" / "assets" / "traces"
    atlas.mkdir(parents=True, exist_ok=True)
    metadata = bundle.utterances.copy(); metadata["utterance_id"] = metadata.utterance_id.astype(str)
    metadata = metadata.set_index("utterance_id")
    examples.to_csv(output / "tables" / "top_examples.csv", index=False)
    try: examples.to_parquet(output / "tables" / "top_examples.parquet", index=False)
    except Exception: pass
    if not (include_spectrograms or include_audio or include_traces):
        for stale in atlas.glob("*.html"):
            stale.unlink(missing_ok=True)
        return
    if "observed_active" in health.columns:
        atlas_units = health[health.observed_active.fillna(False)]
    else:
        atlas_units = health[~health.dead]
    if len(examples):
        atlas_units = atlas_units[atlas_units["unit"].isin(examples["unit"].astype(int).unique())]
    else:
        atlas_units = atlas_units.iloc[0:0]
    for _, unit in atlas_units.iterrows():
        uid = int(unit.unit)
        ex = examples[examples.unit == uid]
        assoc = scores[scores.unit == uid].sort_values("score", ascending=False).head(12) if scores is not None and len(scores) else pd.DataFrame()
        strongest = assoc.iloc[0] if len(assoc) else None
        ex_rows = []
        trace_html = ""
        if include_traces and len(ex):
            trace_path = trace_assets / f"{uid}.png"
            _activation_trace(cache, uid, int(ex.iloc[0].frame), trace_path)
            trace_html = f"<h2>Activation trace</h2><img class='plot panel' src='../assets/traces/{uid}.png'>"
        for _, e in ex.iterrows():
            media_html = ""
            if include_audio:
                media_html += (
                    f"<audio controls preload='none' "
                    f"src='{html.escape(Path(str(e['audio_path'])).as_uri())}'></audio>"
                )
            if include_spectrograms:
                spec_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(e.utterance_id)) + ".png"
                spec = assets / spec_name
                if str(e.utterance_id) in metadata.index:
                    _spectrogram(bundle, metadata.loc[str(e.utterance_id)], spec)
                rel_spec = f"../assets/spectrograms/{spec_name}"
                media_html += f"<img class='plot' src='{rel_spec}'>"
            false_positive = ""
            if strongest is not None and str(strongest.get("level", "")):
                observed = {"phone": e.get("phone", ""), "speaker_id": e.get("speaker_id", ""),
                            "emotion": e.get("emotion", "")}.get(str(strongest.get("factor", "")))
                if observed is not None and str(observed) != str(strongest.get("level")):
                    false_positive = " · <span class='warn'>candidate false positive</span>"
            ex_rows.append(f"""
              <div class='panel'><b>{html.escape(str(e.get('example_type','top')))} #{int(e['rank'])} {html.escape(str(e['utterance_id']))}</b>
              · phone <code>{html.escape(str(e['phone']))}</code> · activation {float(e['activation']):.4g}{false_positive}<br>
              {media_html}<p>{html.escape(str(e['transcript']))}</p></div>""")
        assoc_html = assoc.to_html(index=False, escape=True, classes="assoc") if len(assoc) else "<p>No factor scores.</p>"
        page = f"""<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style><title>Unit {uid}</title></head>
        <body><header><a style='color:white' href='../index.html'>← atlas</a><h1>Unit {uid} · <span class='{unit.route}'>{unit.route}</span></h1></header>
        <main><div class='cards'><div class='card'>frame frequency<br><b>{unit.frame_frequency:.4%}</b></div>
        <div class='card'>utterance frequency<br><b>{unit.utterance_frequency:.4%}</b></div>
        <div class='card'>route confidence<br><b>{unit.route_probability:.3f}</b></div>
        <div class='card'>mean active value<br><b>{unit.mean_when_active:.4g}</b></div></div>
        <h2>Factor associations</h2>{assoc_html}{trace_html}<h2>Activation examples</h2><div class='grid'>{''.join(ex_rows)}</div></main></body></html>"""
        (atlas / f"{uid}.html").write_text(page, encoding="utf-8")


def build_report(
    output: Path, resolved: ResolvedModel, completed: list[str], summaries: dict[str, Any],
    tables: dict[str, pd.DataFrame], warnings: list[str], plots: list[Path],
    *, profile: str = "full",
) -> Path:
    health = tables.get("health", pd.DataFrame())
    rows = []
    unit_page_dir = output / "report" / "units"
    has_unit_pages = unit_page_dir.exists() and any(unit_page_dir.glob("*.html"))
    if len(health):
        display_health = health
        if "observed_active" in display_health.columns:
            display_health = display_health[display_health["observed_active"].fillna(False)]
        for _, row in display_health.sort_values("frame_frequency", ascending=False).iterrows():
            unit = int(row.unit)
            if (unit_page_dir / f"{unit}.html").exists():
                unit_html = f"<a href='units/{unit}.html'>{unit}</a>"
            else:
                unit_html = str(unit)
            rows.append(f"<tr data-route='{row.route}'><td>{unit_html}</td>"
                        f"<td class='{row.route}'>{row.route}</td><td>{row.frame_frequency:.5%}</td>"
                        f"<td>{row.route_probability:.3f}</td><td>{row.mean_abs_contribution:.4g}</td></tr>")
    plot_parts = []
    for p in plots:
        key = p.stem
        caption = PLOT_CAPTIONS.get(key, "")
        title = PLOT_TITLES.get(key, key.replace('_', ' ').title())
        plot_parts.append(
            f"<div class='panel'><h3>{html.escape(title)}</h3>"
            f"<img class='plot' src='../{p.relative_to(output)}'>"
            + (f"<p class='muted'>{html.escape(caption)}</p>" if caption else "")
            + "</div>"
        )
    plot_html = "".join(plot_parts)
    warning_html = "".join(f"<p class='warn'>{html.escape(w)}</p>" for w in warnings)
    if profile == "quick":
        warning_html += (
            "<p class='warn'><b>Quick smoke test:</b> this report uses a small, "
            "speaker-balanced subset to verify the pipeline. Percentages, selected "
            "phones/speakers, and apparent zero leakage are exploratory and must be "
            "recomputed with <code>--profile full</code> before scientific use.</p>"
        )
    health_summary = summaries.get("health", {})
    dead_comparable = bool(health_summary.get("deadness_comparable_to_training", False))
    if health_summary and not dead_comparable:
        warning_html += (
            "<p class='warn'>Train-like deadness is not estimable in this run: "
            f"only {int(health_summary.get('deadness_analysis_batches', 0))} analysis batches "
            f"were observed; at least two windows of "
            f"{int(health_summary.get('deadness_threshold_batches', 0))} batches are required. "
            "Unobserved units are reported separately.</p>"
        )
    route_table_html = ""
    route_summary = tables.get("route_summary", pd.DataFrame())
    if len(route_summary):
        show = [c for c in (
            "route", "units", "active_units", "unobserved_fraction",
            "train_like_dead_fraction",
            "phone_selective_fraction", "speaker_selective_fraction",
            "mixed_phone_speaker_fraction",
            "route_violation_fraction",
        ) if c in route_summary.columns]
        route_table_html = (
            "<h2>Disentanglement summary by route</h2><div class='scroll'>"
            + route_summary[show].to_html(index=False, escape=True, classes="tight")
            + "</div>"
        )
    leaky_html = ""
    leaky = tables.get("leaky", pd.DataFrame())
    if len(leaky):
        show = [c for c in (
            "unit", "route", "issue_tags", "phone_like_score", "speaker_score",
            "leakage_score", "frame_frequency",
        ) if c in leaky.columns]
        leaky_html = (
            "<h2>Highest-risk mixed/leaky units</h2><p class='muted'>"
            "These are route violations or mixed phone/speaker units, sorted by the relevant leakage score.</p>"
            "<div class='scroll'>"
            + leaky.sort_values("leakage_score", ascending=False).head(30)[show].to_html(
                index=False, escape=True, classes="tight")
            + "</div>"
        )
    disent = summaries.get("disentanglement", {}).get("thesis_summary", {})
    thesis_cards = ""
    if disent:
        focus = str(disent.get("focus", "speaker_content"))
        l_leak_label = "L speaker/content leak" if focus == "speaker_content" else "L para leak"
        card_items = [
            ("L phone units", disent.get("L_phone_selective_fraction")),
            ("L speaker leak", disent.get("L_speaker_leak_fraction")),
            (l_leak_label, disent.get("L_speaker_content_leak_fraction", disent.get("L_paralinguistic_leak_fraction"))),
            ("P speaker units", disent.get("P_speaker_selective_fraction")),
            ("P phone leak", disent.get("P_phone_leak_fraction")),
            ("mixed phone/speaker", disent.get("mixed_phone_speaker_fraction_all")),
            ("route violations", disent.get("route_violation_fraction_all")),
        ]
        summary_title = "Smoke-test unit summary" if profile == "quick" else "Thesis-facing unit summary"
        thesis_cards = f"<h2>{summary_title}</h2><div class='cards'>" + "".join(
            f"<div class='card'>{html.escape(label)}<br><b>{value:.2%}</b></div>"
            for label, value in card_items if isinstance(value, (int, float)) and np.isfinite(value)
        ) + "</div>"
    score_summary = summaries.get("phone_speaker_scores", {})
    score_cards = ""
    if score_summary:
        cats = score_summary.get("categories", {})
        score_cards = "<h2>Positive phone/speaker association categories</h2><div class='cards'>" + "".join(
            f"<div class='card'>{html.escape(str(label))}<br><b>{int(value)}</b></div>"
            for label, value in sorted(cats.items())
        ) + "</div>"
    separation_html = ""
    separation = tables.get("representation_separation", pd.DataFrame())
    if len(separation):
        show = [c for c in (
            "target", "route", "labels", "points", "evaluation_points",
            "linear_probe_balanced_accuracy", "balanced_accuracy", "mean_cosine_margin",
        ) if c in separation.columns]
        probe_cards = ""
        if "linear_probe_balanced_accuracy" in separation.columns:
            def probe_value(target: str, route: str) -> float | None:
                match = separation[(separation["target"] == target) & (separation["route"] == route)]
                if match.empty:
                    return None
                value = match.iloc[0]["linear_probe_balanced_accuracy"]
                return float(value) if pd.notna(value) else None

            phone_l, phone_p = probe_value("phone", "L"), probe_value("phone", "P")
            speaker_l, speaker_p = probe_value("speaker_id", "L"), probe_value("speaker_id", "P")
            comparisons = []
            if phone_l is not None and phone_p is not None:
                comparisons.append(("phone probe L − P", phone_l - phone_p))
            if speaker_l is not None and speaker_p is not None:
                comparisons.append(("speaker probe P − L", speaker_p - speaker_l))
            if comparisons:
                probe_cards = "<div class='cards'>" + "".join(
                    f"<div class='card'>{html.escape(label)}<br><b>{value:+.3f}</b></div>"
                    for label, value in comparisons
                ) + "</div>"
        separation_html = (
            "<h2>Probing-style route-vector separation</h2>"
            "<p class='muted'>Frozen linear probes and nearest-centroid metrics use the same stratified "
            "holdout in the original route space. The expected disentanglement signature is positive "
            "phone L−P and speaker P−L. This is convergent unit-analysis evidence, not a replacement "
            "for the independent probing experiment. The phone and speaker probe-confusion figures below "
            "are row-normalized predictions on that untouched holdout: phone should be diagonal in L, "
            "speaker should be diagonal in P. The old 39-phone unit-coverage map is retained but explicitly "
            "deprecated because it measures unit firing coverage rather than classification.</p>" + probe_cards +
            "<div class='scroll'>" + separation[show].to_html(index=False, escape=True, classes="tight") + "</div>"
        )
    geometry_html = ""
    classifier_free_geometry = tables.get("classifier_free_geometry_summary", pd.DataFrame())
    if len(classifier_free_geometry):
        show = [c for c in (
            "target", "route", "labels", "controlled_pairs", "bootstrap_clusters",
            "mean_same_cosine", "mean_different_cosine", "paired_cosine_difference",
            "ci95_low", "ci95_high", "paired_same_greater_fraction", "rank_auc",
        ) if c in classifier_free_geometry.columns]
        geometry_html = (
            "<h2>Classifier-free full-space geometry</h2>"
            "<p class='muted'>No classifier is trained. Each anchor is matched to one same-label and "
            "one different-label observation. Phone comparisons cross speaker and utterance; speaker "
            "comparisons cross transcript/content and utterance. L and P use identical pairs. Confidence "
            "intervals are cluster bootstraps over the paired cosine difference.</p>"
            "<div class='scroll'>" + classifier_free_geometry[show].to_html(
                index=False, escape=True, classes="tight") + "</div>"
        )
    swap_html = ""
    swap_summary = tables.get("swap_summary", pd.DataFrame())
    if len(swap_summary):
        show = [c for c in (
            "mode", "pairs", "phone_recipient_accuracy",
            "donor_speaker_match", "recipient_speaker_match",
            "donor_speaker_probability", "recipient_speaker_probability",
            "reconstruction_shift_mse",
        ) if c in swap_summary.columns]
        swap_html = (
            "<h2>Feature-level latent swapping</h2>"
            "<p class='muted'>These are interventions on SAE latents reconstructed back into "
            "SPEAR feature space; they are not waveform or listening tests. The main intervention "
            "combines recipient L with donor P. Recipient-phone accuracy measures content "
            "preservation, while donor- and recipient-speaker scores measure identity transfer. "
            "The complementary donor-L/recipient-P swap is the main control. The shuffled-route "
            "mask diagnostic is retained, but it can overlap true P units—especially when a learned "
            "model assigns about half its capacity to P—so it must not be treated as a clean negative control. "
            "The independent phone and speaker evaluators are trained on unswapped SAE "
            "reconstructions from their fitting partition, never on swapped examples. Baseline "
            "performance therefore remains the required validity check.</p>"
            "<div class='scroll'>" + swap_summary[show].to_html(
                index=False, escape=True, classes="tight") + "</div>"
        )
    unit_section_title = "Unit atlas" if has_unit_pages else "Unit table"
    unit_section_note = (
        "<p class='muted'>Per-unit pages are generated only when atlas assets are requested "
        "with <code>--atlas-assets traces</code>, <code>spectrograms</code>, or "
        "<code>audio</code>. Default mode keeps the report table-only to avoid thousands "
        "of low-information HTML pages.</p>"
        if not has_unit_pages else ""
    )
    unit_section_note += (
        "<p class='muted'>The table displays only units observed in the extracted "
        "Top-K sample. Complete assigned-capacity rows remain in the CSV tables.</p>"
    )
    page = f"""<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style><title>SAE Unit Analysis</title></head>
    <body><header><h1>SAE Unit Analysis</h1><p>{html.escape(resolved.checkpoint.name)} · {', '.join(completed)}</p></header><main>
    {warning_html}<div class='cards'><div class='card'>K<br><b>{resolved.config['K']}</b></div>
    <div class='card'>active units<br><b>{summaries.get('health',{}).get('active_units','—')}</b></div>
    <div class='card'>unobserved units<br><b>{summaries.get('health',{}).get('unobserved_units','—')}</b></div>
    <div class='card'>train-like dead<br><b>{health_summary.get('train_like_dead_units','—') if dead_comparable else 'N/A'}</b></div>
    <div class='card'>frames<br><b>{summaries.get('health',{}).get('frames','—')}</b></div>
    <div class='card'>format<br><b>{resolved.source_format}</b></div></div>
    {thesis_cards}{score_cards}{separation_html}{geometry_html}{swap_html}{route_table_html}{leaky_html}
    <h2>Figures</h2><div class='grid'>{plot_html}</div><h2>{unit_section_title}</h2>{unit_section_note}
    <label>Route <select id='route'><option value=''>all</option><option>L</option><option>P</option><option>U</option></select></label>
    <label>Unit <input id='search' placeholder='unit id'></label>
    <table><thead><tr><th>unit</th><th>route</th><th>frame frequency</th><th>route confidence</th><th>contribution</th></tr></thead>
    <tbody id='units'>{''.join(rows)}</tbody></table>
    <script>function filt(){{let q=document.getElementById('search').value,r=document.getElementById('route').value;document.querySelectorAll('#units tr').forEach(x=>x.style.display=(!r||x.dataset.route===r)&&(!q||x.cells[0].innerText.includes(q))?'':'none')}}document.getElementById('route').onchange=filt;document.getElementById('search').oninput=filt</script>
    </main></body></html>"""
    path = output / "report" / "index.html"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
