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


def _plot(path: Path, draw) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    _sns()
    fig = plt.figure(figsize=(8.6, 5.2))
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
    if profiles is not None and len(profiles):
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
    disent = tables.get("disentanglement")
    if disent is not None and len(disent):
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
            ax.text(threshold + .03, ax.get_ylim()[1] * .92, "speaker-selective", fontsize=8, color="#17233c")
            ax.text(ax.get_xlim()[1] * .70, threshold + .03, "phone-selective", fontsize=8, color="#17233c")
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
        score_frame = active[score_cols].apply(pd.to_numeric, errors="coerce")
        score_frame = score_frame.loc[:, score_frame.notna().any(axis=0)]
        score_frame = score_frame.loc[:, score_frame.std(axis=0, skipna=True).fillna(0) > 0]
        if len(score_frame.columns) >= 2 and len(score_frame) >= 3:
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
            "prosody_selective_fraction", "metadata_paralinguistic_selective_fraction",
            "mixed_phone_speaker_fraction",
            "route_violation_fraction",
        ]
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
    if leaky is not None and len(leaky) and "leakage_score" in leaky:
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
    if "observed_active" in health.columns:
        atlas_units = health[health.observed_active.fillna(False)]
    else:
        atlas_units = health[~health.dead]
    for _, unit in atlas_units.iterrows():
        uid = int(unit.unit)
        ex = examples[examples.unit == uid]
        assoc = scores[scores.unit == uid].sort_values("score", ascending=False).head(12) if scores is not None and len(scores) else pd.DataFrame()
        strongest = assoc.iloc[0] if len(assoc) else None
        ex_rows = []
        trace_html = ""
        if len(ex):
            trace_path = trace_assets / f"{uid}.png"
            _activation_trace(cache, uid, int(ex.iloc[0].frame), trace_path)
            trace_html = f"<h2>Activation trace</h2><img class='plot panel' src='../assets/traces/{uid}.png'>"
        for _, e in ex.iterrows():
            spec_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(e.utterance_id)) + ".png"
            spec = assets / spec_name
            if str(e.utterance_id) in metadata.index:
                _spectrogram(bundle, metadata.loc[str(e.utterance_id)], spec)
            rel_spec = f"../assets/spectrograms/{spec_name}"
            false_positive = ""
            if strongest is not None and str(strongest.get("level", "")):
                observed = {"phone": e.get("phone", ""), "speaker_id": e.get("speaker_id", ""),
                            "emotion": e.get("emotion", "")}.get(str(strongest.get("factor", "")))
                if observed is not None and str(observed) != str(strongest.get("level")):
                    false_positive = " · <span class='warn'>candidate false positive</span>"
            ex_rows.append(f"""
              <div class='panel'><b>{html.escape(str(e.get('example_type','top')))} #{int(e['rank'])} {html.escape(str(e['utterance_id']))}</b>
              · phone <code>{html.escape(str(e['phone']))}</code> · activation {float(e['activation']):.4g}{false_positive}<br>
              <audio controls preload='none' src='{html.escape(Path(str(e['audio_path'])).as_uri())}'></audio>
              <img class='plot' src='{rel_spec}'><p>{html.escape(str(e['transcript']))}</p></div>""")
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
) -> Path:
    health = tables.get("health", pd.DataFrame())
    rows = []
    if len(health):
        for _, row in health.sort_values("frame_frequency", ascending=False).iterrows():
            rows.append(f"<tr data-route='{row.route}'><td><a href='units/{int(row.unit)}.html'>{int(row.unit)}</a></td>"
                        f"<td class='{row.route}'>{row.route}</td><td>{row.frame_frequency:.5%}</td>"
                        f"<td>{row.route_probability:.3f}</td><td>{row.mean_abs_contribution:.4g}</td></tr>")
    plot_html = "".join(f"<div class='panel'><img class='plot' src='../{p.relative_to(output)}'></div>" for p in plots)
    warning_html = "".join(f"<p class='warn'>{html.escape(w)}</p>" for w in warnings)
    route_table_html = ""
    route_summary = tables.get("route_summary", pd.DataFrame())
    if len(route_summary):
        show = [c for c in (
            "route", "units", "active_units", "unobserved_fraction",
            "train_like_dead_fraction",
            "phone_selective_fraction", "speaker_selective_fraction",
            "prosody_selective_fraction", "metadata_paralinguistic_selective_fraction",
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
            "prosody_score", "metadata_paralinguistic_score", "paralinguistic_score",
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
        thesis_cards = "<h2>Thesis-facing unit summary</h2><div class='cards'>" + "".join(
            f"<div class='card'>{html.escape(label)}<br><b>{value:.2%}</b></div>"
            for label, value in card_items if isinstance(value, (int, float)) and np.isfinite(value)
        ) + "</div>"
    page = f"""<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style><title>SAE Unit Analysis</title></head>
    <body><header><h1>SAE Unit Analysis</h1><p>{html.escape(resolved.checkpoint.name)} · {', '.join(completed)}</p></header><main>
    {warning_html}<div class='cards'><div class='card'>K<br><b>{resolved.config['K']}</b></div>
    <div class='card'>active units<br><b>{summaries.get('health',{}).get('active_units','—')}</b></div>
    <div class='card'>unobserved units<br><b>{summaries.get('health',{}).get('unobserved_units','—')}</b></div>
    <div class='card'>train-like dead<br><b>{summaries.get('health',{}).get('train_like_dead_units','—')}</b></div>
    <div class='card'>frames<br><b>{summaries.get('health',{}).get('frames','—')}</b></div>
    <div class='card'>format<br><b>{resolved.source_format}</b></div></div>
    {thesis_cards}{route_table_html}{leaky_html}
    <h2>Figures</h2><div class='grid'>{plot_html}</div><h2>Unit atlas</h2>
    <label>Route <select id='route'><option value=''>all</option><option>L</option><option>P</option><option>U</option></select></label>
    <label>Unit <input id='search' placeholder='unit id'></label>
    <table><thead><tr><th>unit</th><th>route</th><th>frame frequency</th><th>route confidence</th><th>contribution</th></tr></thead>
    <tbody id='units'>{''.join(rows)}</tbody></table>
    <script>function filt(){{let q=document.getElementById('search').value,r=document.getElementById('route').value;document.querySelectorAll('#units tr').forEach(x=>x.style.display=(!r||x.dataset.route===r)&&(!q||x.cells[0].innerText.includes(q))?'':'none')}}document.getElementById('route').onchange=filt;document.getElementById('search').oninput=filt</script>
    </main></body></html>"""
    path = output / "report" / "index.html"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
