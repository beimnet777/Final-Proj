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
"""


def _plot(path: Path, draw) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 4.8))
    draw(fig)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def make_plots(output: Path, tables: dict[str, pd.DataFrame]) -> list[Path]:
    made = []
    health = tables.get("health")
    if health is not None and len(health):
        path = output / "plots" / "route_activity"
        def draw(fig):
            ax = fig.subplots()
            for route, group in health[~health.dead].groupby("route"):
                ax.hist(np.log10(group.frame_frequency.clip(lower=1e-9)), bins=40, alpha=.55, label=route)
            ax.set(xlabel="log10 frame firing frequency", ylabel="units", title="Active-unit firing distribution")
            ax.legend()
        _plot(path, draw); made.append(path.with_suffix(".png"))
    profiles = tables.get("profiles")
    if profiles is not None and len(profiles):
        path = output / "plots" / "unit_factor_alignment"
        def draw(fig):
            ax = fig.subplots()
            colors = {"L":"#087f5b", "P":"#d9480f", "U":"#7048e8", "unassigned":"#667085"}
            for route, group in profiles.groupby("route"):
                ax.scatter(group.linguistic_score, group.paralinguistic_score, s=8, alpha=.45,
                           label=route, color=colors.get(route))
            ax.set(xlabel="linguistic selectivity", ylabel="paralinguistic selectivity",
                   title="Unit factor alignment")
            ax.legend()
        _plot(path, draw); made.append(path.with_suffix(".png"))
    causal = tables.get("causal")
    if causal is not None and len(causal) and "target_delta" in causal:
        path = output / "plots" / "causal_curves"
        def draw(fig):
            ax = fig.subplots()
            for (family, mode), g in causal.dropna(subset=["target_delta"]).groupby(["family", "mode"]):
                ax.plot(g.budget, g.target_delta, marker="o", label=f"{family}/{mode}")
            ax.axhline(0, color="black", lw=.8); ax.set(xscale="log", xlabel="units", ylabel="target metric delta",
                                                        title="Necessity, sufficiency, and controls")
            ax.legend(fontsize=8)
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
    if destination.exists(): return
    import matplotlib.pyplot as plt
    ui = int(np.searchsorted(cache.offsets + cache.lengths, frame, side="right"))
    sl = cache.utterance_slice(ui)
    idx, val = cache.indices[sl], cache.values[sl].astype(np.float32)
    trace = np.where(idx == unit, val, 0).sum(1)
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    for _, unit in health[~health.dead].iterrows():
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
    page = f"""<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style><title>SAE Unit Analysis</title></head>
    <body><header><h1>SAE Unit Analysis</h1><p>{html.escape(resolved.checkpoint.name)} · {', '.join(completed)}</p></header><main>
    {warning_html}<div class='cards'><div class='card'>K<br><b>{resolved.config['K']}</b></div>
    <div class='card'>active units<br><b>{summaries.get('health',{}).get('active_units','—')}</b></div>
    <div class='card'>frames<br><b>{summaries.get('health',{}).get('frames','—')}</b></div>
    <div class='card'>format<br><b>{resolved.source_format}</b></div></div>
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
