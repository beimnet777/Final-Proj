"""Build one final, self-contained unit-analysis dashboard per model family.

The consolidated reports do not recompute scientific quantities. They collect
the final-checkpoint 12k structural report, the registered 5k Swap-v2 report,
and the shared-sample organization trajectory into a stable presentation layer.
All source reports remain untouched and are linked for provenance.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelReport:
    slug: str
    label: str
    short_label: str
    status: str
    interpretation: str
    main_result: str
    swap_result: str | None
    trajectory_result: str | None
    color: str


MODELS = (
    ModelReport(
        "fixed_routing", "Fixed routing · 12k final", "Fixed", "Primary control",
        "The fixed 240L/16P partition is the feasibility control. It shows the clearest "
        "functional phone/speaker double dissociation, but its L route has severe inactive capacity.",
        "libri_fixed_240L16P_step12000_final_12k_mps",
        "libri_fixed_240L16P_step12000_final_swapv2_5k_mps", "fixed_routing", "#3155a4",
    ),
    ModelReport(
        "naive_learned", "Naive learned routing · 12k final", "Naive", "Negative control",
        "Learning routes forever keeps broad unit coverage but does not localize speaker identity "
        "cleanly to P. It is the moving-target negative control, not a successful learned router.",
        "libri_learned_naive_step12000_final_12k_mps",
        "libri_learned_naive_step12000_final_swapv2_5k_mps", "naive_learned", "#767676",
    ),
    ModelReport(
        "quota_freeze", "Quota-freeze routing · 20k final", "Quota-freeze", "Primary learned result",
        "Freezing the learned quota at 4k stabilizes the route assignment. The final model recovers "
        "the intended functional double dissociation, although observed L capacity still contracts.",
        "libri_learned_qfreeze4k_step20000_final_12k_mps",
        "libri_learned_qfreeze4k_step20000_final_swapv2_5k_mps", "quota_freeze", "#1b9e77",
    ),
    ModelReport(
        "post_gp", "Quota-freeze + post-GP · 20k final", "Post-GP", "Learned variant",
        "The post-GP continuation preserves strong functional routing and gives the strongest donor-"
        "speaker transfer among the learned variants, with the same L-capacity caveat.",
        "libri_learned_qfreeze4k_postgp030_step20000_final_12k_mps",
        "libri_learned_qfreeze4k_postgp030_step20000_final_swapv2_5k_mps", "post_gp", "#d95f02",
    ),
    ModelReport(
        "ramp_5k", "Quota-freeze + 5k ramp · 20k final", "Ramp-5k", "Learned variant",
        "The slower adversarial ramp also produces the intended functional allocation. Its result is "
        "strong but slightly weaker than post-GP on the registered identity-transfer intervention.",
        "libri_learned_qfreeze4k_ramp5k_step20000_final_12k_mps",
        "libri_learned_qfreeze4k_ramp5k_step20000_final_swapv2_5k_mps", "ramp_5k", "#e6ab02",
    ),
    ModelReport(
        "unrouted_baseline", "Unrouted Top-K baseline · 12k final", "Unrouted", "Capacity control",
        "The reconstruction-only baseline keeps essentially all units observable, but it has no L/P "
        "partition. Route-level MIG/SAP/DCI and route swapping are therefore undefined by design.",
        "libri_unrouted_baseline_step12000_final_12k_mps", None, None, "#8c510a",
    ),
)


ROUTED_MAIN_PLOTS = (
    ("route_activity", "Route activity and observed capacity",
     "How frequently observed units enter frame-level Top-K. The 12k health table, not visual density alone, defines training-comparable deadness."),
    ("phone_score_vs_speaker_score", "Unit-level phone and speaker association",
     "Each point is an observed unit. PhoneScore uses positive directional activity AUROC; SpeakerScore uses positive utterance-level point-biserial correlation."),
    ("phone_unit_alignment_ranked", "Held-out 39-phone unit alignment atlas",
     "Unique phone–unit assignments are selected on train+validation and evaluated on untouched test frames. Filled markers are held-out specificity margins."),
    ("route_selectivity_composition", "Route selectivity composition",
     "Fractions use observed-active units as the denominator; assigned-capacity fractions remain available in the copied CSV tables."),
    ("route_probe_accuracy", "Held-out phone–speaker probe crossover",
     "Quantitative route-vector evidence: phone decoding should favour L and speaker decoding should favour P."),
    ("route_classifier_free_geometry", "Classifier-free full-space geometry",
     "Controlled cosine pairs use identical observations in both routes. Phone pairs cross speakers; speaker pairs cross content."),
    ("route_factor_information", "Grouped route MIG, SAP and DCI",
     "These scores compare the complete zL and zP subspaces on identical held-out observations; they are not within-route unit-independence scores."),
    ("route_factor_contrasts", "Desired-route contrasts and capacity control",
     "Positive contrasts mean phone favours zL or speaker favours zP. Capacity-matched estimates check whether the result is only route size."),
    ("route_phone_probe_confusion", "Held-out phone probe confusion",
     "A true classification confusion matrix. The intended signature is a strong L diagonal and a diffuse P matrix."),
    ("route_speaker_probe_confusion", "Held-out speaker probe confusion",
     "A true classification confusion matrix. The intended signature is a diffuse L matrix and a strong P diagonal."),
    ("route_phone_representation_embedding", "Phone PCA",
     "Centered linear/global projection of the same held-out phone observations in L and P."),
    ("route_phone_representation_umap", "Phone UMAP",
     "Supplementary cosine-UMAP of the PCA observations. Local gaps are descriptive, not a statistical test."),
    ("route_speaker_representation_embedding", "Speaker PCA",
     "Centered linear/global projection of identical held-out speaker observations in L and P."),
    ("route_speaker_representation_umap", "Speaker UMAP",
     "Supplementary local-neighbourhood view. Quantitative claims use probes and full-space geometry."),
)

UNROUTED_MAIN_PLOTS = (
    ("unit_activity", "Unit activity", "Observed activity across the single unrouted Top-K space."),
    ("phone_score_vs_speaker_score", "Unit-level phone and speaker association",
     "The same association definitions as routed models, but without an L/P assignment."),
    ("phone_unit_alignment_ranked", "Held-out 39-phone unit alignment atlas",
     "Unique phone–unit assignments selected on train+validation and evaluated on test."),
    ("unrouted_selectivity_composition", "Unrouted selectivity composition",
     "Phone-, speaker-, mixed-, and other-unit fractions in the observed unrouted space."),
    ("unrouted_top_associated_units", "Top associated units",
     "Compact view of the strongest phone- and speaker-associated units."),
)

SWAP_PLOTS = (
    ("latent_swap_outcomes", "Registered latent-swap outcomes",
     "Across 250 fixed pairs, P swapping should retain recipient phone content and match donor speaker; the complementary L swap tests the reverse intervention."),
    ("latent_swap_paired_effects", "Paired intervention effects",
     "Pairwise changes relative to each recipient baseline, with recipient and donor speaker clustering respected by the intervals."),
    ("latent_swap_interpolation", "P-route interpolation",
     "A graded recipient-to-donor P intervention. A smooth identity transition strengthens the causal interpretation beyond an endpoint-only swap."),
)

TRAJECTORY_PLOTS = (
    ("trajectory_metrics", "Organization trajectory",
     "Route capacity, fixed-sample observed coverage, phone/speaker association contrasts, and route stability across training."),
    ("unit_association_snapshots", "Unit organization snapshots",
     "PhoneScore versus SpeakerScore at each checkpoint using the same 40,000 stored SPEAR frames."),
    ("unit_fate_heatmap", "Unit fate through training",
     "Identity-resolved route/category states. Unobserved means absent on the fixed analysis sample, not necessarily historically dead during training."),
)

MAIN_TABLES = (
    "route_disentanglement_summary.csv", "thesis_disentanglement_summary.csv",
    "speech_factor_metrics.csv", "route_dci_evidence.csv",
    "route_classifier_free_geometry_summary.csv", "route_representation_separation.csv",
    "unit_phone_speaker_scores.csv", "phone_selected_units.csv",
    "phone_units_ranked.csv", "speaker_units_ranked.csv", "unrouted_unit_summary.csv",
)
SWAP_TABLES = (
    "swap_mode_summary.csv", "swap_pair_contrasts.csv", "swap_content_speaker_grid.csv",
    "swap_pairs.csv", "swaps.csv",
)
TRAJECTORY_TABLES = ("trajectory_metrics.csv", "state_transitions.csv", "unit_trajectories.csv")
MAIN_METADATA = (
    "run_manifest.json", "resolved_model.json", "health.json", "selectivity.json",
    "phone_speaker_scores.json", "classifier_free_geometry.json",
    "route_representation_embeddings.json", "speech_factor_metrics.json", "summary.json",
)
SWAP_METADATA = ("run_manifest.json", "swap.json", "summary.json")
TRAJECTORY_METADATA = ("trajectory_manifest.json",)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3, percent: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{100 * float(value):.{digits - 1}f}%" if percent else f"{float(value):.{digits}f}"


def _rel_link(source: Path, report_dir: Path) -> str:
    return html.escape(os.path.relpath(source.resolve(), report_dir.resolve()))


def _copy(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _factor_contrasts(main: Path) -> dict[str, float]:
    path = main / "tables" / "speech_factor_metrics.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    rows = table[
        (table["component"] == "route_contrast")
        & (table["capacity_mode"] == "all_observed")
        & (table["control"] == "observed")
    ]
    out: dict[str, float] = {}
    for _, row in rows.iterrows():
        out[f"{row['metric']}_{row['target']}"] = float(row["value"])
    alignment = table[
        (table["component"] == "directional_alignment")
        & (table["target"] == "mean")
        & (table["capacity_mode"] == "all_observed")
        & (table["control"] == "observed")
    ]
    if not alignment.empty:
        out["DCI_alignment"] = float(alignment.iloc[0]["value"])
    return out


def _health(main: Path) -> dict[str, Any]:
    return _read_json(main / "health.json") if (main / "health.json").exists() else {}


def _swap_metrics(swap: Path | None) -> dict[str, float]:
    if swap is None:
        return {}
    path = swap / "tables" / "swap_mode_summary.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    baseline = table[table["mode"] == "baseline"].iloc[0]
    p_swap = table[table["mode"] == "P_from_donor"].iloc[0]
    l_swap = table[table["mode"] == "L_from_donor"].iloc[0]
    return {
        "baseline_phone": float(baseline["phone_recipient_accuracy"]),
        "P_phone": float(p_swap["phone_recipient_accuracy"]),
        "P_phone_retention": float(p_swap["phone_recipient_accuracy"] / baseline["phone_recipient_accuracy"]),
        "P_donor": float(p_swap["donor_speaker_match"]),
        "L_phone": float(l_swap["phone_recipient_accuracy"]),
        "L_phone_replacement": float(1 - l_swap["phone_recipient_accuracy"] / baseline["phone_recipient_accuracy"]),
        "L_recipient": float(l_swap["recipient_speaker_match"]),
        "pairs": int(p_swap["pairs"]),
    }


def _route_health(health: dict[str, Any], route: str, field: str) -> float | None:
    for row in health.get("route_summary", []):
        if str(row.get("route")) == route:
            value = row.get(field)
            return None if value is None else float(value)
    return None


def _route_record(health: dict[str, Any], route: str) -> dict[str, Any]:
    return next((row for row in health.get("route_summary", []) if str(row.get("route")) == route), {})


def _cards(model: ModelReport, metrics: dict[str, Any]) -> str:
    health, factor, swap = metrics["health"], metrics["factor"], metrics["swap"]
    values = [
        ("Observed units", f"{int(health.get('active_units', 0)):,}/{int(health.get('K', 0)):,}"),
        ("Training-like dead", _fmt(health.get("train_like_dead_fraction"), percent=True)),
    ]
    if model.swap_result:
        values.extend([
            ("DCI phone L−P", _fmt(factor.get("DCI_phone"))),
            ("DCI speaker P−L", _fmt(factor.get("DCI_speaker_id"))),
            ("P-swap donor match", _fmt(swap.get("P_donor"), percent=True)),
            ("L-swap recipient match", _fmt(swap.get("L_recipient"), percent=True)),
        ])
    return "".join(
        f"<div class='metric'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>"
        for label, value in values
    )


def _metric_table(model: ModelReport, metrics: dict[str, Any]) -> str:
    health, factor, swap = metrics["health"], metrics["factor"], metrics["swap"]
    rows = [
        ("12k observed units", f"{int(health.get('active_units', 0)):,} / {int(health.get('K', 0)):,}",
         "Units selected at least once across the complete 12k extraction."),
        ("Training-comparable dead fraction", _fmt(health.get("train_like_dead_fraction"), percent=True),
         "Frozen-checkpoint replay of the 256-batch training deadness window; not reconstructed historical counters."),
    ]
    if model.swap_result:
        l_route, p_route = _route_record(health, "L"), _route_record(health, "P")
        rows.extend([
            ("L observed / assigned", f"{int(l_route.get('active_units', 0)):,} / {int(l_route.get('assigned_units', 0)):,}",
             "Observed capacity in the linguistic route across the complete extraction."),
            ("P observed / assigned", f"{int(p_route.get('active_units', 0)):,} / {int(p_route.get('assigned_units', 0)):,}",
             "Observed capacity in the paralinguistic route across the complete extraction."),
            ("Selected slots per frame (L / P)",
             f"{float(l_route.get('active_slots_per_frame', 0)):.1f} / {float(p_route.get('active_slots_per_frame', 0)):.1f}",
             "Mean allocation of the 256 frame-level Top-K selections."),
            ("L-route dead fraction", _fmt(_route_health(health, "L", "train_like_dead_fraction"), percent=True),
             "Assigned L capacity inactive over the replay window."),
            ("P-route dead fraction", _fmt(_route_health(health, "P", "train_like_dead_fraction"), percent=True),
             "Assigned P capacity inactive over the replay window."),
            ("MIG phone L−P", _fmt(factor.get("MIG_phone")), "Positive means phone information favours the complete zL vector."),
            ("MIG speaker P−L", _fmt(factor.get("MIG_speaker_id")), "Positive means speaker information favours the complete zP vector."),
            ("SAP phone L−P", _fmt(factor.get("SAP_phone")), "Held-out linear balanced-accuracy contrast."),
            ("SAP speaker P−L", _fmt(factor.get("SAP_speaker_id")), "Held-out linear balanced-accuracy contrast."),
            ("DCI phone L−P", _fmt(factor.get("DCI_phone")), "Held-out nonlinear informativeness contrast."),
            ("DCI speaker P−L", _fmt(factor.get("DCI_speaker_id")), "Held-out nonlinear informativeness contrast."),
            ("P-swap phone retention", _fmt(swap.get("P_phone_retention"), percent=True),
             "P-swap phone accuracy divided by unswapped baseline accuracy."),
            ("P-swap donor-speaker match", _fmt(swap.get("P_donor"), percent=True),
             "Registered 250-pair feature intervention."),
            ("L-swap phone replacement", _fmt(swap.get("L_phone_replacement"), percent=True),
             "One minus L-swap recipient-phone accuracy divided by baseline."),
            ("L-swap recipient-speaker match", _fmt(swap.get("L_recipient"), percent=True),
             "Complementary intervention identity control."),
        ])
    body = "".join(
        f"<tr><td>{html.escape(name)}</td><td><strong>{html.escape(value)}</strong></td><td>{html.escape(note)}</td></tr>"
        for name, value, note in rows
    )
    return f"<div class='scroll'><table><thead><tr><th>Measure</th><th>Value</th><th>Interpretation</th></tr></thead><tbody>{body}</tbody></table></div>"


def _copy_plot_group(
    source: Path, specs: tuple[tuple[str, str, str], ...], destination: Path, prefix: str,
) -> list[tuple[str, str, str]]:
    copied = []
    for stem, title, caption in specs:
        src = source / "plots" / f"{stem}.png"
        relative = Path("plots") / f"{prefix}_{stem}.png"
        if _copy(src, destination / relative):
            copied.append((str(relative), title, caption))
        pdf = source / "plots" / f"{stem}.pdf"
        _copy(pdf, destination / "plots" / f"{prefix}_{stem}.pdf")
    return copied


def _figure_grid(figures: list[tuple[str, str, str]]) -> str:
    return "<div class='figure-grid'>" + "".join(
        "<figure><a href='../" + html.escape(path) + "'><img loading='lazy' src='../" + html.escape(path) + "'></a>"
        f"<figcaption><strong>{html.escape(title)}</strong><br>{html.escape(caption)}</figcaption></figure>"
        for path, title, caption in figures
    ) + "</div>"


def _copy_tables(source: Path, names: tuple[str, ...], destination: Path, prefix: str) -> list[tuple[str, str]]:
    links = []
    for name in names:
        src = source / "tables" / name
        dst = destination / "tables" / f"{prefix}_{name}"
        if _copy(src, dst):
            links.append((f"../tables/{dst.name}", f"{prefix}: {name}"))
    return links


def _copy_metadata(source: Path, names: tuple[str, ...], destination: Path, prefix: str) -> list[tuple[str, str]]:
    links = []
    for name in names:
        src = source / name
        dst = destination / "metadata" / f"{prefix}_{name}"
        if _copy(src, dst):
            links.append((f"../metadata/{dst.name}", f"{prefix}: {name}"))
    return links


def _source_links(model: ModelReport, results: Path, report_dir: Path) -> str:
    links = [(results / model.main_result / "report" / "index.html", "Complete 12k final-checkpoint report")]
    if model.swap_result:
        links.append((results / model.swap_result / "report" / "index.html", "Complete registered Swap-v2 report"))
    if model.trajectory_result:
        links.append((results / "unit_organization_trajectories_5k_shared_sample" / model.trajectory_result / "report" / "index.html", "Complete organization-trajectory report"))
    return "<ul>" + "".join(
        f"<li><a href='{_rel_link(path, report_dir)}'>{html.escape(label)}</a></li>" for path, label in links
    ) + "</ul>"


def _style() -> str:
    return """
    :root{--ink:#162033;--muted:#5e6879;--line:#dfe5ef;--paper:#fff;--bg:#f4f7fb;--accent:#3155a4}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;line-height:1.55}
    main{max-width:1500px;margin:auto;padding:30px}.hero{padding:34px;border-radius:20px;color:white;background:linear-gradient(125deg,#162b57,var(--accent) 62%,#4796a8);box-shadow:0 16px 40px #19315b25}
    h1{font-size:clamp(2rem,4vw,3.5rem);line-height:1.05;margin:.2rem 0 1rem} h2{margin-top:2.2rem;font-size:1.6rem} h3{margin:.2rem 0 .6rem} p{max-width:1000px}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.75rem;font-weight:800;opacity:.8}
    .badge{display:inline-block;padding:.32rem .65rem;border-radius:999px;background:#eaf0ff;color:#24488f;font-weight:750;font-size:.8rem}.hero .badge{background:#ffffff24;color:#fff}
    .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin:22px 0}.metric,.panel,figure,.model-card{background:var(--paper);border:1px solid var(--line);border-radius:14px;box-shadow:0 4px 16px #20304c0a}.metric{padding:15px}.metric span{display:block;color:var(--muted);font-size:.82rem}.metric strong{display:block;font-size:1.45rem;margin-top:.25rem}
    .panel{padding:20px;margin:16px 0}.note{border-left:4px solid #e6ab02;background:#fff9e6;padding:14px 16px;border-radius:6px}.good{border-left-color:#1b9e77;background:#edf9f5}.muted{color:var(--muted)}
    .figure-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px}.figure-grid figure{margin:0;padding:14px;overflow:hidden}.figure-grid img{display:block;width:100%;height:auto;border-radius:8px}.figure-grid figcaption{font-size:.9rem;color:var(--muted);padding:11px 4px 3px}.figure-grid figcaption strong{color:var(--ink)}
    table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#eef3fb;font-size:.82rem;text-transform:uppercase;letter-spacing:.04em}.scroll{overflow:auto;border:1px solid var(--line);border-radius:12px}
    a{color:#3155a4;text-decoration:none}a:hover{text-decoration:underline}.nav{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}.nav a{background:white;border:1px solid var(--line);padding:8px 12px;border-radius:9px}.model-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.model-card{padding:20px;border-top:5px solid var(--model-color)}.model-card p{color:var(--muted)}.equation{font-family:ui-monospace,SFMono-Regular,monospace;background:#eef3fb;border-radius:9px;padding:12px;margin:9px 0;overflow:auto}
    @media(max-width:600px){main{padding:16px}.hero{padding:24px}.figure-grid{grid-template-columns:1fr}}
    """


def _model_page(model: ModelReport, results: Path, output: Path) -> dict[str, Any]:
    main = results / model.main_result
    swap = results / model.swap_result if model.swap_result else None
    trajectory = (
        results / "unit_organization_trajectories_5k_shared_sample" / model.trajectory_result
        if model.trajectory_result else None
    )
    for required in (main / "report" / "index.html", main / "health.json"):
        if not required.exists():
            raise FileNotFoundError(f"Missing required final evidence for {model.label}: {required}")
    if swap is not None and not (swap / "report" / "index.html").exists():
        raise FileNotFoundError(f"Missing Swap-v2 evidence for {model.label}: {swap}")
    if trajectory is not None and not (trajectory / "report" / "index.html").exists():
        raise FileNotFoundError(f"Missing trajectory evidence for {model.label}: {trajectory}")

    model_dir = output / model.slug
    report_dir = model_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    health = _health(main)
    factor = _factor_contrasts(main)
    swap_values = _swap_metrics(swap)
    metrics = {"health": health, "factor": factor, "swap": swap_values}

    main_specs = ROUTED_MAIN_PLOTS if model.swap_result else UNROUTED_MAIN_PLOTS
    main_figures = _copy_plot_group(main, main_specs, model_dir, "main")
    swap_figures = _copy_plot_group(swap, SWAP_PLOTS, model_dir, "swap") if swap else []
    trajectory_figures = _copy_plot_group(trajectory, TRAJECTORY_PLOTS, model_dir, "trajectory") if trajectory else []
    table_links = _copy_tables(main, MAIN_TABLES, model_dir, "main")
    if swap:
        table_links.extend(_copy_tables(swap, SWAP_TABLES, model_dir, "swapv2"))
    if trajectory:
        table_links.extend(_copy_tables(trajectory, TRAJECTORY_TABLES, model_dir, "trajectory"))
    table_html = "<ul>" + "".join(
        f"<li><a href='{html.escape(path)}'>{html.escape(label)}</a></li>" for path, label in table_links
    ) + "</ul>"
    metadata_links = _copy_metadata(main, MAIN_METADATA, model_dir, "main")
    if swap:
        metadata_links.extend(_copy_metadata(swap, SWAP_METADATA, model_dir, "swapv2"))
    if trajectory:
        metadata_links.extend(_copy_metadata(trajectory, TRAJECTORY_METADATA, model_dir, "trajectory"))
    metadata_html = "<ul>" + "".join(
        f"<li><a href='{html.escape(path)}'>{html.escape(label)}</a></li>" for path, label in metadata_links
    ) + "</ul>"

    intervention = ""
    trajectory_section = ""
    if swap_figures:
        intervention = (
            "<h2>4. Registered feature interventions</h2>"
            "<p>The 5k Swap-v2 protocol uses the same 250 recipient/donor pairs for every routed model. "
            "Evaluators are fitted only on disjoint unswapped reconstructions. These are reconstructed "
            "SPEAR features, not generated waveform evidence.</p>" + _figure_grid(swap_figures)
        )
    if trajectory_figures:
        trajectory_section = (
            "<h2>5. Organization through training</h2>"
            "<p>All snapshots use the same 40,000 raw-SPEAR frames sampled from 5,000 utterances. "
            "Unit identities are tracked only within this training family; they are not matched across independently trained models.</p>"
            + _figure_grid(trajectory_figures)
        )

    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>{html.escape(model.label)} · Final SAE unit analysis</title><style>{_style()}</style></head><body><main>
    <section class='hero' style='--accent:{model.color}'><span class='eyebrow'>Final consolidated SAE unit analysis</span>
    <h1>{html.escape(model.label)}</h1><span class='badge'>{html.escape(model.status)}</span>
    <p>{html.escape(model.interpretation)}</p></section>
    <nav class='nav'><a href='../../report/index.html'>← All models</a><a href='#numbers'>Numbers</a><a href='#figures'>Figures</a><a href='#sources'>Sources and tables</a></nav>
    <div class='metrics'>{_cards(model, metrics)}</div>
    <section class='panel'><h2>What is combined here</h2><p>The structural/unit report uses all 12,000 utterances
    (10k train, 1k validation, 1k untouched test; 7.03M frames). The extensive intervention uses the registered
    5k subset and 250 pairs. The trajectory uses a fixed 40k-frame sample. The sample size is stated beside each
    evidence family so values from different protocols are not silently pooled.</p></section>
    <h2 id='numbers'>1. Final numerical summary</h2>{_metric_table(model, metrics)}
    <h2 id='figures'>2. Unit health and association</h2>{_figure_grid(main_figures[:4] if model.swap_result else main_figures)}
    {"<h2>3. Route-level held-out evidence</h2><p>Probes, classifier-free geometry, PCA/UMAP, and grouped MIG/SAP/DCI all evaluate complete zL and zP vectors. The projections are descriptive; the quantitative tests remain in the original route spaces.</p>" + _figure_grid(main_figures[4:]) if model.swap_result else "<section class='panel note'><h2>3. Route evidence is not defined</h2><p>This checkpoint has one unrouted Top-K space. It is a capacity/organization baseline and cannot support L-versus-P metrics or route interventions.</p></section>"}
    {intervention}{trajectory_section}
    <h2 id='sources'>6. Source reports and copied tables</h2><section class='panel'><h3>Complete source reports</h3>{_source_links(model, results, report_dir)}
    <h3>Copied final tables</h3>{table_html}<h3>Copied manifests and metadata</h3>{metadata_html}</section>
    <section class='panel note'><h3>Scope boundary</h3><p>This page consolidates unit- and feature-level evidence. Voice conversion remains separate until the direct vocoder passes reconstruction and listening gates.</p></section>
    </main></body></html>"""
    (report_dir / "index.html").write_text(page, encoding="utf-8")
    manifest = {
        "format": "sae_final_consolidated_model_v1", "model": model.label,
        "status": model.status, "main_result": str(main.resolve()),
        "swap_result": str(swap.resolve()) if swap else None,
        "trajectory_result": str(trajectory.resolve()) if trajectory else None,
        "protocols": {"structural_utterances": 12000, "swap_utterances": 5000 if swap else None,
                      "swap_pairs": int(swap_values.get("pairs", 0)) if swap else None,
                      "trajectory_frames": 40000 if trajectory else None},
        "metrics": metrics,
    }
    (model_dir / "consolidated_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"model": model, "metrics": metrics, "report": report_dir / "index.html"}


def _cross_model_plots(records: list[dict[str, Any]], output: Path) -> None:
    routed = [record for record in records if record["model"].swap_result]
    labels = [record["model"].short_label for record in routed]
    colors = [record["model"].color for record in routed]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), sharey=True)
    for axis, target, title in zip(axes, ("phone", "speaker_id"), ("Phone → L", "Speaker → P")):
        for offset, metric, marker in ((-.18, "MIG", "o"), (0, "SAP", "s"), (.18, "DCI", "D")):
            values = [record["metrics"]["factor"].get(f"{metric}_{target}", np.nan) for record in routed]
            axis.scatter(values, y + offset, marker=marker, s=75, label=metric, edgecolor="white", linewidth=.7,
                         c=colors, zorder=3)
        axis.axvline(0, color="#9da7b5", lw=1)
        axis.grid(axis="x", color="#e5eaf1")
        axis.set_title(title, weight="bold")
        axis.set_xlabel("Desired-route contrast")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, ncol=3, loc="upper right")
    fig.suptitle("Whole-route information at the final checkpoint", fontsize=15, weight="bold")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / "plots" / f"cross_model_route_evidence.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), sharey=True)
    for i, record in enumerate(routed):
        health, swap = record["metrics"]["health"], record["metrics"]["swap"]
        axes[0].barh(i - .16, _route_health(health, "L", "train_like_dead_fraction"), height=.3,
                     color="#3155a4", label="L" if i == 0 else None)
        axes[0].barh(i + .16, _route_health(health, "P", "train_like_dead_fraction"), height=.3,
                     color="#d95f02", label="P" if i == 0 else None)
        axes[1].barh(i - .16, swap.get("P_donor", np.nan), height=.3,
                     color="#1b9e77", label="P swap → donor" if i == 0 else None)
        axes[1].barh(i + .16, swap.get("L_recipient", np.nan), height=.3,
                     color="#7570b3", label="L swap → recipient" if i == 0 else None)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].tick_params(axis="y", labelleft=True)
    axes[0].invert_yaxis()
    axes[0].set_title("Training-comparable route deadness", weight="bold")
    axes[1].set_title("Registered identity intervention", weight="bold")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.grid(axis="x", color="#e5eaf1")
        axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2)
    axes[0].set_xlabel("Dead fraction"); axes[1].set_xlabel("Match fraction")
    fig.suptitle("Capacity health and functional routing are distinct", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, .08, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / "plots" / f"cross_model_health_swap.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _root_page(records: list[dict[str, Any]], results: Path, output: Path) -> None:
    cards = "".join(
        f"<article class='model-card' style='--model-color:{record['model'].color}'><span class='badge'>{html.escape(record['model'].status)}</span>"
        f"<h3>{html.escape(record['model'].label)}</h3><p>{html.escape(record['model'].interpretation)}</p>"
        f"<a href='../{record['model'].slug}/report/index.html'>Open consolidated report →</a></article>"
        for record in records
    )
    rows = []
    for record in records:
        model, metrics = record["model"], record["metrics"]
        health, factor, swap = metrics["health"], metrics["factor"], metrics["swap"]
        rows.append(
            f"<tr><td><a href='../{model.slug}/report/index.html'>{html.escape(model.short_label)}</a></td>"
            f"<td>{int(health.get('active_units', 0)):,}/{int(health.get('K', 0)):,}</td>"
            f"<td>{_fmt(health.get('train_like_dead_fraction'), percent=True)}</td>"
            f"<td>{_fmt(_route_health(health, 'L', 'train_like_dead_fraction'), percent=True)}</td>"
            f"<td>{_fmt(_route_health(health, 'P', 'train_like_dead_fraction'), percent=True)}</td>"
            f"<td>{_fmt(factor.get('DCI_phone'))}</td><td>{_fmt(factor.get('DCI_speaker_id'))}</td>"
            f"<td>{_fmt(swap.get('P_donor'), percent=True)}</td><td>{_fmt(swap.get('L_recipient'), percent=True)}</td></tr>"
        )
    comparison = (
        "<div class='scroll'><table><thead><tr><th>Model</th><th>Observed units</th><th>All dead</th><th>L dead</th><th>P dead</th>"
        "<th>DCI phone L−P</th><th>DCI speaker P−L</th><th>P donor</th><th>L recipient</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )
    swap_comparison = results / "swap_protocol_comparison_5models_5k" / "report" / "index.html"
    trajectory_comparison = results / "unit_organization_trajectories_5k_shared_sample" / "report" / "index.html"
    report_dir = output / "report"
    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>Final SAE unit-analysis collection</title><style>{_style()}</style></head><body><main>
    <section class='hero'><span class='eyebrow'>Final evidence collection</span><h1>SAE phone–speaker unit analysis</h1>
    <p>One entry point for every final checkpoint. Existing reports are preserved; this collection copies the essential figures/tables and keeps links to full provenance.</p>
    <span class='badge'>Unit and feature evidence complete · voice conversion separate</span></section>
    <h2>How the evidence is computed</h2><section class='panel'><p>The final structural analysis uses 12,000 speaker-balanced utterances: 10,000 train, 1,000 validation, and 1,000 untouched test utterances (7.03M frames). Unit selection and display choices use train+validation only.</p>
    <div class='equation'>Phone association: A⁺(p,u) = max[0, P(u active | p) − P(u active | not p)]</div>
    <div class='equation'>Speaker association: r⁺(s,u) = max[0, point-biserial correlation(speaker s, mean utterance activation u)]</div>
    <div class='equation'>Classifier-free geometry: Δcos = mean[cos(anchor,same label) − cos(anchor,different label)]</div>
    <div class='equation'>Intervention: ĥ(r←dP)=g(zL(r),zP(d)); complementary ĥ(r←dL)=g(zL(d),zP(r))</div>
    <p>Grouped MIG/SAP/DCI compare the complete zL and zP vectors. PCA and UMAP visualize the same held-out observations but are not statistical tests.</p></section>
    <h2>Final models</h2><div class='model-grid'>{cards}</div>
    <h2>Cross-model numerical ledger</h2><p class='muted'>Structural/deadness/DCI values use the 12k final analyses. Intervention columns use the same registered 250 pairs from the 5k Swap-v2 protocol. Dashes mean the measure is structurally undefined.</p>{comparison}
    <h2>Cross-model figures</h2><div class='figure-grid'>
    <figure><a href='../plots/cross_model_route_evidence.png'><img src='../plots/cross_model_route_evidence.png'></a><figcaption><strong>Whole-route final evidence.</strong><br>Desired-route MIG, SAP and DCI contrasts across all routed final checkpoints.</figcaption></figure>
    <figure><a href='../plots/cross_model_health_swap.png'><img src='../plots/cross_model_health_swap.png'></a><figcaption><strong>Health versus function.</strong><br>High unit coverage is not equivalent to correct routing: naive learned routing is the key negative control.</figcaption></figure>
    </div>
    <h2>Complete comparison reports</h2><section class='panel'><ul>
    <li><a href='{_rel_link(swap_comparison, report_dir)}'>Five-model registered Swap-v2 comparison</a></li>
    <li><a href='{_rel_link(trajectory_comparison, report_dir)}'>Five-family organization-trajectory comparison</a></li>
    </ul></section>
    <section class='panel note'><h3>What remains</h3><p>The unit-analysis suite is complete after consolidation. The remaining application-level extension is waveform reconstruction and voice conversion. It should be reported only after the direct vocoder passes original-SPEAR and SAE-reconstruction listening gates.</p></section>
    </main></body></html>"""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "index.html").write_text(page, encoding="utf-8")

    summary_rows = []
    for record in records:
        model, metrics = record["model"], record["metrics"]
        health, factor, swap = metrics["health"], metrics["factor"], metrics["swap"]
        summary_rows.append({
            "model": model.label, "status": model.status,
            "observed_units": health.get("active_units"), "assigned_units": health.get("K"),
            "train_like_dead_fraction": health.get("train_like_dead_fraction"),
            "L_dead_fraction": _route_health(health, "L", "train_like_dead_fraction"),
            "P_dead_fraction": _route_health(health, "P", "train_like_dead_fraction"),
            "MIG_phone_L_minus_P": factor.get("MIG_phone"),
            "MIG_speaker_P_minus_L": factor.get("MIG_speaker_id"),
            "SAP_phone_L_minus_P": factor.get("SAP_phone"),
            "SAP_speaker_P_minus_L": factor.get("SAP_speaker_id"),
            "DCI_phone_L_minus_P": factor.get("DCI_phone"),
            "DCI_speaker_P_minus_L": factor.get("DCI_speaker_id"),
            "P_swap_donor_match": swap.get("P_donor"),
            "L_swap_recipient_match": swap.get("L_recipient"),
        })
    (output / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output / "tables" / "final_model_summary.csv", index=False)


def build(results: Path, output: Path) -> Path:
    results, output = results.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    records = [_model_page(model, results, output) for model in MODELS]
    _cross_model_plots(records, output)
    _root_page(records, results, output)
    manifest = {
        "format": "sae_final_consolidated_collection_v1",
        "models": [record["model"].label for record in records],
        "root_report": str((output / "report" / "index.html").resolve()),
        "voice_conversion_included": False,
    }
    (output / "collection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output / "report" / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate final SAE unit-analysis reports.")
    parser.add_argument("--results", type=Path, default=Path("SAEUnitAnalysis/results"))
    parser.add_argument("--output", type=Path, default=Path("SAEUnitAnalysis/results/final_unit_analysis"))
    args = parser.parse_args()
    report = build(args.results, args.output)
    print(f"[final-reports] report: {report.resolve()}")


if __name__ == "__main__":
    main()
