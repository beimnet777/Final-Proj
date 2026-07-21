from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import AnalysisError


DEFAULT_RESULTS = (
    ("Fixed routing", "libri_fixed_240L16P_step12000_final_swapv2_5k_mps"),
    ("Naive learned", "libri_learned_naive_step12000_final_swapv2_5k_mps"),
    ("Quota-freeze", "libri_learned_qfreeze4k_step20000_final_swapv2_5k_mps"),
    ("Learned post-GP", "libri_learned_qfreeze4k_postgp030_step20000_final_swapv2_5k_mps"),
    ("Learned ramp-5k", "libri_learned_qfreeze4k_ramp5k_step20000_final_swapv2_5k_mps"),
)

SIGNATURES = (
    ("P phone retention", "P_from_donor", "phone_retention"),
    ("P donor identity", "P_from_donor", "donor_speaker_match"),
    ("L phone replacement", "L_from_donor", "phone_replacement"),
    ("L recipient identity", "L_from_donor", "recipient_speaker_match"),
)


def _parse_results(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        root = Path("SAEUnitAnalysis/results")
        return [(label, root / name) for label, name in DEFAULT_RESULTS]
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise AnalysisError("Each --result must be LABEL=PATH.")
        label, raw_path = value.split("=", 1)
        parsed.append((label.strip(), Path(raw_path).expanduser()))
    return parsed


def _load(results: list[tuple[str, Path]]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    registry: pd.DataFrame | None = None
    metadata_rows = []
    pair_columns = ["pair", "recipient", "donor", "same_speaker_donor"]
    for label, result in results:
        swaps_path = result / "tables" / "swaps.csv"
        pairs_path = result / "tables" / "swap_pairs.csv"
        summary_path = result / "swap.json"
        if not swaps_path.exists() or not pairs_path.exists():
            raise AnalysisError(f"Missing upgraded swap outputs in {result}.")
        swaps = pd.read_csv(swaps_path)
        pairs = pd.read_csv(pairs_path)
        current_registry = pairs[pair_columns].sort_values("pair").reset_index(drop=True)
        if registry is None:
            registry = current_registry
        elif not current_registry.equals(registry):
            raise AnalysisError(
                f"{result} does not use the same recipient/donor pair registry."
            )
        required_modes = {
            "baseline", "P_from_donor", "L_from_donor",
            "matched_P_subset_from_donor", "matched_nonP_from_donor",
        }
        missing_modes = required_modes - set(swaps["mode"].astype(str))
        if missing_modes:
            raise AnalysisError(
                f"{result} is missing corrected swap modes: {sorted(missing_modes)}"
            )
        tables[label] = swaps
        summary = json.loads(summary_path.read_text())
        match = summary.get("matched_non_p_control", {})
        metadata_rows.append({
            "checkpoint": label,
            "result": str(result.resolve()),
            "pairs": int(len(pairs)),
            "recipient_speakers": int(pairs["recipient_speaker"].nunique()),
            "donor_speakers": int(pairs["donor_speaker"].nunique()),
            **{f"matching_{key}": value for key, value in match.items()},
        })
    return tables, pd.DataFrame(metadata_rows)


def _paired_wide(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "phone_recipient_accuracy", "donor_speaker_match",
        "recipient_speaker_match", "donor_speaker_probability",
        "recipient_speaker_probability",
    ]
    index = ["pair", "recipient_speaker", "donor_speaker"]
    selected = table[table["mode"].isin({
        "baseline", "P_from_donor", "L_from_donor",
        "matched_P_subset_from_donor", "matched_nonP_from_donor",
    })]
    return selected.pivot(index=index, columns="mode", values=columns).sort_index()


def _two_way_weights(
    recipients: np.ndarray,
    donors: np.ndarray,
    *,
    rng: np.random.Generator,
    repetitions: int,
) -> np.ndarray:
    recipient_levels = np.unique(recipients)
    donor_levels = np.unique(donors)
    weights = np.empty((int(repetitions), len(recipients)), dtype=np.int16)
    for repetition in range(int(repetitions)):
        sampled_recipients = rng.choice(
            recipient_levels, size=len(recipient_levels), replace=True,
        )
        sampled_donors = rng.choice(donor_levels, size=len(donor_levels), replace=True)
        recipient_counts = {
            value: int(np.sum(sampled_recipients == value)) for value in recipient_levels
        }
        donor_counts = {
            value: int(np.sum(sampled_donors == value)) for value in donor_levels
        }
        weights[repetition] = np.asarray([
            recipient_counts[recipient] * donor_counts[donor]
            for recipient, donor in zip(recipients, donors)
        ], dtype=np.int16)
    return weights


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = weights.sum(axis=1)
    numerator = weights @ values
    fallback = float(np.mean(values))
    return np.divide(
        numerator, denominator, out=np.full(len(weights), fallback, dtype=float),
        where=denominator > 0,
    )


def signature_table(
    tables: dict[str, pd.DataFrame], *, seed: int = 42, repetitions: int = 1000,
) -> pd.DataFrame:
    rows = []
    for checkpoint_index, (label, table) in enumerate(tables.items()):
        wide = _paired_wide(table)
        recipients = wide.index.get_level_values("recipient_speaker").astype(str).to_numpy()
        donors = wide.index.get_level_values("donor_speaker").astype(str).to_numpy()
        weights = _two_way_weights(
            recipients, donors,
            rng=np.random.default_rng(seed + checkpoint_index * 1009),
            repetitions=repetitions,
        )
        baseline_phone = wide[("phone_recipient_accuracy", "baseline")].to_numpy(float)
        baseline_draws = _weighted_mean(baseline_phone, weights)
        for display, mode, metric in SIGNATURES:
            if metric == "phone_retention":
                values = wide[("phone_recipient_accuracy", mode)].to_numpy(float)
                estimate = float(values.mean() / max(baseline_phone.mean(), 1e-12))
                draws = _weighted_mean(values, weights) / np.maximum(baseline_draws, 1e-12)
            elif metric == "phone_replacement":
                values = wide[("phone_recipient_accuracy", mode)].to_numpy(float)
                estimate = float(1.0 - values.mean() / max(baseline_phone.mean(), 1e-12))
                draws = 1.0 - _weighted_mean(values, weights) / np.maximum(baseline_draws, 1e-12)
            else:
                values = wide[(metric, mode)].to_numpy(float)
                estimate = float(values.mean())
                draws = _weighted_mean(values, weights)
            low, high = np.quantile(draws, [.025, .975])
            rows.append({
                "checkpoint": label,
                "criterion": display,
                "mode": mode,
                "metric": metric,
                "value": estimate,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "pairs": int(len(wide)),
                "interval_method": "two_way_recipient_donor_speaker_bootstrap",
            })
    return pd.DataFrame(rows)


def control_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, table in tables.items():
        contrast_path = None
        # Recalculate means directly so this utility is independent of report formatting.
        wide = _paired_wide(table)
        for mode in ("matched_P_subset_from_donor", "matched_nonP_from_donor"):
            rows.append({
                "checkpoint": label,
                "mode": mode,
                "phone_accuracy": float(
                    wide[("phone_recipient_accuracy", mode)].to_numpy(float).mean()
                ),
                "donor_speaker_match": float(
                    wide[("donor_speaker_match", mode)].to_numpy(float).mean()
                ),
                "donor_speaker_probability": float(
                    wide[("donor_speaker_probability", mode)].to_numpy(float).mean()
                ),
            })
    return pd.DataFrame(rows)


def make_comparison(
    results: list[tuple[str, Path]], output: Path, *, seed: int = 42,
) -> Path:
    import matplotlib.pyplot as plt

    tables, metadata = _load(results)
    signatures = signature_table(tables, seed=seed)
    controls = control_table(tables)
    output = output.resolve()
    plots = output / "plots"
    report = output / "report"
    output_tables = output / "tables"
    for directory in (plots, report, output_tables):
        directory.mkdir(parents=True, exist_ok=True)
    signatures.to_csv(output_tables / "swap_double_dissociation.csv", index=False)
    controls.to_csv(output_tables / "swap_matched_controls.csv", index=False)
    metadata.to_csv(output_tables / "swap_protocol_registry.csv", index=False)

    colors = ("#0b7285", "#7b2cbf", "#e8590c", "#2b8a3e", "#364fc7")
    checkpoints = list(tables)
    criteria = [item[0] for item in SIGNATURES]
    x = np.arange(len(criteria), dtype=float)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        group = signatures[signatures["checkpoint"] == checkpoint].set_index("criterion").loc[criteria]
        values = group["value"].to_numpy(float)
        low = group["ci95_low"].to_numpy(float)
        high = group["ci95_high"].to_numpy(float)
        ax.plot(x, values, marker="o", markersize=7, linewidth=2.3,
                color=colors[checkpoint_index], label=checkpoint)
        ax.fill_between(x, low, high, color=colors[checkpoint_index], alpha=.10)
    ax.axhline(1, color="#495057", linestyle="--", linewidth=1, alpha=.7)
    ax.set(
        xticks=x, xticklabels=criteria, ylabel="specificity score (higher is better)",
        ylim=(0, 1.08), title="Cross-checkpoint latent-swap double dissociation",
    )
    ax.grid(axis="y", color="#dee2e6", linewidth=.8)
    ax.legend(frameon=False, ncol=len(checkpoints), loc="lower center")
    fig.tight_layout()
    fig.savefig(plots / "swap_double_dissociation.png", dpi=220, bbox_inches="tight")
    fig.savefig(plots / "swap_double_dissociation.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        group = controls[controls["checkpoint"] == checkpoint].set_index("mode")
        non_p = group.loc["matched_nonP_from_donor"]
        p = group.loc["matched_P_subset_from_donor"]
        color = colors[checkpoint_index]
        ax.annotate(
            "", xy=(p.phone_accuracy, p.donor_speaker_probability),
            xytext=(non_p.phone_accuracy, non_p.donor_speaker_probability),
            arrowprops={"arrowstyle": "->", "lw": 2.2, "color": color},
        )
        ax.scatter(
            [non_p.phone_accuracy, p.phone_accuracy],
            [non_p.donor_speaker_probability, p.donor_speaker_probability],
            s=[60, 95], color=color, edgecolor="white", linewidth=1.2,
            label=checkpoint,
        )
        ax.text(p.phone_accuracy, p.donor_speaker_probability, "  P subset",
                color=color, va="center", fontsize=8)
    ax.set(
        xlabel="recipient-phone accuracy", ylabel="donor-speaker probability",
        title="Matched unit-count control: non-P to P-subset swap",
    )
    ax.grid(color="#e9ecef", linewidth=.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots / "swap_matched_control_arrows.png", dpi=220, bbox_inches="tight")
    fig.savefig(plots / "swap_matched_control_arrows.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in (
            "checkpoint", "pairs", "recipient_speakers", "donor_speakers",
            "matching_full_p_units", "matching_full_non_p_units",
            "matching_matched_units", "matching_target_p_fraction",
        )) + "</tr>"
        for _, row in metadata.iterrows()
    )
    page = f"""<!doctype html><html><head><meta charset='utf-8'><style>
body{{font-family:Inter,system-ui,sans-serif;background:#f5f7fb;color:#172033;margin:0;padding:34px}}
.panel{{background:white;border-radius:14px;padding:22px;margin:20px 0;box-shadow:0 3px 14px #1d2b4b18}}
img{{max-width:100%;height:auto}} table{{border-collapse:collapse;width:100%}} td,th{{padding:8px;border-bottom:1px solid #e9ecef;text-align:left}}
</style><title>Latent-swap comparison</title></head><body>
<h1>Latent-swap comparison</h1>
<p>All {len(checkpoints)} checkpoints use the exact same 250 recipient–donor pairs (40 recipient and 40 donor speakers). Intervals use crossed recipient/donor-speaker bootstrap resampling.</p>
<div class='panel'><h2>Double-dissociation signature</h2><img src='../plots/swap_double_dissociation.png'><p>P swapping should retain recipient phones while adopting donor identity; complementary L swapping should replace phone information while retaining recipient identity.</p></div>
<div class='panel'><h2>Capacity-matched control</h2><img src='../plots/swap_matched_control_arrows.png'><p>Each arrow compares an activity- and decoder-norm-matched, equal-count non-P subset with its paired P subset. This avoids treating the full L route as a matched control when learned P is larger.</p></div>
<div class='panel'><h2>Registry and matched subsets</h2><table><thead><tr><th>checkpoint</th><th>pairs</th><th>recipient speakers</th><th>donor speakers</th><th>all P units</th><th>all non-P units</th><th>matched units/side</th><th>P fraction tested</th></tr></thead><tbody>{rows}</tbody></table></div>
</body></html>"""
    page_path = report / "index.html"
    page_path.write_text(page, encoding="utf-8")
    return page_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare corrected latent-swap protocols on one registered pair set."
    )
    parser.add_argument(
        "--result", action="append", default=[], metavar="LABEL=PATH",
        help="Repeat for each result; defaults to fixed, post-GP and ramp-5k outputs.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("SAEUnitAnalysis/results/swap_protocol_comparison_5models_5k"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    page = make_comparison(_parse_results(args.result), args.output, seed=args.seed)
    print(page)


if __name__ == "__main__":
    main()
