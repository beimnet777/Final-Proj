from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .bundle import AnalysisBundle
from .utils import AnalysisError, read_structured


_STRESS_RE = re.compile(r"\d+$")
_SILENCE = {"", "<eps>", "<epsilon>", "sil", "sp", "spn", "nsn"}


def _value(line: str) -> str:
    raw = line.split("=", 1)[1].strip() if "=" in line else line.strip()
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        raw = raw[1:-1]
    return raw.replace('""', '"').strip()


def _float_value(line: str) -> float:
    return float(_value(line))


def _normalise_phone(phone: str, *, preserve_stress: bool) -> str:
    phone = str(phone or "").strip()
    if not preserve_stress:
        phone = _STRESS_RE.sub("", phone)
    return phone


def parse_textgrid(path: Path, *, phone_tier_names: tuple[str, ...] = ("phones", "phone")) -> list[dict[str, Any]]:
    """Parse the long-text TextGrid format produced by MFA.

    The parser intentionally supports the common MFA long format directly so the
    conversion step does not depend on praatio/textgrid being installed on the
    cluster.
    """
    tiers: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    interval: dict[str, Any] | None = None

    def finish_interval() -> None:
        nonlocal interval
        if current is not None and interval is not None:
            if {"xmin", "xmax", "text"} <= set(interval):
                current.setdefault("intervals", []).append(interval)
        interval = None

    def finish_tier() -> None:
        nonlocal current
        finish_interval()
        if current is not None:
            tiers.append(current)
        current = None

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("item ["):
            finish_tier()
            current = {"class": "", "name": "", "intervals": []}
        elif current is not None and line.startswith("class ="):
            current["class"] = _value(line)
        elif current is not None and line.startswith("name ="):
            current["name"] = _value(line)
        elif current is not None and line.startswith("intervals ["):
            finish_interval()
            interval = {}
        elif interval is not None and line.startswith("xmin ="):
            interval["xmin"] = _float_value(line)
        elif interval is not None and line.startswith("xmax ="):
            interval["xmax"] = _float_value(line)
        elif interval is not None and line.startswith("text ="):
            interval["text"] = _value(line)

    finish_tier()

    wanted = {x.lower() for x in phone_tier_names}
    for tier in tiers:
        if str(tier.get("class", "")).lower() == "intervaltier" and str(tier.get("name", "")).lower() in wanted:
            return list(tier.get("intervals", []))
    available = [str(t.get("name", "")) for t in tiers]
    raise AnalysisError(f"No phone tier named one of {sorted(wanted)} in {path}; available tiers={available}")


def _load_utterance_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = str(row.get("corpus_stem") or row.get("stem") or row.get("utterance_id") or "").strip()
            utt_id = str(row.get("utterance_id") or stem).strip()
            if stem and utt_id:
                mapping[stem] = utt_id
    return mapping


def _ensure_output_bundle(bundle_root: Path, output: Path) -> None:
    bundle_root = bundle_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output == bundle_root:
        return
    for name in ("utterances.csv", "utterances.parquet", "dataset.yaml"):
        src = bundle_root / name
        if src.exists():
            shutil.copy2(src, output / name)
    audio_src = bundle_root / "audio"
    audio_dst = output / "audio"
    if audio_src.exists() and not audio_dst.exists():
        audio_dst.symlink_to(audio_src.resolve())


def _update_dataset_yaml(bundle_root: Path) -> None:
    path = bundle_root / "dataset.yaml"
    raw = read_structured(path)
    raw["alignments"] = "alignments.csv"
    factors = list(raw.get("factors", []))
    names = {str(f.get("name")) for f in factors if isinstance(f, dict)}
    if "phone" not in names:
        factors.insert(
            0,
            {
                "name": "phone",
                "family": "linguistic",
                "level": "frame",
                "type": "categorical",
                "source": "alignment",
            },
        )
    raw["factors"] = factors
    notes = str(raw.get("notes", "")).strip()
    addition = "Phone alignments imported from MFA TextGrid outputs."
    raw["notes"] = f"{notes} {addition}".strip() if notes else addition
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def import_alignments(
    bundle_root: Path,
    mfa_output: Path,
    *,
    output: Path | None = None,
    utterance_map: Path | None = None,
    phone_tier: str = "phones",
    preserve_stress: bool = False,
    keep_silence: bool = False,
    min_coverage: float = 0.95,
) -> None:
    bundle_root = bundle_root.resolve()
    output_root = (output or bundle_root).resolve()
    _ensure_output_bundle(bundle_root, output_root)

    bundle = AnalysisBundle(output_root)
    known = set(bundle.utterances["utterance_id"].astype(str))
    stem_to_utt = _load_utterance_map(utterance_map)
    rows: list[dict[str, str]] = []
    unknown: list[str] = []
    aligned_utterances: set[str] = set()

    textgrids = sorted(Path(mfa_output).rglob("*.TextGrid")) + sorted(Path(mfa_output).rglob("*.textgrid"))
    if not textgrids:
        raise SystemExit(f"No TextGrid files found under {mfa_output}")

    for tg in textgrids:
        stem = tg.stem
        utt_id = stem_to_utt.get(stem, stem)
        if utt_id not in known:
            unknown.append(stem)
            continue
        intervals = parse_textgrid(tg, phone_tier_names=(phone_tier, "phones", "phone"))
        aligned_utterances.add(utt_id)
        for item in intervals:
            phone = _normalise_phone(str(item["text"]), preserve_stress=preserve_stress)
            if not keep_silence and phone.lower() in _SILENCE:
                continue
            start = float(item["xmin"])
            end = float(item["xmax"])
            if end <= start:
                continue
            rows.append(
                {
                    "utterance_id": utt_id,
                    "start_sec": f"{start:.6f}",
                    "end_sec": f"{end:.6f}",
                    "phone": phone,
                }
            )

    if unknown:
        preview = ", ".join(unknown[:10])
        raise SystemExit(
            f"{len(unknown)} TextGrid files did not map to bundle utterances. "
            f"First unknown stems: {preview}. Pass --utterance-map if MFA renamed files."
        )
    if not rows:
        raise SystemExit("No phone intervals were imported from MFA TextGrids.")
    coverage = len(aligned_utterances) / max(len(known), 1)
    if coverage < min_coverage:
        raise SystemExit(
            f"MFA alignment coverage is too low: {len(aligned_utterances)}/{len(known)} "
            f"utterances ({coverage:.1%}) have TextGrids; required >= {min_coverage:.1%}. "
            "If this was an intentional subset, pass --min-coverage 0."
        )

    with (output_root / "alignments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["utterance_id", "start_sec", "end_sec", "phone"])
        writer.writeheader()
        writer.writerows(rows)

    _update_dataset_yaml(output_root)
    # Re-load for validation, including unknown utterance and interval checks.
    AnalysisBundle(output_root)
    phones = sorted({row["phone"] for row in rows})
    print(f"Wrote MFA-aligned SAE bundle: {output_root}")
    print(f"textgrids={len(textgrids)} aligned_utterances={len(aligned_utterances)}/{len(known)} alignments={len(rows)} phones={len(phones)}")
    print(f"dataset={output_root / 'dataset.yaml'}")
    print(f"alignments={output_root / 'alignments.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MFA TextGrid phone alignments into SAEUnitAnalysis "
            "alignments.csv and attach them to a LibriSpeech analysis bundle."
        )
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--mfa-output", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output bundle. Defaults to updating --bundle in place.",
    )
    parser.add_argument(
        "--utterance-map",
        type=Path,
        default=None,
        help="CSV from prepare_librispeech_mfa_corpus.py. If omitted, TextGrid stem is used as utterance_id.",
    )
    parser.add_argument("--phone-tier", default="phones")
    parser.add_argument(
        "--preserve-stress",
        action="store_true",
        help="Keep ARPABET stress digits such as AH0. Default strips them to AH.",
    )
    parser.add_argument(
        "--keep-silence",
        action="store_true",
        help="Keep MFA silence/noise intervals instead of leaving them unaligned.",
    )
    parser.add_argument(
        "--min-coverage",
        default=0.95,
        type=float,
        help="Minimum fraction of bundle utterances that must have TextGrids. Use 0 for intentional subsets.",
    )
    args = parser.parse_args()
    import_alignments(
        args.bundle,
        args.mfa_output,
        output=args.output,
        utterance_map=args.utterance_map,
        phone_tier=args.phone_tier,
        preserve_stress=args.preserve_stress,
        keep_silence=args.keep_silence,
        min_coverage=args.min_coverage,
    )


if __name__ == "__main__":
    main()
