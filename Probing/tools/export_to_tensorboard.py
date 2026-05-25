#!/usr/bin/env python3
"""Convert all Probing run logs to TensorBoard event files.

Sources used
------------
ASR  — structured train.csv / eval.csv / layer_weights.csv
PR   — SLURM stdout log (per-step loss + per-epoch val PER + layer weights)
SID  — SLURM stdout log (per-step loss + per-epoch val ACC + layer weights)
ER   — SLURM stdout log; 5 folds in a single file, split into fold sub-dirs

Output layout
-------------
    tb_exports/
        asr/lstm/
        asr/weighted_lstm/
        er/final/fold1/ … fold5/ + summary/
        er/weighted/fold1/ … fold5/ + summary/
        pr/final/
        pr/weighted/
        sid/final/
        sid/weighted/

Usage
-----
    python tools/export_to_tensorboard.py [--clean]
    tensorboard --logdir tb_exports/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.tensorboard import SummaryWriter

PROBING_ROOT = Path(__file__).parent.parent
_N_LAYERS    = 13
_LW_TAGS     = [f"layer_weights/layer_{i:02d}" for i in range(_N_LAYERS)]

# ---------------------------------------------------------------- regexps ---

_TRAIN_RE = re.compile(
    r"epoch\s+(\d+)/\d+\s+step\s+(\d+)\s+loss\s+([\d.]+)\s+lr\s+([\S]+)"
)
_VAL_PER_RE  = re.compile(r"\[val\]\s+epoch\s+(\d+)\s+PER\s+([\d.]+)")
_VAL_ACC_RE  = re.compile(r"\[val\]\s+epoch\s+(\d+)\s+acc\s+([\d.]+)")
_TEST_PER_RE = re.compile(r"\[test\]\s+epoch\s+(\d+)\s+PER\s+([\d.]+)")
_TEST_ACC_RE = re.compile(r"\[test\]\s+epoch\s+(\d+)\s+acc\s+([\d.]+)")
_LW_RE       = re.compile(r"layer weights: \[([^\]]+)\]")
_FOLD_RE     = re.compile(r"FOLD\s+(\d+)\s+/")


# --------------------------------------------------------------- data type ---

@dataclass
class FoldData:
    train_steps:  List[Tuple[int, float, float]] = field(default_factory=list)
    val_metrics:  List[Tuple[int, str, float]]   = field(default_factory=list)
    test_metrics: List[Tuple[int, str, float]]   = field(default_factory=list)
    layer_weights: List[Tuple[int, List[float]]] = field(default_factory=list)


def parse_stdout_log(log_path: Path) -> Dict[int, FoldData]:
    """Parse a SLURM stdout log into per-fold FoldData.

    For PR / SID (single run) the result has only fold key 0.
    For ER the result has fold keys 1..5.
    Layer weights are buffered until the matching val line so they share
    the same epoch index.
    """
    folds: Dict[int, FoldData] = {}
    current_fold = 0     # 0 = not yet inside an ER fold / single-run tasks
    pending_lw: Optional[Tuple[int, List[float]]] = None

    with log_path.open() as f:
        for line in f:
            # ---- ER fold boundary
            m = _FOLD_RE.search(line)
            if m:
                current_fold = int(m.group(1))
                pending_lw = None
                continue

            fold = folds.setdefault(current_fold, FoldData())

            # ---- train step
            m = _TRAIN_RE.search(line)
            if m:
                step = int(m.group(2))
                loss = float(m.group(3))
                lr   = float(m.group(4))
                fold.train_steps.append((step, loss, lr))
                continue

            # ---- layer weights (buffer until matching val epoch)
            m = _LW_RE.search(line)
            if m:
                vals = [float(v) for v in m.group(1).split()]
                # epoch index unknown yet — will attach when we see val line
                pending_lw = vals
                continue

            # ---- val metric
            m = _VAL_PER_RE.search(line) or _VAL_ACC_RE.search(line)
            if m:
                epoch  = int(m.group(1))
                tag    = "per" if "PER" in line else "acc"
                value  = float(m.group(2))
                fold.val_metrics.append((epoch, tag, value))
                if pending_lw is not None:
                    fold.layer_weights.append((epoch, pending_lw))
                    pending_lw = None
                continue

            # ---- test metric
            m = _TEST_PER_RE.search(line) or _TEST_ACC_RE.search(line)
            if m:
                epoch  = int(m.group(1))
                tag    = "per" if "PER" in line else "acc"
                value  = float(m.group(2))
                fold.test_metrics.append((epoch, tag, value))

    return folds


# ---------------------------------------------------------- layout helpers ---

def _layout(task: str) -> dict:
    lw = {"Layer Weights": {"Softmax Mix": ["Multiline", _LW_TAGS]}}

    if task == "asr":
        return {
            "Training":   {"CTC Loss": ["Multiline", ["train/loss"]],
                           "LR":       ["Multiline", ["train/lr"]]},
            "Error Rates":{"CER": ["Multiline", ["val/cer", "test/cer"]],
                           "WER": ["Multiline", ["val/wer", "test/wer"]]},
            **lw,
        }
    if task in ("er", "sid"):
        return {
            "Training":  {"CE Loss": ["Multiline", ["train/loss"]],
                          "LR":      ["Multiline", ["train/lr"]]},
            "Accuracy":  {"Val & Test ACC": ["Multiline", ["val/acc", "test/acc"]]},
            **lw,
        }
    if task == "pr":
        return {
            "Training":          {"CTC Loss": ["Multiline", ["train/loss"]],
                                  "LR":       ["Multiline", ["train/lr"]]},
            "Phone Error Rate":  {"PER (lower is better)":
                                  ["Multiline", ["val/per", "test/per"]]},
            **lw,
        }
    return {}


def _writer(out_dir: Path, task: str) -> SummaryWriter:
    out_dir.mkdir(parents=True, exist_ok=True)
    w = SummaryWriter(log_dir=str(out_dir))
    lay = _layout(task)
    if lay:
        w.add_custom_scalars(lay)
    return w


def _write_fold(w: SummaryWriter, fd: FoldData) -> None:
    for step, loss, lr in fd.train_steps:
        w.add_scalar("train/loss", loss, step)
        w.add_scalar("train/lr",   lr,   step)
    for epoch, tag, val in fd.val_metrics:
        w.add_scalar(f"val/{tag}", val, epoch)
    for epoch, tag, val in fd.test_metrics:
        w.add_scalar(f"test/{tag}", val, epoch)
    for epoch, weights in fd.layer_weights:
        for i, wt in enumerate(weights):
            w.add_scalar(f"layer_weights/layer_{i:02d}", float(wt), epoch)


# -------------------------------------------------------------------- ASR ----

def _asr_probe(name: str) -> str:
    parts = name.split("_")
    return "_".join(parts[2:]) if len(parts) > 2 else name


def export_asr_run(run_dir: Path, out_root: Path) -> None:
    probe = _asr_probe(run_dir.name)
    out   = out_root / "asr" / probe
    print(f"  [ASR] {run_dir.name}  →  asr/{probe}")
    w = _writer(out, "asr")

    train_csv = run_dir / "train.csv"
    if train_csv.exists():
        with train_csv.open() as f:
            for row in csv.DictReader(f):
                step = int(row["step"])
                w.add_scalar("train/loss", float(row["loss"]), step)
                w.add_scalar("train/lr",   float(row["lr"]),   step)

    eval_csv = run_dir / "eval.csv"
    if eval_csv.exists():
        with eval_csv.open() as f:
            for row in csv.DictReader(f):
                epoch = int(row["epoch"])
                split = row["split"]
                w.add_scalar(f"{split}/cer", float(row["cer"]), epoch)
                w.add_scalar(f"{split}/wer", float(row["wer"]), epoch)

    lw_csv = run_dir / "layer_weights.csv"
    if lw_csv.exists():
        with lw_csv.open() as f:
            for row in csv.DictReader(f):
                epoch   = int(row["epoch"])
                weights = [float(row[f"layer_{i}"]) for i in range(_N_LAYERS)
                           if f"layer_{i}" in row]
                for i, wt in enumerate(weights):
                    w.add_scalar(f"layer_weights/layer_{i:02d}", wt, epoch)

    w.flush(); w.close()


# ----------------------------------------------------------------- PR / SID --

def _probe_from_log(name: str) -> str:
    """'out.spear_pr_weighted_librispeech100.29473110' → 'weighted'"""
    m = re.search(r"spear_\w+?_(final|weighted(?:_lstm)?|lstm)", name)
    return m.group(1) if m else "unknown"


def export_single_task_log(log_path: Path, task: str, out_root: Path) -> None:
    probe = _probe_from_log(log_path.name)
    out   = out_root / task / probe
    print(f"  [{task.upper()}] {log_path.name}  →  {task}/{probe}")

    folds = parse_stdout_log(log_path)
    # Single-run tasks produce fold key 0 (no FOLD boundary in log)
    fd = folds.get(0, FoldData())

    w = _writer(out, task)
    _write_fold(w, fd)
    w.flush(); w.close()


# -------------------------------------------------------------------- ER -----

def export_er_log(log_path: Path, out_root: Path) -> None:
    probe = _probe_from_log(log_path.name)
    print(f"  [ER]  {log_path.name}  →  er/{probe}/fold1..5 + summary")

    folds = parse_stdout_log(log_path)

    # Per-fold training curves
    for fold_num, fd in sorted(folds.items()):
        if fold_num == 0:
            continue   # shouldn't occur for ER logs
        out = out_root / "er" / probe / f"fold{fold_num}"
        w   = _writer(out, "er")
        _write_fold(w, fd)
        w.flush(); w.close()

    # Summary: one test/acc scalar per fold (x-axis = fold number)
    # and the mean across folds
    out_sum = out_root / "er" / probe / "summary"
    w = _writer(out_sum, "er")
    accs = []
    for fold_num, fd in sorted(folds.items()):
        if fold_num == 0:
            continue
        for _, tag, val in fd.test_metrics:
            if tag == "acc":
                w.add_scalar("test/acc", val, fold_num)
                accs.append(val)
    if accs:
        w.add_scalar("test/mean_acc", sum(accs) / len(accs), 0)
    w.flush(); w.close()


# ---------------------------------------------------------------- ER summary-JSON (legacy) ---

def _probe_from_er_name(stem: str) -> str:
    m = re.search(r"_er_(\w+)_summary$", stem)
    return m.group(1) if m else stem


def export_er_summary_json(summary_path: Path, out_root: Path) -> None:
    """Used only when no stdout log exists — writes final layer weights."""
    with summary_path.open() as f:
        s = json.load(f)
    probe   = _probe_from_er_name(summary_path.stem)
    out_sum = out_root / "er" / probe / "summary"
    print(f"  [ER summary JSON]  →  er/{probe}/summary  (layer weights only)")
    w = _writer(out_sum, "er")
    for fold_str, weights in sorted(
        s.get("layer_weights_per_fold", {}).items(), key=lambda kv: int(kv[0])
    ):
        if weights:
            for i, wt in enumerate(weights):
                w.add_scalar(f"layer_weights/layer_{i:02d}", float(wt), int(fold_str))
    w.flush(); w.close()


# -------------------------------------------------------------------- main ---

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", default=str(PROBING_ROOT / "tb_exports"))
    p.add_argument("--probing_root", default=str(PROBING_ROOT))
    p.add_argument("--clean", action="store_true",
                   help="Wipe out_root before exporting.")
    args = p.parse_args()

    out_root     = Path(args.out_root)
    probing_root = Path(args.probing_root)

    if args.clean and out_root.exists():
        print(f"Removing {out_root}")
        shutil.rmtree(out_root)

    print(f"Exporting → {out_root}\n")

    # ---- ASR (structured CSVs)
    asr_runs = probing_root / "asr" / "runs"
    if asr_runs.exists():
        print("ASR:")
        for d in sorted(asr_runs.iterdir()):
            if d.is_dir() and ((d / "train.csv").exists() or (d / "eval.csv").exists()):
                export_asr_run(d, out_root)

    # ---- PR stdout logs
    pr_logs = probing_root / "pr" / "logs"
    if pr_logs.exists():
        print("\nPR:")
        for f in sorted(pr_logs.glob("out.spear_pr_*")):
            export_single_task_log(f, "pr", out_root)

    # ---- SID stdout logs
    sid_logs = probing_root / "sid" / "logs"
    if sid_logs.exists():
        print("\nSID:")
        for f in sorted(sid_logs.glob("out.spear_sid_*")):
            export_single_task_log(f, "sid", out_root)

    # ---- ER stdout logs (multi-fold, one file per probe)
    er_logs = probing_root / "er" / "logs"
    if er_logs.exists():
        print("\nER:")
        for f in sorted(er_logs.glob("out.spear_er_*")):
            export_er_log(f, out_root)

    print(f"\nDone.")
    print(f"\nLaunch with grouped sidebar (recommended):")
    print(f"  bash tools/launch_tensorboard.sh")
    print(f"\nOr manually:")
    print(
        f"  tensorboard --logdir_spec "
        f"ASR:{out_root}/asr,"
        f"ER:{out_root}/er,"
        f"PR:{out_root}/pr,"
        f"SID:{out_root}/sid"
    )


if __name__ == "__main__":
    main()
