"""Emotion Recognition — CLI entry point.

Runs the full 5-fold IEMOCAP cross-validation and prints per-fold + mean ACC.

Usage
-----
    # From the er/ directory:
    python er_run.py --probe final    --iemocap_root /path/to/IEMOCAP_full_release
    python er_run.py --probe weighted --iemocap_root /path/to/IEMOCAP_full_release
    python er_run.py --probe fixed_weighted --iemocap_root /path/to/IEMOCAP_full_release

    # Run only specific folds:
    python er_run.py --probe weighted --iemocap_root /path/... --folds 1 2

    # Use a different encoder:
    python er_run.py --probe weighted --iemocap_root /path/...
                     --model_id facebook/wav2vec2-large-960h --model_family hf
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))
from er_config import ERConfig, EMOTION_NAMES
from er_data import make_er_dataloaders
from er_model import build_er_model
from er_train import fit_er, evaluate_er
from reproducibility import set_seed
from tb_logger import TBLogger


# --------------------------------------------------------------- CLI ----


def parse_args():
    cfg = ERConfig()
    p = argparse.ArgumentParser(
        description="IEMOCAP emotion recognition — 5-fold cross-validation."
    )
    p.add_argument(
        "--probe", choices=["final", "weighted", "fixed_weighted"], default=cfg.probe_type,
        help="Probe head: 'final' (single layer + mean pool + linear) or "
             "'weighted' (softmax layer mix + mean pool + linear) or "
             "'fixed_weighted' (uniform layer average + mean pool + linear).",
    )
    p.add_argument(
        "--iemocap_root", required=True,
        help="Path to the IEMOCAP_full_release directory.",
    )
    p.add_argument("--model_id",     default=cfg.model_id)
    p.add_argument("--model_family", default=cfg.model_family, choices=["spear", "hf"])
    p.add_argument("--epochs",       type=int,   default=cfg.num_epochs)
    p.add_argument("--batch_size",   type=int,   default=cfg.batch_size)
    p.add_argument("--eval_batch_size", type=int, default=cfg.eval_batch_size)
    p.add_argument("--lr",           type=float, default=cfg.learning_rate)
    p.add_argument("--warmup_steps", type=int,   default=cfg.warmup_steps)
    p.add_argument("--layer_idx",    type=int,   default=cfg.layer_idx,
                   help="For --probe final: which encoder layer to use (0-based, -1=last).")
    p.add_argument("--runs_dir",       default=None,
                   help="Directory for run summaries. Default: er/runs/")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Directory for checkpoints. Default: er/checkpoints/")
    p.add_argument("--log_dir",        default=None,
                   help="Directory for log files. Default: er/logs/")
    p.add_argument("--folds",   nargs="+", type=int, default=list(range(1, 6)),
                   help="Which folds to run (default: all 5).")
    p.add_argument("--seed", type=int, default=cfg.seed)
    args = p.parse_args()

    cfg.probe_type      = args.probe
    cfg.iemocap_root    = Path(args.iemocap_root)
    cfg.model_id        = args.model_id
    cfg.model_family    = args.model_family
    cfg.num_epochs      = args.epochs
    cfg.batch_size      = args.batch_size
    cfg.eval_batch_size = args.eval_batch_size
    cfg.learning_rate   = args.lr
    cfg.warmup_steps    = args.warmup_steps
    cfg.layer_idx       = args.layer_idx
    cfg.runs_dir        = Path(args.runs_dir)       if args.runs_dir       else cfg.runs_dir
    cfg.checkpoint_dir  = Path(args.checkpoint_dir) if args.checkpoint_dir else cfg.checkpoint_dir
    cfg.log_dir         = Path(args.log_dir)         if args.log_dir         else cfg.log_dir
    cfg.seed            = args.seed
    return cfg, args.folds


# ---------------------------------------------------------------- Main ---


def main() -> None:
    cfg, folds = parse_args()
    set_seed(cfg.seed)

    # Ensure all output directories exist up front.
    for d in (cfg.runs_dir, cfg.checkpoint_dir, cfg.log_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== probe_type   : {cfg.probe_type}")
    print(f"=== model_id     : {cfg.model_id}")
    print(f"=== model_family : {cfg.model_family}")
    print(f"=== iemocap_root : {cfg.iemocap_root}")
    print(f"=== checkpoint   : {cfg.checkpoint_dir}")
    print(f"=== runs         : {cfg.runs_dir}")
    print(f"=== logs         : {cfg.log_dir}")
    print(f"=== folds        : {folds}")
    print(f"=== emotions     : {EMOTION_NAMES}")
    print(f"=== epochs       : {cfg.num_epochs}")
    print(f"=== lr           : {cfg.learning_rate}")
    print(f"=== seed         : {cfg.seed}")

    fold_results: dict[int, float] = {}
    fold_layer_weights: dict[int, list] = {}
    base_ckpt_dir = cfg.checkpoint_dir

    for fold in folds:
        print(f"\n{'=' * 62}")
        print(f"  FOLD {fold} / 5   (test session = Session{fold})")
        print(f"{'=' * 62}")

        cfg.test_fold      = fold
        cfg.checkpoint_dir = base_ckpt_dir / f"fold{fold}"
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed)

        tb = TBLogger(cfg.runs_dir / "tb", run_name=f"fold{fold}", task="er")

        encoder, probe = build_er_model(cfg)
        train_dl, val_dl, test_dl = make_er_dataloaders(cfg)

        best_val_acc = fit_er(cfg, encoder, probe, train_dl, val_dl, tb=tb)
        print(f"\n[fold {fold}] best val acc : {best_val_acc:.4f}")

        test_metrics = evaluate_er(
            cfg, encoder, probe, test_dl, label="test", epoch=cfg.num_epochs, tb=tb
        )
        fold_results[fold] = test_metrics["acc"]

        if hasattr(probe, "layer_weights"):
            fold_layer_weights[fold] = probe.layer_weights.tolist()

        print(f"[fold {fold}] test acc     : {test_metrics['acc']:.4f}")
        tb.close()

    # ------------------------------------------------- Summary printout
    print(f"\n{'=' * 62}")
    print("  CROSS-VALIDATION RESULTS")
    print(f"{'=' * 62}")
    for fold in sorted(fold_results):
        print(f"  fold {fold} : {fold_results[fold]:.4f}")
    mean_acc = sum(fold_results.values()) / len(fold_results)
    print(f"  ─────────────────────")
    print(f"  mean ACC  : {mean_acc:.4f}")
    print(f"{'=' * 62}")

    # ------------------------------------------------- Save summary JSON
    runs_dir = Path(cfg.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = runs_dir / f"{ts}_er_{cfg.probe_type}_summary.json"
    summary = {
        "probe_type":   cfg.probe_type,
        "model_id":     cfg.model_id,
        "model_family": cfg.model_family,
        "seed":         cfg.seed,
        "deterministic": True,
        "num_workers":  cfg.num_workers,
        "superb_split_seed": 0,
        "folds_run":    folds,
        "fold_test_acc": fold_results,
        "mean_acc":     mean_acc,
        "layer_weights_per_fold": fold_layer_weights,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[run done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
