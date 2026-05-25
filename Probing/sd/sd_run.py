"""Spoof Detection — CLI entry point.

Trains on ASVspoof 2019 LA train, validates on ASV19 LA dev each epoch,
then evaluates on all configured test sets ONCE after training completes.

Usage
-----
    python sd_run.py --probe weighted --asv19_la_root /path/to/ASVspoof2019/LA

    # With cross-dataset test sets:
    python sd_run.py --probe weighted \\
        --asv19_la_root /data/ASVspoof2019/LA \\
        --asv21_la_root /data/ASVspoof2021_LA \\
        --asv21_la_keys /data/ASVspoof2021_keys/keys_2021_LA_eval.txt \\
        --asv21_df_root /data/ASVspoof2021_DF \\
        --asv21_df_keys /data/ASVspoof2021_keys/keys_2021_DF_eval.txt \\
        --itw_root      /data/ITW
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from sd_config import SDConfig
from sd_data   import make_sd_train_dataloaders, make_test_dataloaders
from sd_model  import build_sd_model
from sd_train  import fit_sd, evaluate_sd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> SDConfig:
    cfg = SDConfig()
    p = argparse.ArgumentParser(
        description="Spoof Detection probing — ASVspoof 2019 LA training, "
                    "multi-dataset EER evaluation."
    )
    # ── core ────────────────────────────────────────────────────────────────
    p.add_argument("--probe",         choices=["final", "weighted"],
                   default=cfg.probe_type)
    p.add_argument("--asv19_la_root", required=True,
                   help="ASVspoof 2019 LA root (contains LA_cm_protocols/ and flac/).")
    p.add_argument("--model_id",      default=cfg.model_id)
    p.add_argument("--model_family",  default=cfg.model_family,
                   choices=["spear", "hf"])
    # ── optional test sets ──────────────────────────────────────────────────
    p.add_argument("--asv21_la_root", default=None)
    p.add_argument("--asv21_la_keys", default=None)
    p.add_argument("--asv21_df_root", default=None)
    p.add_argument("--asv21_df_keys", default=None)
    p.add_argument("--itw_root",      default=None)
    p.add_argument("--dfeval24_root", default=None)
    p.add_argument("--dfeval24_keys", default=None)
    p.add_argument("--famous_figures_root", default=None)
    p.add_argument("--famous_figures_keys", default=None)
    p.add_argument("--asvspoofld_root", default=None)
    p.add_argument("--asvspoofld_keys", default=None)
    # ── training ────────────────────────────────────────────────────────────
    p.add_argument("--epochs",          type=int,   default=cfg.num_epochs)
    p.add_argument("--batch_size",      type=int,   default=cfg.batch_size)
    p.add_argument("--eval_batch_size", type=int,   default=cfg.eval_batch_size)
    p.add_argument("--lr",              type=float, default=cfg.learning_rate)
    p.add_argument("--warmup_steps",    type=int,   default=cfg.warmup_steps)
    p.add_argument("--proj_dim",        type=int,   default=cfg.proj_dim)
    p.add_argument("--mlp_hidden",      type=int,   default=cfg.mlp_hidden)
    p.add_argument("--layer_idx",       type=int,   default=cfg.layer_idx)
    p.add_argument("--runs_dir",        default=None)
    p.add_argument("--checkpoint_dir",  default=None)
    p.add_argument("--log_dir",         default=None)
    p.add_argument("--seed",            type=int,   default=cfg.seed)
    p.add_argument("--num_workers",     type=int,   default=cfg.num_workers,
                   help="DataLoader worker processes. Set 0 to avoid /dev/shm issues on HPC.")

    args = p.parse_args()
    cfg.probe_type     = args.probe
    cfg.asv19_la_root  = Path(args.asv19_la_root)
    cfg.model_id       = args.model_id
    cfg.model_family   = args.model_family
    cfg.num_epochs     = args.epochs
    cfg.batch_size     = args.batch_size
    cfg.eval_batch_size = args.eval_batch_size
    cfg.learning_rate  = args.lr
    cfg.warmup_steps   = args.warmup_steps
    cfg.proj_dim       = args.proj_dim
    cfg.mlp_hidden     = args.mlp_hidden
    cfg.layer_idx      = args.layer_idx
    cfg.seed           = args.seed
    cfg.num_workers    = args.num_workers
    cfg.runs_dir       = Path(args.runs_dir)       if args.runs_dir       else cfg.runs_dir
    cfg.checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else cfg.checkpoint_dir
    cfg.log_dir        = Path(args.log_dir)         if args.log_dir        else cfg.log_dir

    def _opt(v): return Path(v) if v else None
    cfg.asv21_la_root        = _opt(args.asv21_la_root)
    cfg.asv21_la_keys        = _opt(args.asv21_la_keys)
    cfg.asv21_df_root        = _opt(args.asv21_df_root)
    cfg.asv21_df_keys        = _opt(args.asv21_df_keys)
    cfg.itw_root             = _opt(args.itw_root)
    cfg.dfeval24_root        = _opt(args.dfeval24_root)
    cfg.dfeval24_keys        = _opt(args.dfeval24_keys)
    cfg.famous_figures_root  = _opt(args.famous_figures_root)
    cfg.famous_figures_keys  = _opt(args.famous_figures_keys)
    cfg.asvspoofld_root      = _opt(args.asvspoofld_root)
    cfg.asvspoofld_keys      = _opt(args.asvspoofld_keys)
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    for d in (cfg.runs_dir, cfg.checkpoint_dir, cfg.log_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== probe_type   : {cfg.probe_type}")
    print(f"=== model_id     : {cfg.model_id}")
    print(f"=== model_family : {cfg.model_family}")
    print(f"=== asv19_la_root: {cfg.asv19_la_root}")
    print(f"=== epochs       : {cfg.num_epochs}  lr={cfg.learning_rate}")

    # ── Data ────────────────────────────────────────────────────────────────
    train_dl, val_dl = make_sd_train_dataloaders(cfg)
    test_dls         = make_test_dataloaders(cfg)

    # ── Model ───────────────────────────────────────────────────────────────
    encoder, probe = build_sd_model(cfg)

    # ── Train (val EER logged each epoch) ───────────────────────────────────
    best_val_eer = fit_sd(cfg, encoder, probe, train_dl, val_dl)
    print(f"\n[SD] best val EER : {best_val_eer:.4f}")

    # ── Test evaluation (ONE pass per dataset after training) ───────────────
    print("\n=== Test-set evaluation ===")
    test_results: dict = {}
    for ds_name, dl in test_dls.items():
        metrics = evaluate_sd(cfg, encoder, probe, dl,
                              label=ds_name, epoch=cfg.num_epochs)
        test_results[ds_name] = metrics["eer"]
        print(f"  {ds_name:<20s}  EER = {metrics['eer']:.4f}")

    # ── Layer weights ────────────────────────────────────────────────────────
    layer_weights = None
    if cfg.probe_type == "weighted" and hasattr(probe, "layer_weights"):
        layer_weights = probe.layer_weights.tolist()
        print("\nLearned softmax weights over encoder layers:")
        for i, w in enumerate(layer_weights):
            print(f"  layer {i:>2d}: {w:.4f}")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = cfg.runs_dir / f"{ts}_sd_{cfg.probe_type}_summary.json"
    summary = {
        "probe_type":    cfg.probe_type,
        "model_id":      cfg.model_id,
        "model_family":  cfg.model_family,
        "best_val_eer":  best_val_eer,
        "test_eer":      test_results,
        "layer_weights": layer_weights,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[run done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
