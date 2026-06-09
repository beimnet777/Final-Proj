"""Phone Recognition — CLI entry point.

Runs a single train → val → test pass on LibriSpeech 100h.

Usage
-----
    # From the pr/ directory:
    python pr_run.py --probe final
    python pr_run.py --probe weighted
    python pr_run.py --probe fixed_weighted

    # Different encoder:
    python pr_run.py --probe weighted \\
                     --model_id facebook/wav2vec2-large-960h --model_family hf
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
from pr_config import PRConfig
from pr_data   import make_pr_dataloaders
from pr_model  import build_pr_model
from pr_train  import fit_pr, evaluate_pr
from tb_logger import TBLogger


# ---------------------------------------------------------------- Seed ---


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------- CLI ----


def parse_args() -> PRConfig:
    cfg = PRConfig()
    p = argparse.ArgumentParser(
        description="LibriSpeech 100h phone recognition probing."
    )
    p.add_argument(
        "--probe", choices=["final", "weighted", "fixed_weighted"], default=cfg.probe_type,
        help="Probe head type.",
    )
    p.add_argument("--model_id",        default=cfg.model_id)
    p.add_argument("--model_family",    default=cfg.model_family,
                   choices=["spear", "hf"])
    p.add_argument("--epochs",          type=int,   default=cfg.num_epochs)
    p.add_argument("--batch_size",      type=int,   default=cfg.batch_size)
    p.add_argument("--eval_batch_size", type=int,   default=cfg.eval_batch_size)
    p.add_argument("--lr",              type=float, default=cfg.learning_rate)
    p.add_argument("--warmup_steps",    type=int,   default=cfg.warmup_steps)
    p.add_argument("--layer_idx",       type=int,   default=cfg.layer_idx)
    p.add_argument("--data_cache_dir",  default=str(cfg.data_cache_dir))
    p.add_argument("--lexicon_path",    default=str(cfg.librispeech_lexicon),
                   help="Path to librispeech-lexicon.txt from openslr.org/11.")
    p.add_argument("--max_examples",    type=int, default=cfg.max_examples,
                   help="Cap each split to this many examples (0 = no cap; for smoke tests).")
    p.add_argument("--runs_dir",        default=None)
    p.add_argument("--checkpoint_dir",  default=None)
    p.add_argument("--log_dir",         default=None)
    p.add_argument("--seed",            type=int,   default=cfg.seed)
    args = p.parse_args()

    cfg.probe_type      = args.probe
    cfg.model_id        = args.model_id
    cfg.model_family    = args.model_family
    cfg.num_epochs      = args.epochs
    cfg.batch_size      = args.batch_size
    cfg.eval_batch_size = args.eval_batch_size
    cfg.learning_rate   = args.lr
    cfg.warmup_steps    = args.warmup_steps
    cfg.layer_idx       = args.layer_idx
    cfg.data_cache_dir  = Path(args.data_cache_dir)
    cfg.librispeech_lexicon = Path(args.lexicon_path)
    cfg.max_examples        = args.max_examples
    cfg.runs_dir        = Path(args.runs_dir)       if args.runs_dir       else cfg.runs_dir
    cfg.checkpoint_dir  = Path(args.checkpoint_dir) if args.checkpoint_dir else cfg.checkpoint_dir
    cfg.log_dir         = Path(args.log_dir)         if args.log_dir        else cfg.log_dir
    cfg.seed            = args.seed
    return cfg


# ---------------------------------------------------------------- Main ---


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    for d in (cfg.runs_dir, cfg.checkpoint_dir, cfg.log_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== probe_type   : {cfg.probe_type}")
    print(f"=== model_id     : {cfg.model_id}")
    print(f"=== model_family : {cfg.model_family}")
    print(f"=== checkpoint   : {cfg.checkpoint_dir}")
    print(f"=== runs         : {cfg.runs_dir}")
    print(f"=== epochs       : {cfg.num_epochs}  lr={cfg.learning_rate}")

    tokenizer, train_dl, val_dl, test_dl = make_pr_dataloaders(cfg)
    encoder, probe = build_pr_model(cfg)

    from datetime import datetime as _dt
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    tb = TBLogger(cfg.runs_dir / "tb", run_name=f"{ts}_{cfg.probe_type}", task="pr")

    best_val_per = fit_pr(cfg, encoder, probe, tokenizer, train_dl, val_dl, tb=tb)
    print(f"\n[PR] best val PER : {best_val_per:.4f}")

    print("\n=== Final test-set evaluation ===")
    test_metrics = evaluate_pr(cfg, encoder, probe, tokenizer, test_dl,
                               label="test", epoch=cfg.num_epochs, tb=tb)
    print(f"[PR] test PER     : {test_metrics['per']:.4f}")

    # Layer weights (learned or fixed weighted probes).
    layer_weights = None
    if hasattr(probe, "layer_weights"):
        layer_weights = probe.layer_weights.tolist()
        label = "Fixed uniform weights" if cfg.probe_type == "fixed_weighted" else "Learned softmax weights"
        print(f"\n{label} over encoder layers:")
        for i, w in enumerate(layer_weights):
            print(f"  layer {i:>2d}: {w:.4f}")

    # Save summary JSON.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = cfg.runs_dir / f"{ts}_pr_{cfg.probe_type}_summary.json"
    summary = {
        "probe_type":    cfg.probe_type,
        "model_id":      cfg.model_id,
        "model_family":  cfg.model_family,
        "vocab_size":    cfg.vocab_size,
        "best_val_per":  best_val_per,
        "test_per":      test_metrics["per"],
        "layer_weights": layer_weights,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    tb.close()
    print(f"\n[run done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
