"""Speaker Identification — CLI entry point.

Runs a single train → val → test pass on VoxCeleb1.

Usage
-----
    # From the sid/ directory:
    python sid_run.py --probe final    --voxceleb1_root /path/to/VoxCeleb1
    python sid_run.py --probe weighted --voxceleb1_root /path/to/VoxCeleb1
    python sid_run.py --probe fixed_weighted --voxceleb1_root /path/to/VoxCeleb1

    # Different encoder:
    python sid_run.py --probe weighted --voxceleb1_root /path/...
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
from sid_config import SIDConfig
from sid_data   import make_sid_dataloaders
from sid_model  import build_sid_model
from sid_train  import fit_sid, evaluate_sid
from reproducibility import set_seed
from tb_logger  import TBLogger


# --------------------------------------------------------------- CLI ----


def parse_args():
    cfg = SIDConfig()
    p = argparse.ArgumentParser(
        description="VoxCeleb1 speaker identification probing."
    )
    p.add_argument(
        "--probe", choices=["final", "weighted", "fixed_weighted"], default=cfg.probe_type,
        help="Probe head: 'final' (single layer + mean pool + linear) or "
             "'weighted' (softmax layer mix + mean pool + linear) or "
             "'fixed_weighted' (uniform layer average + mean pool + linear).",
    )
    p.add_argument("--voxceleb1_root", required=True,
                   help="Path to the VoxCeleb1 root directory (contains dev/ and test/).")
    p.add_argument("--model_id",     default=cfg.model_id)
    p.add_argument("--model_family", default=cfg.model_family,
                   choices=["spear", "hf", "disentanglement"])
    p.add_argument("--checkpoint_path", default=None,
                   help="Disentanglement checkpoint when --model_family=disentanglement.")
    p.add_argument("--representation_source", choices=["z_t", "z_L", "z_P"],
                   default=cfg.representation_source)
    p.add_argument("--epochs",          type=int,   default=cfg.num_epochs)
    p.add_argument("--batch_size",      type=int,   default=cfg.batch_size)
    p.add_argument("--eval_batch_size", type=int,   default=cfg.eval_batch_size)
    p.add_argument("--lr",              type=float, default=cfg.learning_rate)
    p.add_argument("--warmup_steps",    type=int,   default=cfg.warmup_steps)
    p.add_argument("--layer_idx",       type=int,   default=cfg.layer_idx,
                   help="For --probe final: which encoder layer to use (0-based, -1=last).")
    p.add_argument("--runs_dir",        default=None,
                   help="Directory for run summaries. Default: sid/runs/")
    p.add_argument("--checkpoint_dir",  default=None,
                   help="Directory for checkpoints. Default: sid/checkpoints/")
    p.add_argument("--log_dir",         default=None,
                   help="Directory for log files. Default: sid/logs/")
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--num_workers", type=int, default=cfg.num_workers,
                   help="DataLoader worker processes. Set 0 to avoid /dev/shm issues on HPC.")
    p.add_argument("--train_max_duration_s", type=float, default=cfg.train_max_duration_s,
                   help="Random-crop training utterances to at most this many seconds (SUPERB: 8.0). "
                        "Val and test are always evaluated on full utterances.")
    p.add_argument("--max_examples", type=int, default=cfg.max_examples,
                   help="Cap each split to this many examples (0 = no cap; for smoke tests).")
    args = p.parse_args()

    cfg.probe_type      = args.probe
    cfg.voxceleb1_root  = Path(args.voxceleb1_root)
    cfg.model_id        = args.model_id
    cfg.model_family    = args.model_family
    cfg.checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    cfg.representation_source = args.representation_source
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
    cfg.num_workers     = args.num_workers
    cfg.train_max_duration_s = args.train_max_duration_s
    cfg.max_examples         = args.max_examples
    return cfg


# ---------------------------------------------------------------- Main ---


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    for d in (cfg.runs_dir, cfg.checkpoint_dir, cfg.log_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== probe_type    : {cfg.probe_type}")
    print(f"=== model_id      : {cfg.model_id}")
    print(f"=== model_family  : {cfg.model_family}")
    if cfg.model_family == "disentanglement":
        print(f"=== source        : {cfg.representation_source}")
        print(f"=== encoder ckpt  : {cfg.checkpoint_path}")
    print(f"=== voxceleb1_root: {cfg.voxceleb1_root}")
    print(f"=== checkpoint    : {cfg.checkpoint_dir}")
    print(f"=== runs          : {cfg.runs_dir}")
    print(f"=== logs          : {cfg.log_dir}")
    print(f"=== epochs        : {cfg.num_epochs}  lr={cfg.learning_rate}")
    print(f"=== seed          : {cfg.seed}")

    train_dl, val_dl, test_dl = make_sid_dataloaders(cfg)
    encoder, probe = build_sid_model(cfg)

    from datetime import datetime as _dt
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    tb = TBLogger(cfg.runs_dir / "tb", run_name=f"{ts}_{cfg.probe_type}", task="sid")

    best_val_acc = fit_sid(cfg, encoder, probe, train_dl, val_dl, tb=tb)
    print(f"\n[SID] best val acc : {best_val_acc:.4f}")

    print("\n=== Final test-set evaluation ===")
    test_metrics = evaluate_sid(cfg, encoder, probe, test_dl,
                                label="test", epoch=cfg.num_epochs, tb=tb)
    print(f"[SID] test acc     : {test_metrics['acc']:.4f}")

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
    summary_path = cfg.runs_dir / f"{ts}_sid_{cfg.probe_type}_summary.json"
    summary = {
        "probe_type":     cfg.probe_type,
        "model_id":       cfg.model_id,
        "model_family":   cfg.model_family,
        "encoder_checkpoint": str(cfg.checkpoint_path) if cfg.checkpoint_path else None,
        "representation_source": cfg.representation_source,
        "seed":           cfg.seed,
        "deterministic":  True,
        "num_workers":    cfg.num_workers,
        "num_speakers":   cfg.num_classes,
        "best_val_acc":   best_val_acc,
        "test_acc":       test_metrics["acc"],
        "layer_weights":  layer_weights,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    tb.close()
    print(f"\n[run done] summary saved to {summary_path}")


if __name__ == "__main__":
    main()
