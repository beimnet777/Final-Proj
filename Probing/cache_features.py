"""One-shot feature extraction for the final-layer probe.

Runs the frozen SPEAR encoder once over train, val, and test splits and saves
the final transformer layer's hidden states (fp16) to disk.  Subsequent
training runs load these cached tensors, skipping the 596M-param encoder
entirely — cutting training time from ~800 h to < 1 h.

Cache layout (one file per split, each ~7 GB at fp16 for 100 h train):
    cache_dir/
        train.pt  -- list of {"feat": (T_i, 1280) fp16, "target": (T_text,), "text": str}
        val.pt
        test.pt

Usage
-----
    python cache_features.py --cache_dir ./feature_cache --train_hours 100
    # then probe with:
    python run.py --probe final --feature_cache_dir ./feature_cache
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.data import DataLoader

from config import Config
from data import make_dataloaders
from model import FrozenSpear


# ---------------------------------------------------------------------------

def _extract_split(encoder: FrozenSpear, dl: DataLoader, split: str,
                   layer_idx: int, device: torch.device) -> list:
    """Run encoder over every batch in dl; return list of per-example dicts."""
    records = []
    n_batches = len(dl)
    print(f"[extract:{split}]  {len(dl.dataset)} examples  {n_batches} batches")

    with torch.no_grad():
        for batch_idx, (audios, audio_lens, targets, target_lens, texts) in enumerate(dl):
            audios    = audios.to(device, non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)

            layers = encoder(audios, audio_lens)          # list of 13 × (B, T, D)
            feat_batch = layers[layer_idx]                # (B, T, D)  final layer
            frame_lens = encoder.output_lengths(audio_lens)  # (B,)

            # Unpack batch → per-example, move to CPU, cast to fp16.
            B = feat_batch.size(0)
            for i in range(B):
                t = frame_lens[i].item()
                records.append({
                    "feat":   feat_batch[i, :t].cpu().half(),  # (T_i, 1280) fp16
                    "target": targets[i, : target_lens[i]].cpu(),
                    "text":   texts[i],
                })

            if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1:
                print(f"[extract:{split}]   batch {batch_idx + 1:>4d}/{n_batches}  "
                      f"examples so far: {len(records)}")

    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir",    default="./feature_cache",
                   help="Directory to write the cached .pt files.")
    p.add_argument("--train_hours",  type=float, default=100.0)
    p.add_argument("--layer_idx",    type=int,   default=-1,
                   help="Which SPEAR layer to cache (0-based, -1 = last).")
    p.add_argument("--batch_size",   type=int,   default=4,
                   help="Batch size for extraction (no grad, but encoder is big).")
    p.add_argument("--model_id", default="marcoyang/spear-xlarge-speech-audio")
    p.add_argument("--data_cache_dir", default="./data")
    p.add_argument("--device",       default="cuda")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done.
    missing = [s for s in ("train", "val", "test")
               if not (cache_dir / f"{s}.pt").exists()]
    if not missing:
        print(f"All splits already cached in {cache_dir} — nothing to do.")
        return

    cfg = Config()
    cfg.train_hours    = args.train_hours
    cfg.batch_size     = args.batch_size
    cfg.eval_batch_size = args.batch_size  # same size during extraction
    cfg.model_id = args.model_id
    cfg.data_cache_dir = Path(args.data_cache_dir)
    # Disable multiprocessing during extraction — the encoder already saturates
    # the GPU; workers just add RAM pressure.
    cfg.num_workers = 0

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested (--device cuda) but torch.cuda.is_available() is False. "
            "Check that the GPU is allocated and the correct PyTorch+CUDA build is active."
        )
    device = torch.device(args.device)
    print(f"Device: {device}  (CUDA available: {torch.cuda.is_available()})")

    print("Loading SPEAR encoder …")
    encoder = FrozenSpear(cfg.model_id)
    encoder.to(device)
    encoder.eval()

    # Resolve layer_idx to a positive index so the same index is stored
    # in the cache metadata regardless of how the user specified it.
    n_layers = encoder.num_layers
    layer_idx = args.layer_idx % n_layers
    print(f"Caching layer {layer_idx} / {n_layers - 1}  (SPEAR final layer)")

    print("Building dataloaders …")
    _tokenizer, train_dl, val_dl, test_dl = make_dataloaders(cfg)

    split_map = {"train": train_dl, "val": val_dl, "test": test_dl}
    for split, dl in split_map.items():
        out_path = cache_dir / f"{split}.pt"
        if out_path.exists():
            print(f"[extract:{split}]  already exists, skipping.")
            continue
        records = _extract_split(encoder, dl, split, layer_idx, device)
        tmp_path = out_path.with_suffix(".pt.tmp")
        torch.save({"layer_idx": layer_idx, "records": records}, tmp_path)
        tmp_path.rename(out_path)
        mb = out_path.stat().st_size / 1e6
        print(f"[extract:{split}]  saved {len(records)} examples → {out_path}  ({mb:.0f} MB)")

    print(f"\nDone.  Run the probe with:\n"
          f"  python run.py --probe final --feature_cache_dir {cache_dir}")


if __name__ == "__main__":
    main()
