"""Per-run structured logging for downstream analysis.

Each run gets its own directory under `runs/`:

    runs/20260513_142300_weighted/
        config.json         # full Config snapshot
        train.csv           # one row per logged train step
        eval.csv            # one row per validation/test pass
        predictions.jsonl   # a few hyp/ref pairs per eval (sanity / qualitative)

CSV files are append-only and have a fixed schema, so you can pandas.read_csv
them straight into a notebook to draw loss curves, CER/WER vs epoch, etc.

Loading example
---------------
    import pandas as pd
    train = pd.read_csv("runs/20260513_142300_weighted/train.csv")
    eval_ = pd.read_csv("runs/20260513_142300_weighted/eval.csv")
    train.plot(x="step", y="loss")
    eval_[eval_.split.str.startswith("val")].plot(x="epoch", y=["cer", "wer"])
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# Fixed column orders so downstream readers don't have to deal with
# header drift between runs.
_TRAIN_COLS = ["step", "epoch", "batch_idx", "loss", "lr", "wall_time"]
_EVAL_COLS = ["epoch", "split", "n_examples", "cer", "wer", "wall_time"]


class RunLogger:
    """Append-only CSV/JSON logger scoped to a single training run."""

    def __init__(self, root: Path, probe_type: str, cfg) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(root) / f"{ts}_{probe_type}"
        self.dir.mkdir(parents=True, exist_ok=True)

        self.train_csv = self.dir / "train.csv"
        self.eval_csv = self.dir / "eval.csv"
        self.predictions_jsonl = self.dir / "predictions.jsonl"
        self.summary_json = self.dir / "summary.json"
        self._t0 = time.time()

        # Snapshot the config so any analysis run later can recover the
        # exact hyperparameters that produced these numbers.
        with (self.dir / "config.json").open("w") as f:
            json.dump(_jsonable(cfg.__dict__), f, indent=2)

        # Initialise CSV files with headers.
        self.train_csv.write_text(",".join(_TRAIN_COLS) + "\n")
        self.eval_csv.write_text(",".join(_EVAL_COLS) + "\n")

        print(f"[logger] writing to {self.dir}")

    # ------------------------------------------------------------------ train

    def log_train_step(self, *, step: int, epoch: int, batch_idx: int,
                       loss: float, lr: float) -> None:
        row = [step, epoch, batch_idx, f"{loss:.6f}", f"{lr:.6e}",
               f"{time.time() - self._t0:.2f}"]
        with self.train_csv.open("a") as f:
            f.write(",".join(str(x) for x in row) + "\n")

    # ------------------------------------------------------------------- eval

    def log_eval(self, *, epoch: int, split: str, n_examples: int,
                 cer: float, wer: float,
                 sample_predictions: Optional[Sequence[Tuple[str, str]]] = None
                 ) -> None:
        """Record one evaluation pass.

        sample_predictions: optional list of (hyp, ref) pairs to dump into
        predictions.jsonl for qualitative inspection.
        """
        row = [epoch, split, n_examples, f"{cer:.6f}", f"{wer:.6f}",
               f"{time.time() - self._t0:.2f}"]
        with self.eval_csv.open("a") as f:
            f.write(",".join(str(x) for x in row) + "\n")

        if sample_predictions:
            with self.predictions_jsonl.open("a") as f:
                for hyp, ref in sample_predictions:
                    f.write(json.dumps({
                        "epoch": epoch,
                        "split": split,
                        "ref": ref,
                        "hyp": hyp,
                    }) + "\n")

    # ------------------------------------------------------- summary / close

    def write_summary(self, summary: dict) -> None:
        """Write a small machine-readable summary at the end of the run.

        Typical contents: best_val_cer, final_test_cer, final_test_wer,
        learned layer weights, total wall time.
        """
        summary = {**_jsonable(summary), "wall_time_total": time.time() - self._t0}
        with self.summary_json.open("w") as f:
            json.dump(summary, f, indent=2)


# --------------------------------------------------------------------- utils


def _jsonable(obj):
    """Recursively coerce non-JSON-serialisable values (Path, etc.) to str."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
