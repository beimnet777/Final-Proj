"""BCE training loop and EER evaluation for Spoof Detection.

Public API
----------
    fit_sd(cfg, encoder, probe, train_dl, val_dl) -> best_val_eer
    evaluate_sd(cfg, encoder, probe, dl, label, epoch) -> {"eer": float, ...}

Label convention: 1 = bonafide,  0 = spoof
Score convention: sigmoid(logit) → high score = more likely bonafide.

EER is computed by finding the threshold where FAR (spoof passed as bonafide)
equals FRR (bonafide rejected), using sklearn.metrics.roc_curve.
Lower EER is better; best checkpoint is saved when val EER decreases.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).parent.parent))
from sd_config import SDConfig


# ====================================================================
# LR schedule
# ====================================================================

def _make_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


# ====================================================================
# EER computation
# ====================================================================

def compute_eer(labels: List[int], scores: List[float]) -> float:
    """Compute Equal Error Rate.

    labels : list of ints  (1=bonafide, 0=spoof)
    scores : list of floats (higher = more likely bonafide, i.e. sigmoid output)
    returns: EER as a fraction in [0, 1]
    """
    labels_arr = np.array(labels, dtype=int)
    scores_arr = np.array(scores, dtype=float)
    if len(np.unique(labels_arr)) < 2:
        return float("nan")   # undefined if only one class present
    fpr, tpr, _ = roc_curve(labels_arr, scores_arr, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(fpr[idx])


# ====================================================================
# Evaluation  (no gradient)
# ====================================================================

@torch.no_grad()
def evaluate_sd(
    cfg: SDConfig,
    encoder,
    probe: nn.Module,
    dl,
    label: str = "val",
    epoch: int = 0,
) -> dict:
    """Score every utterance in dl and return EER.

    Returns {"eer": float, "n_examples": int}.
    """
    if encoder is not None:
        encoder.eval()
    probe.eval()
    device = next(probe.parameters()).device

    all_scores: List[float] = []
    all_labels: List[int]   = []

    for audios, audio_lens, labels in dl:
        audios     = audios.to(device,     non_blocking=True)
        audio_lens = audio_lens.to(device, non_blocking=True)

        hidden     = encoder(audios, audio_lens)
        frame_lens = encoder.output_lengths(audio_lens)
        logits     = probe(hidden, frame_lens).squeeze(-1)   # (B,)
        scores     = torch.sigmoid(logits).cpu().tolist()

        all_scores.extend(scores)
        all_labels.extend(labels.tolist())

    eer = compute_eer(all_labels, all_scores)
    print(f"[{label}] epoch {epoch:>3d}  EER {eer:.4f}  ({len(all_labels)} utts)")
    return {"eer": eer, "n_examples": len(all_labels)}


# ====================================================================
# Training
# ====================================================================

def fit_sd(
    cfg: SDConfig,
    encoder,
    probe: nn.Module,
    train_dl,
    val_dl,
) -> float:
    """Train BCE probe for cfg.num_epochs. Returns best validation EER.

    Lower EER is better. Probe state is restored to best checkpoint before
    returning. Validation (ASV19 LA dev) is run after every epoch.
    Test-set evaluation is done separately in sd_run.py.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    encoder.to(device)
    probe.to(device)

    trainable = [p for p in probe.parameters() if p.requires_grad]
    print(f"frozen encoder params : {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"trainable probe params: {sum(p.numel() for p in trainable):,}")
    print(f"device                : {device}")

    optimizer   = AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.num_epochs * max(1, len(train_dl))
    scheduler   = _make_lr_schedule(optimizer, cfg.warmup_steps, total_steps)
    bce_loss    = nn.BCEWithLogitsLoss()

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_eer   = float("inf")
    best_state = None
    step       = 0

    for epoch in range(cfg.num_epochs):
        probe.train()
        encoder.eval()

        n_batches     = len(train_dl)
        running_loss  = 0.0
        running_count = 0

        for batch_idx, (audios, audio_lens, labels) in enumerate(train_dl):
            audios     = audios.to(device,     non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)
            labels_f   = labels.float().to(device, non_blocking=True)

            with torch.no_grad():
                hidden     = encoder(audios, audio_lens)
                frame_lens = encoder.output_lengths(audio_lens)

            logits = probe(hidden, frame_lens).squeeze(-1)   # (B,)
            loss   = bce_loss(logits, labels_f)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss  += loss.item() * audios.size(0)
            running_count += audios.size(0)
            step += 1

            log_every = max(1, min(cfg.log_every, n_batches // 2 or 1))
            if step % log_every == 0 or batch_idx == n_batches - 1:
                lr_now = optimizer.param_groups[0]["lr"]
                avg    = running_loss / max(1, running_count)
                print(
                    f"  epoch {epoch + 1:>2d}/{cfg.num_epochs}"
                    f"  step {step:>6d}"
                    f"  bce {avg:.4f}"
                    f"  lr {lr_now:.2e}"
                )
                running_loss = running_count = 0

        if hasattr(probe, "layer_weights"):
            weights_str = "  ".join(f"{w:.3f}" for w in probe.layer_weights.tolist())
            print(f"  layer weights: [{weights_str}]")

        metrics = evaluate_sd(cfg, encoder, probe, val_dl,
                               label="val", epoch=epoch + 1)
        val_eer = metrics["eer"]

        if not np.isnan(val_eer) and val_eer < best_eer:
            best_eer   = val_eer
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            ckpt_path  = (
                Path(cfg.checkpoint_dir)
                / f"sd_probe_{cfg.probe_type}_best.pt"
            )
            torch.save(
                {"probe_state": best_state, "val_eer": best_eer, "epoch": epoch + 1},
                ckpt_path,
            )
            print(f"  ✓ saved best probe (val_eer={best_eer:.4f}) → {ckpt_path}")

    if best_state is not None:
        probe.load_state_dict(best_state)

    return best_eer
