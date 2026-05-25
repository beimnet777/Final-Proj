"""Cross-entropy training loop and accuracy evaluation for Speaker ID.

Public API
----------
    fit_sid(cfg, encoder, probe, train_dl, val_dl) -> best_val_acc
    evaluate_sid(cfg, encoder, probe, dl, label, epoch) -> {"acc": float, ...}
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).parent.parent))
from sid_config import SIDConfig
from tb_logger import TBLogger


# --------------------------------------------------------- LR schedule ----


def _make_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------- Evaluation ----


@torch.no_grad()
def evaluate_sid(
    cfg: SIDConfig,
    encoder,
    probe: nn.Module,
    dl,
    label: str = "val",
    epoch: int = 0,
    tb=None,
) -> dict:
    """Compute top-1 accuracy over a DataLoader."""
    if encoder is not None:
        encoder.eval()
    probe.eval()
    device = next(probe.parameters()).device

    correct = 0
    total   = 0

    for audios, audio_lens, labels in dl:
        audios     = audios.to(device,     non_blocking=True)
        audio_lens = audio_lens.to(device, non_blocking=True)
        labels     = labels.to(device,     non_blocking=True)

        hidden     = encoder(audios, audio_lens)
        frame_lens = encoder.output_lengths(audio_lens)
        logits     = probe(hidden, frame_lens)        # (B, num_speakers)

        preds    = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"[{label}] epoch {epoch:>3d}  acc {acc:.4f}  ({correct}/{total})")
    if tb is not None:
        tb.log_eval(epoch, label, {"acc": acc})
    return {"acc": acc, "correct": correct, "total": total}


# ------------------------------------------------------------ Training ----


def fit_sid(
    cfg: SIDConfig,
    encoder,
    probe: nn.Module,
    train_dl,
    val_dl,
    tb=None,
) -> float:
    """Train the probe for cfg.num_epochs with cross-entropy loss.

    Encoder stays frozen throughout. Returns best validation accuracy.
    Probe state is restored to the best checkpoint before returning.
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
    ce_loss     = nn.CrossEntropyLoss()

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_acc   = 0.0
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
            labels     = labels.to(device,     non_blocking=True)

            with torch.no_grad():
                hidden     = encoder(audios, audio_lens)
                frame_lens = encoder.output_lengths(audio_lens)

            logits = probe(hidden, frame_lens)
            loss   = ce_loss(logits, labels)

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
                    f"  loss {avg:.4f}"
                    f"  lr {lr_now:.2e}"
                )
                if tb is not None:
                    tb.log_train_step(step, avg, lr_now)
                running_loss = running_count = 0

        if hasattr(probe, "layer_weights"):
            weights_str = "  ".join(f"{w:.3f}" for w in probe.layer_weights.tolist())
            print(f"  layer weights: [{weights_str}]")
            if tb is not None:
                tb.log_layer_weights(epoch + 1, probe.layer_weights.tolist())

        metrics = evaluate_sid(cfg, encoder, probe, val_dl,
                               label="val", epoch=epoch + 1, tb=tb)
        if metrics["acc"] > best_acc:
            best_acc   = metrics["acc"]
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            ckpt_path  = (
                Path(cfg.checkpoint_dir)
                / f"sid_probe_{cfg.probe_type}_best.pt"
            )
            torch.save(
                {"probe_state": best_state, "val_acc": best_acc, "epoch": epoch + 1},
                ckpt_path,
            )
            print(f"  ✓ saved best probe (val_acc={best_acc:.4f}) → {ckpt_path}")

    if best_state is not None:
        probe.load_state_dict(best_state)

    return best_acc
