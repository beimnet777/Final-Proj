"""CTC training loop and Phone Error Rate evaluation for Phone Recognition.

Public API
----------
    fit_pr(cfg, encoder, probe, tokenizer, train_dl, val_dl) -> best_val_per
    evaluate_pr(cfg, encoder, probe, tokenizer, dl, label, epoch) -> {"per": float, ...}

PER is computed at the phone level: each phone token is treated as a "word"
in jiwer's WER calculation (space-separated phone strings).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import jiwer

sys.path.insert(0, str(Path(__file__).parent.parent))
from pr_config import PRConfig


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
# Greedy CTC decoding
# ====================================================================

def greedy_ctc_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    tokenizer,
    blank_id: int = 0,
) -> List[str]:
    """Best-path CTC: argmax, collapse repeats, drop blanks.

    log_probs : (B, T, V)
    lengths   : (B,)  valid frame counts
    returns   : list of space-separated phone strings, one per example
    """
    preds = log_probs.argmax(dim=-1)   # (B, T)
    out: List[str] = []
    for row, n in zip(preds, lengths.tolist()):
        ids = row[:n].tolist()
        collapsed, prev = [], -1
        for i in ids:
            if i != prev:
                collapsed.append(i)
                prev = i
        keep = [i for i in collapsed if i != blank_id]
        out.append(tokenizer.decode(keep))
    return out


# ====================================================================
# PER helpers
# ====================================================================

def _safe_refs(refs: List[str]) -> List[str]:
    """jiwer requires non-empty reference strings."""
    return [r if r else SPN_FALLBACK for r in refs]

SPN_FALLBACK = "spn"


def phones_from_text(text: str, tokenizer) -> str:
    """Convert a plain transcript to a reference phone string for PER."""
    from pr_data import text_to_phones, _get_lexicon, _LEXICON
    # Reuse the already-loaded lexicon (cached by _get_lexicon after first call).
    # _LEXICON is None until make_pr_dataloaders runs, so we rely on the module-
    # level cache being populated before evaluate_pr is ever called.
    import pr_data as _pr_data
    lexicon = _pr_data._LEXICON  # already populated by make_pr_dataloaders
    if lexicon is None:
        raise RuntimeError(
            "pr_data._LEXICON is not loaded. "
            "Call make_pr_dataloaders() before evaluate_pr()."
        )
    phones = text_to_phones(text, lexicon)
    ids    = tokenizer.encode(phones).tolist()
    return tokenizer.decode(ids)


# ====================================================================
# Evaluation
# ====================================================================

@torch.no_grad()
def evaluate_pr(
    cfg: PRConfig,
    encoder,
    probe: nn.Module,
    tokenizer,
    dl,
    label: str = "val",
    epoch: int = 0,
) -> dict:
    """Compute PER over a DataLoader."""
    if encoder is not None:
        encoder.eval()
    probe.eval()
    device = next(probe.parameters()).device

    all_hyps: List[str] = []
    all_refs: List[str] = []

    for audios, audio_lens, targets, target_lens, texts in dl:
        audios     = audios.to(device,     non_blocking=True)
        audio_lens = audio_lens.to(device, non_blocking=True)

        hidden     = encoder(audios, audio_lens)
        frame_lens = encoder.output_lengths(audio_lens)

        log_probs  = probe(hidden, frame_lens)      # (B, T, V)
        hyps       = greedy_ctc_decode(
            log_probs.cpu(), frame_lens.cpu(), tokenizer
        )

        refs = [phones_from_text(t, tokenizer) for t in texts]
        all_hyps.extend(hyps)
        all_refs.extend(refs)

    safe_refs = _safe_refs(all_refs)
    per = jiwer.wer(safe_refs, all_hyps)
    print(f"[{label}] epoch {epoch:>3d}  PER {per:.4f}")
    return {"per": per, "n_examples": len(all_refs)}


# ====================================================================
# Training
# ====================================================================

def fit_pr(
    cfg: PRConfig,
    encoder,
    probe: nn.Module,
    tokenizer,
    train_dl,
    val_dl,
) -> float:
    """Train CTC probe for cfg.num_epochs. Returns best validation PER.

    Lower PER is better. Probe state is restored to the best checkpoint
    before returning.
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
    ctc_loss    = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_per   = float("inf")
    best_state = None
    step       = 0

    for epoch in range(cfg.num_epochs):
        probe.train()
        encoder.eval()

        n_batches     = len(train_dl)
        running_loss  = 0.0
        running_count = 0

        for batch_idx, (audios, audio_lens, targets, target_lens, _) in enumerate(train_dl):
            audios      = audios.to(device,      non_blocking=True)
            audio_lens  = audio_lens.to(device,  non_blocking=True)
            targets     = targets.to(device,     non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            with torch.no_grad():
                hidden     = encoder(audios, audio_lens)
                frame_lens = encoder.output_lengths(audio_lens)

            log_probs = probe(hidden, frame_lens)   # (B, T, V)

            # CTCLoss expects (T, B, V)
            loss = ctc_loss(
                log_probs.permute(1, 0, 2),
                targets,
                frame_lens,
                target_lens,
            )

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
                running_loss = running_count = 0

        if hasattr(probe, "layer_weights"):
            weights_str = "  ".join(f"{w:.3f}" for w in probe.layer_weights.tolist())
            print(f"  layer weights: [{weights_str}]")

        metrics = evaluate_pr(cfg, encoder, probe, tokenizer, val_dl,
                              label="val", epoch=epoch + 1)
        val_per = metrics["per"]
        if val_per < best_per:
            best_per   = val_per
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            ckpt_path  = (
                Path(cfg.checkpoint_dir)
                / f"pr_probe_{cfg.probe_type}_best.pt"
            )
            torch.save(
                {"probe_state": best_state, "val_per": best_per, "epoch": epoch + 1},
                ckpt_path,
            )
            print(f"  ✓ saved best probe (val_per={best_per:.4f}) → {ckpt_path}")

    if best_state is not None:
        probe.load_state_dict(best_state)

    return best_per
