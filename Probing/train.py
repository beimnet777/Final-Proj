"""CTC training loop, greedy CTC decoding, and CER/WER evaluation.

Public entrypoints:
    fit(cfg, encoder, probe, tokenizer, train_dl, val_dl, logger=None)
    evaluate(cfg, encoder, probe, tokenizer, dl, label="eval", epoch=0,
             logger=None) -> {"cer": float, "wer": float}
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import jiwer

from config import Config
from logger import RunLogger
from model import FrozenSpear


# ----------------------------------------------------------- CTC decoding ---


def greedy_ctc_decode(logits: torch.Tensor, lengths: torch.Tensor,
                      tokenizer, blank_id: int) -> List[str]:
    """Best-path CTC decoding: argmax per frame, collapse repeats, drop blanks.

    logits   : (B, T, V)
    lengths  : (B,)             number of valid frames per example
    returns  : list[str]        decoded transcript per example
    """
    preds = logits.argmax(dim=-1)            # (B, T)
    out: List[str] = []
    for row, n in zip(preds, lengths.tolist()):
        ids = row[:n].tolist()
        # 1) collapse runs of repeated symbols
        collapsed = []
        prev = -1
        for i in ids:
            if i != prev:
                collapsed.append(i)
                prev = i
        # 2) drop blanks
        keep = [i for i in collapsed if i != blank_id]
        out.append(tokenizer.decode(keep))
    return out


# ----------------------------------------------------------- CER / WER -----

# jiwer expects non-empty references; replace empty strings with a single space
# so the corpus-level error rate stays well-defined.
def _safe(refs: List[str]) -> List[str]:
    return [r if r else " " for r in refs]


def compute_cer(hyps: List[str], refs: List[str]) -> float:
    return float(jiwer.cer(_safe(refs), hyps))


def compute_wer(hyps: List[str], refs: List[str]) -> float:
    return float(jiwer.wer(_safe(refs), hyps))


# ----------------------------------------------------------- LR schedule ----


def _make_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)  # linear decay to 0
    return LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------- Evaluation ----


@torch.no_grad()
def evaluate(cfg: Config, encoder: Optional[FrozenSpear], probe: nn.Module,
             tokenizer, dl, label: str = "eval",
             epoch: int = 0, logger: Optional[RunLogger] = None) -> dict:
    if encoder is not None:
        encoder.eval()
    probe.eval()
    device = next(probe.parameters()).device
    n_total = len(dl.dataset)
    n_batches = len(dl)
    print(f"\n[{label}] starting evaluation  examples={n_total}  "
          f"batches={n_batches}  device={device}")

    all_hyps: List[str] = []
    all_refs: List[str] = []
    log_every = max(1, n_batches // 10)  # ~10 progress lines per eval

    for batch_idx, (audios, audio_lens, _targets, _target_lens, texts) in enumerate(dl):
        print(f"[{label}] evaluating batch {batch_idx + 1:>4d}/{n_batches} ...")
        audios = audios.to(device, non_blocking=True)
        audio_lens = audio_lens.to(device, non_blocking=True)

        if encoder is None:
            # Cached mode: audios is already (B, T, D) extracted features;
            # audio_lens are pre-computed frame counts.
            layers = [audios]          # wrap as 1-element list for the probe
            frame_lens = audio_lens
        else:
            layers = encoder(audios, audio_lens)              # list of (B, T, D)
            frame_lens = encoder.output_lengths(audio_lens)   # (B,)
        logits = probe(layers)                            # (B, T, V)

        hyps = greedy_ctc_decode(logits, frame_lens, tokenizer, cfg.blank_id)
        refs = [t.lower() for t in texts]
        all_hyps.extend(hyps)
        all_refs.extend(refs)

        if (batch_idx + 1) % log_every == 0 or batch_idx == n_batches - 1:
            running_cer = compute_cer(all_hyps, all_refs)
            running_wer = compute_wer(all_hyps, all_refs)
            print(f"[{label}]   batch {batch_idx + 1:>4d}/{n_batches}  "
                  f"seen {len(all_hyps):>5d}/{n_total}  "
                  f"running cer {running_cer:.4f}  wer {running_wer:.4f}")

    cer = compute_cer(all_hyps, all_refs)
    wer = compute_wer(all_hyps, all_refs)

    # Show a handful of hyp/ref pairs so you can sanity-check the decoder.
    n_show = min(3, len(all_hyps))
    if n_show:
        print(f"[{label}] sample predictions:")
        for i in range(n_show):
            print(f"  REF[{i}]: {all_refs[i][:120]}")
            print(f"  HYP[{i}]: {all_hyps[i][:120]}")

    print(f"[{label}] DONE  cer {cer:.4f}  wer {wer:.4f}")

    if logger is not None:
        sample_pairs = [(all_hyps[i], all_refs[i]) for i in range(n_show)]
        logger.log_eval(
            epoch=epoch, split=label, n_examples=n_total,
            cer=cer, wer=wer, sample_predictions=sample_pairs,
        )

    return {"cer": cer, "wer": wer}


# ----------------------------------------------------------- Training ------


def fit(cfg: Config, encoder: Optional[FrozenSpear], probe: nn.Module, tokenizer,
        train_dl, val_dl, logger: Optional[RunLogger] = None) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if encoder is not None:
        encoder.to(device)
    probe.to(device)

    # Sanity print.
    trainable = [p for p in probe.parameters() if p.requires_grad]
    if encoder is not None:
        n_frozen = sum(p.numel() for p in encoder.parameters())
        print(f"frozen encoder params : {n_frozen:,}")
    else:
        print("frozen encoder params : (cached — encoder not loaded)")
    n_learn = sum(p.numel() for p in trainable)
    print(f"trainable probe params: {n_learn:,}")

    optimizer = AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.num_epochs * max(1, len(train_dl))
    scheduler = _make_lr_schedule(optimizer, cfg.warmup_steps, total_steps)

    ctc_loss = nn.CTCLoss(blank=cfg.blank_id, zero_infinity=True)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")
    step = 0

    for epoch in range(cfg.num_epochs):
        probe.train()
        if encoder is not None:
            encoder.eval()  # belt-and-braces; FrozenSpear.train() already enforces this

        n_batches = len(train_dl)
        print(f"\nepoch {epoch + 1}/{cfg.num_epochs}  ({n_batches} batches) ...")
        running_loss = 0.0
        running_count = 0

        for batch_idx, (audios, audio_lens, targets, target_lens, _texts) in enumerate(train_dl):
            audios = audios.to(device, non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            # 1) Encoder forward (or use pre-extracted cached features).
            if encoder is None:
                # Cached mode: audios is already (B, T, D); audio_lens are
                # pre-computed frame counts — no encoder call needed.
                layers = [audios]
                input_lens = audio_lens
            else:
                with torch.no_grad():
                    layers = encoder(audios, audio_lens)
                input_lens = encoder.output_lengths(audio_lens)  # (B,)

            # 2) Probe forward. Gradient flows here only.
            logits = probe(layers)                        # (B, T_frames, V)

            # 3) Log-probs and CTC. CTC wants (T, B, V).
            log_probs = F.log_softmax(logits, dim=-1)     # (B, T, V)
            log_probs_t = log_probs.transpose(0, 1)       # (T, B, V)
            loss = ctc_loss(log_probs_t, targets, input_lens, target_lens)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * audios.size(0)
            running_count += audios.size(0)
            step += 1
            log_every = max(1, min(cfg.log_every, n_batches // 2 or 1))
            if step % log_every == 0 or batch_idx == n_batches - 1:
                lr_now = optimizer.param_groups[0]["lr"]
                avg = running_loss / max(1, running_count)
                print(f"  step {step:>6d}  loss {avg:.4f}  lr {lr_now:.2e}")
                if logger is not None:
                    logger.log_train_step(
                        step=step, epoch=epoch + 1, batch_idx=batch_idx,
                        loss=avg, lr=lr_now,
                    )
                running_loss = 0.0
                running_count = 0

        # Snapshot the current weighted-probe layer mixture for analysis.
        # Logged every epoch (not just eval epochs) so the trajectory is dense.
        if logger is not None and hasattr(probe, "layer_weights"):
            logger.log_layer_weights(
                epoch=epoch + 1,
                weights=probe.layer_weights.tolist(),
            )

        # Validation at the end of each epoch (configurable).
        if (epoch + 1) % cfg.eval_every_epochs == 0:
            metrics = evaluate(cfg, encoder, probe, tokenizer, val_dl,
                               label="val", epoch=epoch + 1, logger=logger)
            print(f"epoch {epoch + 1:>3d}  val_cer {metrics['cer']:.4f}  "
                  f"val_wer {metrics['wer']:.4f}")

            if metrics["cer"] < best_cer:
                best_cer = metrics["cer"]
                ckpt = Path(cfg.checkpoint_dir) / f"probe_{cfg.probe_type}_best.pt"
                torch.save({
                    "probe_state": probe.state_dict(),
                    "cfg": cfg.__dict__,
                    "val_cer": metrics["cer"],
                    "val_wer": metrics["wer"],
                    "epoch": epoch + 1,
                }, ckpt)
                print(f"  saved best probe to {ckpt}")
