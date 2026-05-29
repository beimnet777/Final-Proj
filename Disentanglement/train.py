"""Training loop for the SAE reconstruction system.

Public API
----------
    run(cfg)  — trains the SAE on LibriSpeech, returns path to best checkpoint.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import DISConfig
from model import DISModel, build_dis_model
from data.dataset import make_dis_dataloaders
from losses import recon_loss
from tb_logger import DISLogger


# ---------------------------------------------------------------- helpers

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_scheduler(
    optimizer: AdamW,
    warmup: int,
    total: int,
    lr_max: float,
    lr_min: float,
) -> LambdaLR:
    """Linear warmup then cosine decay from lr_max → lr_min."""
    def _lr(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (lr_min + (lr_max - lr_min) * cosine) / lr_max
    return LambdaLR(optimizer, _lr)


def _count_params(model: DISModel):
    frozen    = sum(p.numel() for p in model.encoder._spear.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen, trainable


def _save_checkpoint(
    path: Path,
    model: DISModel,
    optimizer: AdamW,
    scheduler: LambdaLR,
    step: int,
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("encoder._spear.")
    }
    torch.save({
        "step":             step,
        "best_metric":      best_metric,
        "model_state":      trainable_state,
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
    }, path)


# ---------------------------------------------------------------- validation

@torch.no_grad()
def _evaluate(
    model: DISModel,
    val_dl,
    device: torch.device,
    use_bf16: bool,
) -> float:
    model.eval()
    total, n = 0.0, 0
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
    for audios, audio_lengths in val_dl:
        audios        = audios.to(device)
        audio_lengths = audio_lengths.to(device)
        with ctx:
            out = model(audios, audio_lengths)
            loss = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
        total += loss.item()
        n     += 1
    model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------- gradient norm snapshot

def _log_grad_norms(
    model: DISModel,
    batch,
    cfg: DISConfig,
    step: int,
    tb: DISLogger,
    use_bf16: bool,
) -> None:
    """SAE gradient norm from recon loss — no weight updates."""
    device = torch.device(cfg.device)
    audios, audio_lengths = batch
    audios        = audios.to(device, non_blocking=True)
    audio_lengths = audio_lengths.to(device, non_blocking=True)

    params = [p for p in model.sae.parameters() if p.requires_grad]
    ctx    = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)

    with ctx:
        out  = model(audios, audio_lengths)
        loss = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
        grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)

    norm = float(sum(g.norm(2).item() ** 2 for g in grads if g is not None) ** 0.5)
    print(f"  [grad_norm @{step}]  SAE |g|={norm:.5f}")
    tb.log_grad_norms(step, {"recon": norm})


# ---------------------------------------------------------------- training loop

def run(cfg: DISConfig) -> Path:
    """Train the SAE.  Returns path to best checkpoint."""
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_dl, val_dl = make_dis_dataloaders(cfg)

    model = build_dis_model(cfg)
    frozen, trainable = _count_params(model)
    print(f"[train] frozen={frozen:,}  trainable={trainable:,}  device={device}")
    print(f"[train] K={cfg.K}  topk={cfg.topk}  D={cfg.D}")
    print(f"[train] steps={cfg.total_steps}  batch={cfg.batch_size}  "
          f"lr={cfg.lr:.1e}→{cfg.lr_min:.1e}  warmup={cfg.warmup_steps}")

    model.train()

    optimizer = AdamW(
        model.sae.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = _make_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps, cfg.lr, cfg.lr_min)

    use_bf16 = cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler   = torch.amp.GradScaler("cuda", enabled=(not use_bf16))

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb = DISLogger(cfg.runs_dir / "tb", run_name=f"sae_{ts}")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    best_metric = float("inf")
    best_ckpt   = cfg.checkpoint_dir / "best.pt"
    train_iter  = iter(train_dl)

    print(f"[train] starting training loop …")

    step = 0
    while step < cfg.total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        # ---- gradient norm snapshot (extra forward, no weight update)
        if step % cfg.grad_log_every == 0:
            _log_grad_norms(model, batch, cfg, step, tb, use_bf16)

        # ---- forward + loss
        audios, audio_lengths = batch
        audios        = audios.to(device, non_blocking=True)
        audio_lengths = audio_lengths.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16
               else torch.autocast("cuda", enabled=False))
        with ctx:
            out  = model(audios, audio_lengths)
            loss = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])

        # ---- backward
        if use_bf16:
            loss.backward()
            nn.utils.clip_grad_norm_(model.sae.parameters(), cfg.grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.sae.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        step += 1

        # ---- logging
        if step % cfg.log_every == 0 or step == 1:
            lr_now  = optimizer.param_groups[0]["lr"]
            density = (out["z_pre"] > 0).float().mean().item()
            print(f"  step {step:>6d}/{cfg.total_steps}  recon={loss.item():.4f}  lr={lr_now:.2e}")
            tb.log_train(step, {"recon": loss.item()})
            tb.log_sae(step, density, cfg.topk)

        # ---- checkpoint (uses val loss to decide best)
        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            val_loss = _evaluate(model, val_dl, device, use_bf16)
            print(f"  [val] step={step}  val_recon={val_loss:.4f}")
            tb.log_val(step, val_loss)

            if val_loss < best_metric:
                best_metric = val_loss
                _save_checkpoint(best_ckpt, model, optimizer, scheduler, step, best_metric)
                print(f"  ✓ best checkpoint (val={best_metric:.4f}) → {best_ckpt}")

            rolling = cfg.checkpoint_dir / f"step{step}.pt"
            _save_checkpoint(rolling, model, optimizer, scheduler, step, best_metric)
            tb.flush()

    tb.close()
    print(f"\n[train] done.  Best val_recon={best_metric:.4f}  checkpoint → {best_ckpt}")
    return best_ckpt
