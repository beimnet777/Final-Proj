"""Single-stage training for the old+Gao disentanglement system.

All losses active from step 1: recon + decorr + route + PR + SID + GRL.
No staged training — routing and task heads optimised from the start,
giving P a reason to exist immediately (SID) and L a reason to be
linguistic (CTC), with GRL preventing speaker leakage into z_L.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import DISConfig
from model import DISModel, build_dis_model
from data.dataset import make_dis_dataloaders, PhoneTokenizer
from losses import recon_loss, ctc_pr_loss, sid_ce_loss, decorr_loss, route_loss
from tb_logger import DISLogger


# ---------------------------------------------------------------- helpers

def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_scheduler(optimizer: AdamW, warmup: int, total: int) -> LambdaLR:
    def _lr(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return 1.0
    return LambdaLR(optimizer, _lr)


def _dann_lambda(step: int, total: int) -> float:
    p = step / max(1, total)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def _gumbel_tau(step: int, total: int, tau_start: float, tau_end: float) -> float:
    return tau_start * (tau_end / tau_start) ** (step / max(1, total))


def _build_optimizer(cfg: DISConfig, model: DISModel) -> AdamW:
    """All parameters optimised from step 1."""
    routing_params = list(model.routing.parameters())
    other_params = (
        list(model.encoder.parameters()) +
        list(model.sae.parameters()) +
        list(model.pr_head.parameters()) +
        list(model.sid_head.parameters()) +
        list(model.grl_head.parameters())
    )
    return AdamW(
        [
            {"params": routing_params, "lr": cfg.lr_routing},
            {"params": other_params,   "lr": cfg.lr_enc_dec},
        ],
        weight_decay=cfg.weight_decay,
    )


def _trainable_params(model: DISModel):
    return (
        list(model.encoder.parameters()) +
        list(model.sae.parameters()) +
        list(model.routing.parameters()) +
        list(model.pr_head.parameters()) +
        list(model.sid_head.parameters()) +
        list(model.grl_head.parameters())
    )


def _count_params(model: DISModel) -> Tuple[int, int]:
    frozen    = sum(p.numel() for p in model.encoder._spear.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen, trainable


def _save_checkpoint(
    path: Path, model: DISModel, optimizer: AdamW,
    scheduler: LambdaLR, step: int, best_metric: float, cfg: DISConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("encoder._spear.")
    }
    torch.save({
        "step": step, "best_metric": best_metric,
        "model_state": trainable_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "num_speakers": cfg.num_speakers,
        "vocab_size":   cfg.vocab_size,
    }, path)


def _load_checkpoint(path: Path, model: DISModel, cfg: DISConfig) -> dict:
    ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    cfg.num_speakers = ckpt.get("num_speakers", cfg.num_speakers)
    cfg.vocab_size   = ckpt.get("vocab_size",   cfg.vocab_size)
    return ckpt


# ---------------------------------------------------------------- fast probe

@torch.no_grad()
def _fast_probe_snapshot(model, val_dl, tokenizer, cfg, step, tb,
                         max_probe_batches=6, probe_steps=150):
    device = cfg.device
    model.eval()
    z_P_bar_list, z_L_mean_list, sid_list, pr_ctc_vals = [], [], [], []

    for i, batch in enumerate(val_dl):
        if i >= max_probe_batches:
            break
        audios, audio_lens, targets, target_lens, speaker_ids, _ = batch
        audios      = audios.to(device)
        audio_lens  = audio_lens.to(device)
        targets     = targets.to(device)
        target_lens = target_lens.to(device)
        speaker_ids = speaker_ids.to(device)

        out = model(audios, audio_lens, grl_lambda=0.0)
        z_P_bar_list.append(out["z_P_bar"].cpu())

        B, T, K = out["z_L"].shape
        mask = (torch.arange(T, device=device).unsqueeze(0)
                < out["out_lengths"].unsqueeze(1)).float().unsqueeze(-1)
        z_L_mean = (out["z_L"] * mask).sum(1) / out["out_lengths"].float().unsqueeze(1).clamp(min=1)
        z_L_mean_list.append(z_L_mean.cpu())
        sid_list.append(speaker_ids.cpu())
        pr_ctc_vals.append(
            ctc_pr_loss(out["pr_logits"], targets, out["out_lengths"], target_lens).item()
        )

    if not z_P_bar_list:
        model.train(); return

    Z_P  = torch.cat(z_P_bar_list)
    Z_L  = torch.cat(z_L_mean_list)
    SIDS = torch.cat(sid_list)

    def _probe(X, y, n_cls):
        fc  = nn.Linear(X.size(1), n_cls).to(device)
        opt = torch.optim.Adam(fc.parameters(), lr=3e-3)
        X_d, y_d = X.to(device), y.to(device)
        for _ in range(probe_steps):
            opt.zero_grad(set_to_none=True)
            nn.CrossEntropyLoss()(fc(X_d), y_d).backward(); opt.step()
        with torch.no_grad():
            acc = (fc(X_d).argmax(-1) == y_d).float().mean().item()
        return acc

    sid_acc  = _probe(Z_P, SIDS, cfg.num_speakers)
    leak_sid = _probe(Z_L, SIDS, cfg.num_speakers)
    pr_ctc_v = sum(pr_ctc_vals) / len(pr_ctc_vals)
    print(f"  [probe @{step}]  SID(z̄_P)={sid_acc:.3f}"
          f"  Leak_SID(z_L)={leak_sid:.3f}  PR_CTC_val={pr_ctc_v:.4f}")
    tb.log_probe(step, {"sid_acc": sid_acc, "leak_sid": leak_sid, "pr_ctc_val": pr_ctc_v})
    model.train()


# ---------------------------------------------------------------- training loop

def run(cfg: DISConfig) -> Path:
    """Single-stage training. Returns path to best checkpoint."""
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    tokenizer, speaker_to_idx, train_dl, val_dl = make_dis_dataloaders(cfg)

    model = build_dis_model(cfg)
    frozen, trainable = _count_params(model)
    print(f"[train] frozen={frozen:,}  trainable={trainable:,}  device={device}")

    model.train()
    optimizer  = _build_optimizer(cfg, model)
    total_steps = cfg.total_steps
    scheduler  = _make_scheduler(optimizer, cfg.warmup_steps, total_steps)

    use_bf16 = cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler   = torch.amp.GradScaler("cuda", enabled=(not use_bf16))

    from datetime import datetime
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb  = DISLogger(cfg.runs_dir / "tb", run_name=f"run_{ts}")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    best_metric = float("inf")
    best_ckpt   = cfg.checkpoint_dir / "best.pt"
    train_iter  = iter(train_dl)

    print(f"[train] {total_steps} steps  batch={cfg.batch_size}  "
          f"delta={cfg.delta}  rho={cfg.rho}")

    step = 0
    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        audios, audio_lens, targets, target_lens, speaker_ids, _ = batch
        audios      = audios.to(device, non_blocking=True)
        audio_lens  = audio_lens.to(device, non_blocking=True)
        targets     = targets.to(device, non_blocking=True)
        target_lens = target_lens.to(device, non_blocking=True)
        speaker_ids = speaker_ids.to(device, non_blocking=True)

        model.routing.tau = _gumbel_tau(
            step, total_steps, cfg.gumbel_tau_start, cfg.gumbel_tau_end)
        grl_lam = _dann_lambda(step, total_steps)

        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16
            else torch.autocast("cuda", enabled=False)
        )
        with autocast_ctx:
            out = model(audios, audio_lens, grl_lambda=grl_lam)

            l_recon  = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
            l_decorr = decorr_loss(out["z_t"], out["out_lengths"], cfg.decorr_max_frames)
            l_route  = route_loss(model.routing.logits)
            l_pr     = ctc_pr_loss(out["pr_logits"], targets,
                                   out["out_lengths"], target_lens)
            l_sid    = sid_ce_loss(out["sid_logits"], speaker_ids)
            l_grl    = sid_ce_loss(out["grl_logits"], speaker_ids)

            total = (l_recon
                     + cfg.delta * l_decorr
                     + cfg.rho   * l_route
                     + cfg.alpha * l_pr
                     + cfg.beta  * l_sid
                     + l_grl)

            losses = {
                "recon":  l_recon.item(),  "decorr": l_decorr.item(),
                "route":  l_route.item(),  "pr":     l_pr.item(),
                "sid":    l_sid.item(),    "grl":    l_grl.item(),
                "total":  total.item(),
            }

        if use_bf16:
            total.backward()
        else:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(_trainable_params(model), cfg.grad_clip)

        if use_bf16:
            optimizer.step()
        else:
            scaler.step(optimizer); scaler.update()

        scheduler.step()
        step += 1

        # ---- logging
        if step % cfg.log_every == 0 or step == 1:
            n_L, n_P, n_U = model.routing.hard_counts
            entropy  = model.routing.routing_entropy
            tau_now  = model.routing.tau
            lr_now   = optimizer.param_groups[1]["lr"]

            with torch.no_grad():
                density = (out["z_pre"] > 0).float().mean().item()

                hard_idx = model.routing.logits.argmax(dim=-1)
                m_L_h = (hard_idx == 0).float().to(out["z_t"].device)
                m_P_h = (hard_idx == 1).float().to(out["z_t"].device)
                m_U_h = (hard_idx == 2).float().to(out["z_t"].device)
                z_act = (out["z_t"] != 0).float()
                T_    = z_act.shape[1]
                fm    = (torch.arange(T_, device=z_act.device).unsqueeze(0)
                         < out["out_lengths"].unsqueeze(1)).float()
                nv    = fm.sum().clamp(min=1)
                act_L = ((z_act * m_L_h).sum(-1) * fm).sum() / nv
                act_P = ((z_act * m_P_h).sum(-1) * fm).sum() / nv
                act_U = ((z_act * m_U_h).sum(-1) * fm).sum() / nv

            print(
                f"  step {step:>6d}/{total_steps}"
                f"  recon={losses['recon']:.4f}"
                f"  pr={losses['pr']:.4f}"
                f"  sid={losses['sid']:.4f}"
                f"  route_H={entropy:.3f}"
                f"  L/P/U={n_L}/{n_P}/{n_U}"
                f"  actL/P/U={act_L.item():.0f}/{act_P.item():.0f}/{act_U.item():.0f}"
                f"  tau={tau_now:.3f}  lr={lr_now:.2e}"
            )
            tb.log_train(step, losses)
            tb.log_routing(step, n_L, n_P, n_U, entropy)
            tb.log_layer_weights(step, model.encoder.layer_weights.tolist())
            tb.log_sae(step, density, act_L.item(), act_P.item(), act_U.item(), cfg.topk)

        if step % cfg.probe_every == 0:
            _fast_probe_snapshot(model, val_dl, tokenizer, cfg, step, tb)
            model.train()

        if step % cfg.ckpt_every == 0 or step == total_steps:
            metric = losses["recon"]
            if metric < best_metric or step == total_steps:
                best_metric = metric
                _save_checkpoint(best_ckpt, model, optimizer, scheduler,
                                 step, best_metric, cfg)
                print(f"  ✓ checkpoint saved (step={step}  metric={best_metric:.4f})"
                      f" → {best_ckpt}")
            rolling = cfg.checkpoint_dir / f"step{step}.pt"
            _save_checkpoint(rolling, model, optimizer, scheduler,
                             step, best_metric, cfg)
            tb.flush()

    tb.close()
    print(f"\n[train] done.  Best checkpoint: {best_ckpt}")
    return best_ckpt
