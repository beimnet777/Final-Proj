"""Training loops for the disentanglement system.

Public API
----------
    run_stage1(cfg)                   — SAE reconstruction on SPEAR features
    run_stage2(cfg, stage1_ckpt)      — full disentanglement objective
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import DISConfig
from model import DISModel, build_dis_model
from data.dataset import make_stage1_dataloaders, make_stage2_dataloaders
from losses import recon_loss, ctc_pr_loss, sid_ce_loss, sid_ce_loss_frames, route_loss, decor_loss, ub_loss
from tb_logger import DISLogger


# ---------------------------------------------------------------- shared helpers

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
    """Linear warmup then cosine decay lr_max → lr_min."""
    def _lr(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (lr_min + (lr_max - lr_min) * cosine) / lr_max
    return LambdaLR(optimizer, _lr)


def _dann_lambda(step: int, total: int) -> float:
    """DANN ramp: 0 at start → 1 at end."""
    p = step / max(1, total)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def _gumbel_tau(step: int, total: int, tau_start: float, tau_end: float) -> float:
    return tau_start * (tau_end / tau_start) ** (step / max(1, total))


def _count_params(model: DISModel):
    frozen    = sum(p.numel() for p in model.encoder._spear.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen, trainable


def _save_checkpoint(path, model, optimizer, scheduler, step, best_metric):
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_state = {k: v for k, v in model.state_dict().items()
                       if not k.startswith("encoder._spear.")}
    torch.save({
        "step":            step,
        "best_metric":     best_metric,
        "model_state":     trainable_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }, path)


def _load_stage1_checkpoint(path: Path, model: DISModel, cfg: DISConfig) -> None:
    ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
    # Stage 1 only trained SAE params — only load those.
    # Routing and task heads are randomly initialised fresh for stage 2.
    sae_state = {k: v for k, v in ckpt["model_state"].items() if k.startswith("sae.")}
    missing, unexpected = model.load_state_dict(sae_state, strict=False)
    non_sae_missing = [k for k in missing if not k.startswith(
        ("routing.", "proj_L.", "proj_P.", "up_L.", "up_P.", "proj_U.", "up_U.",
         "pr_head.", "sid_head.", "grl_head.", "pr_grl_head.", "encoder._spear."))]
    if non_sae_missing:
        print(f"[train] WARNING: unexpected missing SAE keys: {non_sae_missing}")
    print(f"[train] loaded {len(sae_state)} SAE tensors from {path}  (step={ckpt['step']}  val={ckpt['best_metric']:.4f})")


# ---------------------------------------------------------------- PER helpers

def _greedy_ctc_decode(logits: torch.Tensor, lengths: torch.Tensor, blank_id: int = 0):
    """Best-path CTC: argmax → collapse repeats → strip blank.

    logits  : (B, T, V)
    lengths : (B,)
    returns : list of phone-id lists, one per example
    """
    preds = logits.argmax(dim=-1)   # (B, T)
    out = []
    for i, n in enumerate(lengths.tolist()):
        ids, prev = [], -1
        for tok in preds[i, :n].tolist():
            if tok != prev:
                ids.append(tok)
                prev = tok
        out.append([t for t in ids if t != blank_id])
    return out


def _edit_distance(a, b) -> int:
    """Levenshtein distance between two integer sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------- validation

@torch.no_grad()
def _eval_stage1(model, val_dl, device, use_bf16) -> Dict[str, float]:
    model.eval()
    total, n = 0.0, 0
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
    for audios, audio_lengths in val_dl:
        audios, audio_lengths = audios.to(device), audio_lengths.to(device)
        with ctx:
            out  = model(audios, audio_lengths, stage=1)
            loss = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
        total += loss.item(); n += 1
    model.train()
    return {"recon": total / max(n, 1)}


@torch.no_grad()
def _eval_stage2(model, val_dl, device, use_bf16) -> Dict[str, float]:
    model.eval()
    r_total, pr_total = 0.0, 0.0
    per_num, per_den  = 0, 0
    sid_correct, sid_total = 0, 0
    n = 0
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)

    for audios, audio_lengths, targets, target_lengths, speaker_ids in val_dl:
        audios, audio_lengths = audios.to(device), audio_lengths.to(device)
        targets, target_lengths = targets.to(device), target_lengths.to(device)
        speaker_ids = speaker_ids.to(device)

        with ctx:
            out = model(audios, audio_lengths, stage=2, grl_lambda=0.0)
            r   = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
            pr  = ctc_pr_loss(out["pr_logits"], targets, out["out_lengths"], target_lengths)

        r_total  += r.item()
        pr_total += pr.item()
        n        += 1

        # PER — greedy decode vs reference phone ids
        preds = _greedy_ctc_decode(out["pr_logits"], out["out_lengths"])
        for i, pred_ids in enumerate(preds):
            ref_ids = targets[i, :target_lengths[i]].tolist()
            per_num += _edit_distance(pred_ids, ref_ids)
            per_den += len(ref_ids)

        # SID top-1 accuracy
        sid_pred = out["sid_logits"].argmax(dim=-1)
        sid_correct += (sid_pred == speaker_ids).sum().item()
        sid_total   += speaker_ids.size(0)

    model.train()
    return {
        "recon":   r_total   / max(n, 1),
        "pr":      pr_total  / max(n, 1),
        "per":     per_num   / max(per_den, 1),
        "sid_acc": sid_correct / max(sid_total, 1),
    }


# ---------------------------------------------------------------- gradient norm snapshot

def _log_grad_norms_stage1(model, batch, cfg, step, tb, use_bf16) -> None:
    device = torch.device(cfg.device)
    audios, audio_lengths = batch
    audios, audio_lengths = audios.to(device), audio_lengths.to(device)
    params = [p for p in model.sae.parameters() if p.requires_grad]
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
    with ctx:
        out      = model(audios, audio_lengths, stage=1)
        l_recon  = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
        grads_r  = torch.autograd.grad(l_recon, params, create_graph=False, allow_unused=True, retain_graph=True)
    norm_recon = float(sum(g.norm(2).item()**2 for g in grads_r if g is not None) ** 0.5)

    norms = {"recon": norm_recon}
    log_str = f"  [grad_norm @{step}]  recon={norm_recon:.5f}"

    if cfg.decor_weight > 0:
        with ctx:
            l_decor = decor_loss(out["z_t"], audio_lengths)
            grads_d = torch.autograd.grad(cfg.decor_weight * l_decor, params, create_graph=False, allow_unused=True)
        norm_decor = float(sum(g.norm(2).item()**2 for g in grads_d if g is not None) ** 0.5)
        norms["decor"] = norm_decor
        log_str += f"  decor={norm_decor:.5f}"

    print(log_str)
    tb.log_grad_norms(step, norms)


def _log_grad_norms_stage2(model, batch, cfg, step, tb, use_bf16, grl_lam,
                           eff_grl_weight: float = -1.0, grl_p_lam=None) -> None:
    """Per-loss gradient norms on SAE params — used for weight calibration."""
    device = torch.device(cfg.device)
    audios, audio_lengths, targets, target_lengths, speaker_ids = batch
    audios        = audios.to(device)
    audio_lengths = audio_lengths.to(device)
    targets       = targets.to(device)
    target_lengths = target_lengths.to(device)
    speaker_ids   = speaker_ids.to(device)

    sae_params = [p for p in model.sae.parameters() if p.requires_grad]

    def _norm(loss, retain):
        grads = torch.autograd.grad(loss, sae_params, retain_graph=retain,
                                    allow_unused=True, create_graph=False)
        return float(sum(g.norm(2).item()**2 for g in grads if g is not None) ** 0.5)

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
    with ctx:
        _no_routing     = getattr(cfg, 'no_routing', False)
        _projection     = getattr(cfg, 'projection_disentanglement', False)
        _routing_active = (not _no_routing and not _projection and
                           any(p.requires_grad for p in model.routing.parameters()))

        out     = model(audios, audio_lengths, stage=2, grl_lambda=grl_lam,
                        grl_p_lambda=grl_p_lam)
        l_recon = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
        l_pr    = ctc_pr_loss(out["pr_logits"], targets, out["out_lengths"], target_lengths)
        l_sid   = sid_ce_loss(out["sid_logits"], speaker_ids)
        l_grl   = (sid_ce_loss_frames(out["grl_logits"], speaker_ids, out["out_lengths"])
                   if out["grl_logits"].dim() == 3
                   else sid_ce_loss(out["grl_logits"], speaker_ids))
        l_route = (route_loss(model.routing.logits) if _routing_active
                   else l_recon.new_zeros(()))

        raw: Dict[str, float] = {}
        # Exp 1: phoneme GRL on z_P
        _grl_p_w = getattr(cfg, 'grl_phoneme_weight', 0.0)
        l_grl_p_gn = (ctc_pr_loss(out["pr_grl_logits"], targets, out["out_lengths"], target_lengths)
                      if (_grl_p_w > 0 and "pr_grl_logits" in out)
                      else l_recon.new_zeros(()))

        loss_terms = [
            ("recon",    l_recon,              True),
            ("pr",       l_pr,                 True),
            ("sid",      l_sid,                True),
            ("grl",      l_grl,                True),
        ]
        if _grl_p_w > 0:
            # With dann_full_discriminator the weight already lives in the lambda.
            _dann_fix = getattr(cfg, 'dann_full_discriminator', False)
            loss_terms.append(("grl_p", l_grl_p_gn if _dann_fix else _grl_p_w * l_grl_p_gn, True))
        if _routing_active:
            loss_terms.append(("route", cfg.rho * l_route, False))

        for name, loss, retain in loss_terms:
            raw[name] = _norm(loss, retain)
        if not _routing_active:
            raw["route"] = 0.0
        if _grl_p_w == 0:
            raw["grl_p"] = 0.0

    grl_w  = eff_grl_weight if eff_grl_weight >= 0 else cfg.grl_weight
    _grl_p = getattr(cfg, 'grl_phoneme_weight', 0.0)
    norms = {
        "recon":        raw["recon"],
        "pr_raw":       raw["pr"],
        "pr_weighted":  raw["pr"]  * cfg.alpha,
        "sid_raw":      raw["sid"],
        "sid_weighted": raw["sid"] * cfg.beta,
        "grl":          raw["grl"] * grl_w,
        "grl_p":        raw.get("grl_p", 0.0),
        "route":        raw["route"],
    }

    recon_n = norms["recon"]
    lines   = [f"  [grad_norms @{step}]"]
    for k, v in norms.items():
        ratio = v / recon_n if recon_n > 1e-8 else float("nan")
        lines.append(f"    {k:<16s}  |g|={v:.5f}  ratio={ratio:.3f}x recon")
    print("\n".join(lines))
    tb.log_grad_norms(step, norms)


# ================================================================ Stage 1

def run_stage1(cfg: DISConfig) -> Path:
    """Train SAE on SPEAR features.  Returns best checkpoint path."""
    _set_seed(cfg.seed)
    device  = torch.device(cfg.device)

    train_dl, val_dl = make_stage1_dataloaders(cfg)
    model = build_dis_model(cfg)
    frozen, trainable = _count_params(model)
    print(f"[stage 1] frozen={frozen:,}  trainable={trainable:,}  device={device}")
    print(f"[stage 1] K={cfg.K}  topk={cfg.topk}  D={cfg.D}")
    print(f"[stage 1] steps={cfg.total_steps}  batch={cfg.batch_size}  "
          f"lr={cfg.lr:.1e}→{cfg.lr_min:.1e}  warmup={cfg.warmup_steps}")

    model.train()
    optimizer = AdamW(model.sae.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps, cfg.lr, cfg.lr_min)

    use_bf16 = cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler   = torch.amp.GradScaler("cuda", enabled=(not use_bf16))

    from datetime import datetime
    tb = DISLogger(cfg.runs_dir / "tb", run_name=f"stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_metric = float("inf")
    best_ckpt   = cfg.checkpoint_dir / "stage1_best.pt"
    train_iter  = iter(train_dl)

    for step in range(1, cfg.total_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        if step % cfg.grad_log_every == 0:
            _log_grad_norms_stage1(model, batch, cfg, step, tb, use_bf16)

        audios, audio_lengths = batch
        audios        = audios.to(device, non_blocking=True)
        audio_lengths = audio_lengths.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
        with ctx:
            out    = model(audios, audio_lengths, stage=1)
            l_recon = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
            loss    = l_recon
            l_decor = None
            if cfg.decor_weight > 0:
                l_decor = decor_loss(out["z_t"], audio_lengths)
                loss    = loss + cfg.decor_weight * l_decor

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

        if step % cfg.log_every == 0 or step == 1:
            lr_now  = optimizer.param_groups[0]["lr"]
            density = (out["z_pre"] > 0).float().mean().item()
            recon_v = l_recon.item()
            log_dict = {"recon": recon_v, "total": loss.item()}
            if l_decor is not None:
                decor_v = l_decor.item()
                log_dict["decor"]          = decor_v
                log_dict["decor_weighted"] = cfg.decor_weight * decor_v
                print(f"  step {step:>6d}/{cfg.total_steps}  recon={recon_v:.4f}  "
                      f"decor={decor_v:.4f} (w={cfg.decor_weight * decor_v:.4f})  "
                      f"total={loss.item():.4f}  lr={lr_now:.2e}")
            else:
                print(f"  step {step:>6d}/{cfg.total_steps}  recon={recon_v:.4f}  lr={lr_now:.2e}")
            tb.log_train(step, log_dict)
            tb.log_sae(step, density)

        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            val_metrics = _eval_stage1(model, val_dl, device, use_bf16)
            print(f"  [val] step={step}  val_recon={val_metrics['recon']:.4f}")
            tb.log_val(step, val_metrics)
            if val_metrics["recon"] < best_metric:
                best_metric = val_metrics["recon"]
                _save_checkpoint(best_ckpt, model, optimizer, scheduler, step, best_metric)
                print(f"  ✓ best checkpoint (val={best_metric:.4f}) → {best_ckpt}")
            _save_checkpoint(cfg.checkpoint_dir / f"stage1_step{step}.pt",
                             model, optimizer, scheduler, step, best_metric)
            tb.flush()

    tb.close()
    print(f"\n[stage 1] done.  Best val_recon={best_metric:.4f}  → {best_ckpt}")
    return best_ckpt


# ================================================================ Stage 2

def run_stage2(cfg: DISConfig, stage1_ckpt: Optional[Path]) -> Path:
    """Full disentanglement training.  Optionally loads SAE from stage1_ckpt."""
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    tokenizer, train_dl, val_dl = make_stage2_dataloaders(cfg)

    model = build_dis_model(cfg)
    if stage1_ckpt is not None:
        _load_stage1_checkpoint(Path(stage1_ckpt), model, cfg)
    else:
        print("[train] stage2_from_scratch=True — SAE/routing/heads start from initialization")
    frozen, trainable = _count_params(model)
    print(f"[stage 2] frozen={frozen:,}  trainable={trainable:,}  device={device}")
    print(f"[stage 2] speakers={cfg.num_speakers}  vocab={cfg.vocab_size}")
    delay_str  = f"  grl_delay={cfg.grl_delay_steps}" if cfg.grl_delay_steps > 0 else ""
    extra_str  = ""
    if getattr(cfg, 'grl_phoneme_weight', 0.0) > 0:
        extra_str += f"  grl_p={cfg.grl_phoneme_weight}"
    if getattr(cfg, 'ub_weight', 0.0) > 0:
        extra_str += f"  ub={cfg.ub_weight}"
    if getattr(cfg, 'ste_routing', False):
        extra_str += "  ste=True"
    if getattr(cfg, 'hard_gumbel_routing', False):
        extra_str += "  hard_gumbel=True"
    if getattr(cfg, 'projection_disentanglement', False):
        extra_str += f"  projection=True dim={cfg.projection_dim}"
    if getattr(cfg, 'projection_reconstruct', False):
        extra_str += "  recon_via_views=True"
        u_dim = int(getattr(cfg, 'projection_u_dim', 0))
        if u_dim > 0:
            extra_str += f"  z_U(dim={u_dim} l2={cfg.projection_u_l2})"
    if getattr(cfg, 'spear_layernorm', False):
        extra_str += "  spear_ln=True"
    if getattr(cfg, 'grl_frame_level', False):
        extra_str += "  grl_frame_level=True"
    if getattr(cfg, 'instance_norm_zL', False):
        extra_str += "  instance_norm_zL=True"
    if getattr(cfg, 'dann_full_discriminator', False):
        extra_str += "  dann_full_disc=True"
    print(f"[stage 2] α={cfg.alpha}  β={cfg.beta}  grl={cfg.grl_weight}  ρ={cfg.rho}{delay_str}{extra_str}")
    print(f"[stage 2] steps={cfg.stage2_steps}  batch={cfg.batch_size}")

    model.train()

    projection_mode = getattr(cfg, 'projection_disentanglement', False)

    # routing logits may be frozen (fixed_routing); filter to avoid optimizer warnings
    routing_params = ([] if projection_mode
                      else [p for p in model.routing.parameters() if p.requires_grad])
    projection_params = []
    for _m in ("proj_L", "proj_P", "up_L", "up_P", "proj_U", "up_U"):
        if hasattr(model, _m):
            projection_params.extend(getattr(model, _m).parameters())
    param_groups = [
        {"params": list(model.sae.parameters()),         "lr": cfg.lr},
        {"params": routing_params,                       "lr": cfg.lr_routing},
        {"params": (list(model.pr_head.parameters()) +
                    list(model.sid_head.parameters()) +
                    list(model.grl_head.parameters()) +
                    list(model.pr_grl_head.parameters())), "lr": cfg.lr_heads},
    ]
    if projection_params:
        param_groups.insert(2, {"params": projection_params, "lr": cfg.lr_heads})
    optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg.warmup_steps, cfg.stage2_steps, cfg.lr, cfg.lr_min)

    use_bf16 = cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler   = torch.amp.GradScaler("cuda", enabled=(not use_bf16))

    from datetime import datetime
    tb = DISLogger(cfg.runs_dir / "tb", run_name=f"stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_metric    = float("inf")
    best_ckpt      = cfg.checkpoint_dir / "stage2_best.pt"
    train_iter     = iter(train_dl)
    no_routing     = getattr(cfg, 'no_routing', False)
    n_routes       = getattr(cfg, 'n_routes', 3)
    routing_active = not no_routing and not projection_mode and bool(routing_params)
    grl_p_weight   = getattr(cfg, 'grl_phoneme_weight', 0.0)
    dann_fix       = getattr(cfg, 'dann_full_discriminator', False)
    ub_w           = getattr(cfg, 'ub_weight', 0.0)
    u_l2_w         = getattr(cfg, 'projection_u_l2', 0.0)   # L2 penalty on residual z_U
    routing_clip_params = [] if projection_mode else list(model.routing.parameters())
    all_params     = (list(model.sae.parameters()) + routing_clip_params + projection_params +
                      list(model.pr_head.parameters()) + list(model.sid_head.parameters()) +
                      list(model.grl_head.parameters()) + list(model.pr_grl_head.parameters()))

    for step in range(1, cfg.stage2_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        # ---- temperature + DANN ramp
        model.routing.tau = _gumbel_tau(step, cfg.stage2_steps,
                                        cfg.gumbel_tau_start, cfg.gumbel_tau_end)
        grl_active        = (cfg.grl_delay_steps == 0 or step >= cfg.grl_delay_steps)
        ramp              = _dann_lambda(step, cfg.stage2_steps) if grl_active else 0.0
        if dann_fix:
            # Canonical DANN: heads train at full strength; the per-adversary
            # weights act only on the reversed (encoder-side) gradient via lambda.
            grl_lam          = cfg.grl_weight * ramp
            grl_p_lam        = grl_p_weight * ramp
            eff_grl_weight   = 1.0
            eff_grl_p_weight = 1.0 if grl_p_weight > 0 else 0.0
        else:
            grl_lam          = ramp
            grl_p_lam        = None
            eff_grl_weight   = cfg.grl_weight if grl_active else 0.0
            eff_grl_p_weight = grl_p_weight

        if step % cfg.grad_log_every == 0:
            _log_grad_norms_stage2(model, batch, cfg, step, tb, use_bf16, grl_lam,
                                   eff_grl_weight, grl_p_lam)

        audios, audio_lengths, targets, target_lengths, speaker_ids = batch
        audios         = audios.to(device, non_blocking=True)
        audio_lengths  = audio_lengths.to(device, non_blocking=True)
        targets        = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        speaker_ids    = speaker_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
        with ctx:
            out     = model(audios, audio_lengths, stage=2, grl_lambda=grl_lam,
                            grl_p_lambda=grl_p_lam)
            l_recon = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
            l_pr    = ctc_pr_loss(out["pr_logits"], targets, out["out_lengths"], target_lengths)
            l_sid   = sid_ce_loss(out["sid_logits"], speaker_ids)
            l_grl   = (sid_ce_loss_frames(out["grl_logits"], speaker_ids, out["out_lengths"])
                       if out["grl_logits"].dim() == 3
                       else sid_ce_loss(out["grl_logits"], speaker_ids))
            l_route    = (route_loss(model.routing.logits)
                          if routing_active else l_recon.new_zeros(()))
            # Exp 1: phoneme GRL on z_P
            l_grl_p    = (ctc_pr_loss(out["pr_grl_logits"], targets, out["out_lengths"], target_lengths)
                          if (grl_p_weight > 0 and "pr_grl_logits" in out)
                          else l_recon.new_zeros(()))
            # Exp 4: U-bucket information bottleneck
            l_ub       = (ub_loss(out["m_L"], out["m_P"])
                          if (ub_w > 0 and not no_routing and not projection_mode and n_routes == 3 and "m_L" in out)
                          else l_recon.new_zeros(()))
            # Reconstructive projection: L2 activity penalty on the residual z_U
            # (the bottleneck that stops z_U becoming a reconstruction shortcut).
            l_u        = (out["z_U"].pow(2).mean()
                          if (u_l2_w > 0 and "z_U" in out)
                          else l_recon.new_zeros(()))
            total      = (l_recon
                          + cfg.alpha        * l_pr
                          + cfg.beta         * l_sid
                          + eff_grl_weight   * l_grl
                          + eff_grl_p_weight * l_grl_p
                          + cfg.rho          * l_route
                          + ub_w             * l_ub
                          + u_l2_w           * l_u)

        if use_bf16:
            total.backward()
            nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            optimizer.step()
        else:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        if step % cfg.log_every == 0 or step == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            if not no_routing and not projection_mode:
                n_L, n_P, n_U = model.routing.hard_counts
                routing_diag = model.routing.routing_diagnostics
                entropy = routing_diag["balance_entropy"]
            else:
                n_L, n_P, n_U = (cfg.K, 0, 0) if no_routing else (0, 0, 0)
                routing_diag = {}
                entropy = float('nan')
            density = (out["z_pre"] > 0).float().mean().item()
            losses  = {"recon": l_recon.item(), "pr": l_pr.item(),
                       "sid": l_sid.item(), "grl": l_grl.item(),
                       "grl_p": l_grl_p.item(), "ub": l_ub.item(),
                       "route": l_route.item(), "total": total.item()}

            with torch.no_grad():
                z_active = (out["z_t"] != 0).float()                 # (B, T, K)
                B_, T_   = z_active.shape[:2]
                fmask    = (torch.arange(T_, device=z_active.device).unsqueeze(0)
                            < out["out_lengths"].unsqueeze(1)).float()
                n_valid  = fmask.sum().clamp(min=1)
                if not no_routing and not projection_mode:
                    hard_idx = model.routing.logits.argmax(dim=-1)       # (K,)
                    act_L = ((z_active * (hard_idx == 0).float()).sum(-1) * fmask).sum() / n_valid
                    act_P = ((z_active * (hard_idx == 1).float()).sum(-1) * fmask).sum() / n_valid
                    act_U = ((z_active * (hard_idx == 2).float()).sum(-1) * fmask).sum() / n_valid
                else:
                    act_L = act_P = act_U = z_active.new_tensor(float('nan'))

            grl_p_str = f"  grl_p={l_grl_p.item():.4f}" if grl_p_weight > 0 else ""
            ub_str    = f"  ub={l_ub.item():.4f}"        if ub_w > 0        else ""
            print(
                f"  step {step:>6d}/{cfg.stage2_steps}"
                f"  recon={l_recon.item():.4f}"
                f"  pr={l_pr.item():.4f}"
                f"  sid={l_sid.item():.4f}"
                f"  grl={l_grl.item():.4f}"
                f"{grl_p_str}{ub_str}"
                f"  L/P/U={n_L}/{n_P}/{n_U}"
                f"  actL/P/U={act_L.item():.0f}/{act_P.item():.0f}/{act_U.item():.0f}"
                f"  H={entropy:.3f}"
                f"  Hu={routing_diag.get('unit_entropy', float('nan')):.3f}"
                f"  spec<.5={routing_diag.get('specialized_frac_h_lt_0_5', float('nan')):.2f}"
                f"  marg={routing_diag.get('top1_top2_margin', float('nan')):.3f}"
                f"  lstd={routing_diag.get('logit_std', float('nan')):.4f}"
                f"  lr={lr_now:.2e}"
            )
            tb.log_train(step, losses)
            tb.log_routing(step, n_L, n_P, n_U, entropy, routing_diag)
            tb.log_sae(step, density)

        if step % cfg.ckpt_every == 0 or step == cfg.stage2_steps:
            val_metrics = _eval_stage2(model, val_dl, device, use_bf16)
            print(
                f"  [val] step={step}"
                f"  recon={val_metrics['recon']:.4f}"
                f"  pr={val_metrics['pr']:.4f}"
                f"  PER={val_metrics['per']:.3f}"
                f"  sid_acc={val_metrics['sid_acc']:.3f}"
            )
            tb.log_val(step, val_metrics)
            # Stage 2 optimizes disentanglement, not reconstruction.  recon is
            # *lowest* early — before the task/adversary losses reshape z_t — so
            # selecting "best" by recon returns an undertrained checkpoint (for
            # projection runs it picks step ~1000, where the views are near init
            # and only *look* disentangled because nothing is encoded yet).
            # Select by a disentanglement score instead: low phoneme error in
            # z_L + high speaker accuracy in z_P (both lower-is-better).  These
            # are in-training head metrics — a coarse proxy, but monotone enough
            # within a run to avoid the recon trap.  Per-step checkpoints are
            # still saved, so the recon-best is recoverable if ever needed.
            disent_score = val_metrics["per"] + (1.0 - val_metrics["sid_acc"])
            if disent_score < best_metric:
                best_metric = disent_score
                _save_checkpoint(best_ckpt, model, optimizer, scheduler, step, best_metric)
                print(f"  ✓ best checkpoint (disent={best_metric:.4f}  "
                      f"PER={val_metrics['per']:.3f} sid={val_metrics['sid_acc']:.3f}) → {best_ckpt}")
            _save_checkpoint(cfg.checkpoint_dir / f"stage2_step{step}.pt",
                             model, optimizer, scheduler, step, best_metric)
            tb.flush()

    tb.close()
    print(f"\n[stage 2] done.  Best disent_score={best_metric:.4f}  → {best_ckpt}")
    return best_ckpt
