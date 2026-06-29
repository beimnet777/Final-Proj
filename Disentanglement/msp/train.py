"""Standalone MSP-Podcast disentanglement trainer.

Self-contained: imports the model/losses as read-only libraries but owns its loop.
Every batch carries content + speaker + prosody + emotion, so all heads train each
step — no IEMOCAP-every-8, no _cap_loss_by_scaling.  Gradient conflict on the shared
SAE trunk is handled by PCGrad over the cooperative tasks (see grad_conflict.py).
"""
from __future__ import annotations

import contextlib
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_dis_model
from losses import recon_loss, ctc_pr_loss, sid_ce_loss, route_loss, routing_spec_loss

from .data import make_msp_dataloaders, EMOTION_NAMES
from .grad_conflict import PCGrad, named_gradient_diagnostics
from .heads import GELUSpeakerGRLHead
from . import utils as U
from training_runtime import (
    SegmentLimit, append_metrics, atomic_torch_save, checkpoint_payload,
    mirror_file, restore_training_state, validate_resume,
)


def _autocast(dtype):
    """CUDA autocast for the resolved precision, else a CPU-safe no-op."""
    return (torch.autocast("cuda", dtype=dtype)
            if dtype is not None and torch.cuda.is_available()
            else contextlib.nullcontext())


def _amp_dtype(precision: str):
    if not torch.cuda.is_available() or precision == "fp32":
        return None
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but this GPU does not support it")
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _gumbel_tau(step: int, total: int, tau_start: float, tau_end: float) -> float:
    """Geometric anneal of the Gumbel temperature (matches legacy)."""
    return tau_start * (tau_end / tau_start) ** (step / max(1, total))


# ---------------------------------------------------------------- schedule
def _make_lr(cfg):
    base, warm, total, lo = cfg.lr, cfg.warmup_steps, cfg.stage2_steps, cfg.lr_min
    def lr_at(step):
        if step < warm:
            return base * step / max(1, warm)
        t = (step - warm) / max(1, total - warm)
        return lo + 0.5 * (base - lo) * (1 + math.cos(math.pi * min(1.0, t)))
    return lr_at


def _set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, dl, device, amp_dtype, n_emotion) -> Dict[str, float]:
    model.eval()
    pr_num = pr_den = 0
    sid_c = sid_t = 0
    grl_c = grl_t = 0                       # z_L speaker leakage (adversary)
    grlp_num = grlp_den = 0                 # z_P phoneme leakage (adversary)
    conf_P = torch.zeros(n_emotion, n_emotion)   # z_P emotion (task)
    conf_L = torch.zeros(n_emotion, n_emotion)   # z_L emotion (adversary leakage)
    for b in dl:
        audios = b["audios"].to(device); alen = b["audio_lengths"].to(device)
        tgt = b["targets"].to(device); tlen = b["target_lengths"].to(device)
        spk = b["speaker_ids"].to(device); emo = b["emotion"].to(device)
        with _autocast(amp_dtype):
            out = model(audios, alen, stage=2, grl_lambda=0.0, grl_p_lambda=0.0,
                        grl_emotion_lambda=0.0, emit_emotion=True)
        olen = out["out_lengths"]
        n, d = U.ctc_errors(out["pr_logits"].float(), tgt, olen, tlen); pr_num += n; pr_den += d
        if (spk >= 0).all():
            c, t = U.speaker_correct(out["sid_logits"], spk, olen); sid_c += c; sid_t += t
            c, t = U.speaker_correct(out["grl_logits"], spk, olen); grl_c += c; grl_t += t
        if "pr_grl_logits" in out:
            n, d = U.ctc_errors(out["pr_grl_logits"].float(), tgt, olen, tlen)
            grlp_num += n; grlp_den += d
        ep = out["emotion_logits"].argmax(-1)
        for tlab, plab in zip(emo.tolist(), ep.tolist()):
            conf_P[tlab, plab] += 1
        if "emotion_grl_logits" in out:
            el = out["emotion_grl_logits"].argmax(-1)
            for tlab, plab in zip(emo.tolist(), el.tolist()):
                conf_L[tlab, plab] += 1
    model.train()
    m = {
        "per":        pr_num / max(pr_den, 1),
        "sid_acc":    sid_c / max(sid_t, 1),
        "zL_sid_acc": grl_c / max(grl_t, 1),
        "zP_pr_per":  grlp_num / max(grlp_den, 1),
        "zP_emo_uar": U.uar_from_confusion(conf_P),
        "zP_emo_acc": float(conf_P.diag().sum() / conf_P.sum().clamp(min=1)),
        "zL_emo_uar": U.uar_from_confusion(conf_L),
    }
    return m


# ---------------------------------------------------------------- train
def run(cfg, stage1_ckpt: Optional[str] = None) -> Path:
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)
    amp_dtype = _amp_dtype(str(getattr(cfg, "precision", "auto")))

    loaders = make_msp_dataloaders(cfg)
    tokenizer, train_dl, val_dl, test_dl = loaders[0], loaders[1], loaders[2], loaders[3]
    n_emotion = cfg.emotion_num_classes

    model = build_dis_model(cfg).to(device)
    if stage1_ckpt:
        sd = torch.load(stage1_ckpt, map_location="cpu")
        sd = sd.get("model", sd)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[msp] loaded stage1 SAE from {stage1_ckpt} (missing={len(miss)} unexpected={len(unexp)})")
    else:
        print("[msp] training from scratch (SAE/routing/heads from init)")
    # Speaker adversary → GELU (isolation: overrides model.grl_head without
    # editing the shared model/heads.py).
    model.grl_head = GELUSpeakerGRLHead(cfg).to(device)
    grl_norm = (f"per-frame grad norm target={cfg.grl_grad_norm_target:g}"
                if cfg.grl_grad_norm else "plain gradient reversal")
    print(f"[msp] speaker adversary: GELUSpeakerGRLHead (GELU projector, {grl_norm})")
    model.train()

    # ---- emotion class weights from the train manifest (neutral-heavy) ----
    train_rows = train_dl.dataset.rows
    emo_w = U.emotion_class_weights(train_rows, n_emotion, device)
    print(f"[msp] emotion class weights {[round(float(x),2) for x in emo_w]} "
          f"for {EMOTION_NAMES}")

    # ---- param groups (mirror the legacy lr split, MSP heads only) ----
    sae_params   = list(model.sae.parameters())
    rout_params  = [p for p in model.routing.parameters() if p.requires_grad]
    head_params  = (list(model.pr_head.parameters()) + list(model.sid_head.parameters())
                    + list(model.prosody_head.parameters()) + list(model.emotion_head.parameters()))
    disc_params  = (list(model.grl_head.parameters()) + list(model.pr_grl_head.parameters())
                    + list(model.prosody_grl_head.parameters())
                    + list(model.emotion_grl_head.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": sae_params,  "lr": cfg.lr},
        {"params": rout_params, "lr": cfg.lr_routing},
        {"params": head_params, "lr": cfg.lr_heads},
        {"params": disc_params, "lr": cfg.lr_disc},
    ], betas=(0.9, 0.95), weight_decay=0.0)
    lr_at = _make_lr(cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    all_params = sae_params + rout_params + head_params + disc_params

    pcgrad = PCGrad(sae_params, seed=cfg.seed) if cfg.pcgrad else None
    coop_names = set(cfg.pcgrad_tasks)
    ckpt_dir = Path(getattr(cfg, "checkpoint_dir", Path(__file__).resolve().parent / "checkpoints" / "msp_run"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    accumulation = max(1, int(getattr(cfg, "gradient_accumulation_steps", 1)))
    effective_batch = cfg.batch_size * accumulation
    print(f"[msp] steps={cfg.stage2_steps} microbatch={cfg.batch_size} "
          f"accumulation={accumulation} effective_batch={effective_batch} pcgrad={cfg.pcgrad} "
          f"({sorted(coop_names)})  device={device} amp={amp_dtype}")
    print(f"[msp] weights α(pr)={cfg.alpha} β(sid)={cfg.beta} grl={cfg.grl_weight} "
          f"grl_p={cfg.grl_phoneme_weight} pros={cfg.prosody_weight}/{cfg.grl_prosody_weight} "
          f"emo={cfg.emotion_weight}/{cfg.grl_emotion_weight} inv={cfg.inv_weight}")

    best = float("inf")
    step = 0
    resume_value = str(getattr(cfg, "resume", "none"))
    resume_path = ckpt_dir / "latest-resume.pt" if resume_value == "auto" else Path(resume_value)
    if resume_value not in {"none", ""} and resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        validate_resume(saved, dataset_hash=str(getattr(cfg, "dataset_fingerprint", "")),
                        preset=str(getattr(cfg, "experiment_preset", "")), cfg=cfg)
        step, best = restore_training_state(saved, model=model, optimizer=optimizer, scaler=scaler)
        if pcgrad is not None:
            pcgrad.load_state_dict(saved.get("auxiliary", {}).get("pcgrad", {}))
        sampler_state = saved.get("auxiliary", {}).get("train_sampler")
        if sampler_state and hasattr(train_dl.sampler, "load_state_dict"):
            train_dl.sampler.load_state_dict(sampler_state)
        print(f"[msp] exact resume from {resume_path}: step={step} best={best:.4f}")
    elif resume_value not in {"none", "", "auto"}:
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    segment = SegmentLimit(step, int(getattr(cfg, "segment_steps", 0)),
                           float(getattr(cfg, "max_runtime_minutes", 0.0)))
    metrics_path = ckpt_dir / "metrics.jsonl"
    mirror_dir = Path(cfg.drive_mirror) if str(getattr(cfg, "drive_mirror", "")) else None

    def save_runtime(name: str, *, kind: str = "resume", val=None) -> Path:
        path = ckpt_dir / name
        payload = checkpoint_payload(
            model=model, optimizer=optimizer if kind == "resume" else None,
            scaler=scaler if kind == "resume" else None, step=step,
            best_metric=best, cfg=cfg, dataset_hash=str(getattr(cfg, "dataset_fingerprint", "")),
            preset=str(getattr(cfg, "experiment_preset", "")), kind=kind,
            auxiliary={"pcgrad": pcgrad.state_dict() if pcgrad is not None else {},
                       "train_sampler": (train_dl.sampler.state_dict()
                                         if hasattr(train_dl.sampler, "state_dict") else {}),
                       "val": val or {}},
        )
        atomic_torch_save(payload, path); mirror_file(path, mirror_dir)
        if metrics_path.exists(): mirror_file(metrics_path, mirror_dir)
        return path

    train_iter = iter(train_dl)
    micro_index = 0
    task_grad_sums = None
    extra_grad_sum = None
    disc_micro_batches = []
    while step < cfg.stage2_steps:
        try:
            b = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl); b = next(train_iter)
        micro_index += 1
        next_step = step + 1
        boundary = micro_index == accumulation
        log_now = boundary and (next_step % cfg.log_every == 0 or next_step == 1)
        grad_diag = None
        lr_now = lr_at(next_step)
        optimizer.param_groups[0]["lr"] = lr_now          # SAE follows the cosine
        ramp = min(1.0, next_step / max(1, cfg.warmup_steps))
        model.routing.tau = _gumbel_tau(next_step, cfg.stage2_steps,
                                        cfg.gumbel_tau_start, cfg.gumbel_tau_end)

        audios = b["audios"].to(device); alen = b["audio_lengths"].to(device)
        tgt = b["targets"].to(device); tlen = b["target_lengths"].to(device)
        spk = b["speaker_ids"].to(device); emo = b["emotion"].to(device)
        pert = b.get("pert_audios")
        if pert is not None:
            pert = pert.to(device)

        with _autocast(amp_dtype):
            out = model(audios, alen, stage=2,
                        grl_lambda=ramp, grl_p_lambda=ramp,
                        grl_prosody_lambda=ramp, grl_emotion_lambda=ramp,
                        emit_emotion=True)
            model.sae.update_dead(out["z_t"])
            olen = out["out_lengths"]
            # cooperative tasks
            l_recon = recon_loss(out["h_t"], out["h_hat"], olen)
            l_pr    = ctc_pr_loss(out["pr_logits"], tgt, olen, tlen)
            l_sid   = sid_ce_loss(out["sid_logits"], spk)
            p_f0, p_v, p_e = U.prosody_targets_fast(audios, alen, olen)
            l_pros  = U.prosody_train_loss(out["prosody_pred"], p_f0, p_v, p_e, olen)
            l_emo   = F.cross_entropy(out["emotion_logits"], emo, weight=emo_w)
            l_inv   = out["z_L"].new_zeros(())
            if pert is not None:
                out_p = model(pert, alen, stage=2, grl_lambda=0.0, emit_emotion=False)
                l_inv = U.invariance_loss(out["z_L"], out_p["z_L"], olen)
            # adversaries (kept out of PCGrad — their conflict is the mechanism)
            l_grl     = U.speaker_adv_loss(out["grl_logits"], spk, olen)
            l_grl_p   = ctc_pr_loss(out["pr_grl_logits"], tgt, olen, tlen)
            l_prosgrl = U.prosody_train_loss(out["prosody_grl_pred"], p_f0, p_v, p_e, olen)
            l_emogrl  = F.cross_entropy(out["emotion_grl_logits"], emo, weight=emo_w)
            l_route   = route_loss(out["routing_logits"])
            l_spec    = routing_spec_loss(out["routing_logits"])

        coop = {
            "recon":   1.0 * l_recon,
            "pr":      cfg.alpha * l_pr,
            "sid":     cfg.beta * l_sid,
            "prosody": cfg.prosody_weight * l_pros,
            "emotion": cfg.emotion_weight * l_emo,
            "inv":     cfg.inv_weight * l_inv,
        }
        adv = (cfg.grl_weight * l_grl + cfg.grl_phoneme_weight * l_grl_p
               + cfg.grl_prosody_weight * l_prosgrl + cfg.grl_emotion_weight * l_emogrl
               + cfg.routing_spec_weight * l_spec + cfg.rho * l_route)
        total = sum(coop.values()) + adv
        disc_micro_batches.append((
            out["z_L"].detach(), out["z_P"].detach(), olen.detach(),
            spk.detach(), tgt.detach(), tlen.detach(), p_f0.detach(), p_v.detach(),
            p_e.detach(), emo.detach(),
        ))

        if micro_index == 1:
            optimizer.zero_grad(set_to_none=True)
        router_grad_diag = None
        adversary_losses = None
        active_freq_for_router = None
        active_p_soft_after = None
        if log_now:
            adversary_losses = {
                "grl": cfg.grl_weight * l_grl,
                "grl_p": cfg.grl_phoneme_weight * l_grl_p,
                "pros_grl": cfg.grl_prosody_weight * l_prosgrl,
                "emo_grl": cfg.grl_emotion_weight * l_emogrl,
            }
            if cfg.routing_spec_weight > 0:
                adversary_losses["route_spec"] = cfg.routing_spec_weight * l_spec
            if cfg.rho > 0:
                adversary_losses["route_balance"] = cfg.rho * l_route

            with torch.no_grad():
                _z_active = (out["z_t"] != 0).float()
                _T = _z_active.shape[1]
                _valid = (torch.arange(_T, device=device).unsqueeze(0)
                          < olen.unsqueeze(1)).float()
                if out["routing_logits"].dim() == 2:
                    active_freq_for_router = (
                        (_z_active * _valid.unsqueeze(-1)).sum(dim=(0, 1))
                        / _valid.sum().clamp(min=1)
                    )
                else:
                    active_freq_for_router = (
                        (_z_active * _valid.unsqueeze(-1)).sum(dim=1)
                        / _valid.sum(dim=1, keepdim=True).clamp(min=1)
                    )
            route_p_soft = torch.softmax(out["routing_logits"], dim=-1)[..., 1]
            active_p_soft = (
                (active_freq_for_router * route_p_soft).sum()
                / active_freq_for_router.sum().clamp(min=1e-12)
            )
            router_grad_diag = named_gradient_diagnostics(
                {**coop, **adversary_losses}, rout_params, reference=active_p_soft)
        if pcgrad is not None:
            managed = {k: v for k, v in coop.items() if k in coop_names}
            unmanaged = [v for k, v in coop.items() if k not in coop_names]
            shared_extra = adv + sum(unmanaged) if unmanaged else adv
            current = {name: pcgrad.flat_grad(loss).detach() / accumulation
                       for name, loss in managed.items()}
            if task_grad_sums is None:
                task_grad_sums = {name: grad.clone() for name, grad in current.items()}
                extra_grad_sum = pcgrad.flat_grad(shared_extra).detach() / accumulation
            else:
                for name, grad in current.items(): task_grad_sums[name].add_(grad)
                extra_grad_sum.add_(pcgrad.flat_grad(shared_extra).detach() / accumulation)
            scaled_total = total / accumulation
            if scaler.is_enabled(): scaler.scale(scaled_total).backward()
            else: scaled_total.backward()
        else:
            scaled_total = total / accumulation
            if scaler.is_enabled(): scaler.scale(scaled_total).backward()
            else: scaled_total.backward()
        if not boundary:
            continue
        step = next_step
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if pcgrad is not None:
            pcgrad.write_(pcgrad.project_vectors(task_grad_sums) + extra_grad_sum)
        main_pre_clip = float(nn.utils.clip_grad_norm_(all_params, cfg.grad_clip))
        main_clip_scale = min(1.0, cfg.grad_clip / max(main_pre_clip, 1e-12))
        if scaler.is_enabled():
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        if log_now and active_freq_for_router is not None and not model.routing.dynamic:
            with torch.no_grad():
                route_p_after = torch.softmax(model.routing.logits, dim=-1)[:, 1]
                active_p_soft_after = float(
                    (active_freq_for_router * route_p_after).sum()
                    / active_freq_for_router.sum().clamp(min=1e-12)
                )

        # ---- discriminator catch-up: track the moving encoder (no reversal) ----
        disc_pre_clip = 0.0
        disc_clip_scale = 1.0
        if cfg.n_disc_steps > 1:
            for _ in range(cfg.n_disc_steps - 1):
                optimizer.zero_grad(set_to_none=True)
                for (zL_d, zP_d, lens_d, spk_d, tgt_d, tlen_d,
                     pf0_d, pv_d, pe_d, emo_d) in disc_micro_batches:
                    with _autocast(amp_dtype):
                        ld = (U.speaker_adv_loss(model.grl_head(zL_d, lens_d, 0.0), spk_d, lens_d)
                              + ctc_pr_loss(model.pr_grl_head(zP_d, 0.0), tgt_d, lens_d, tlen_d)
                              + U.prosody_train_loss(model.prosody_grl_head(zL_d, 0.0), pf0_d, pv_d, pe_d, lens_d)
                              + F.cross_entropy(model.emotion_grl_head(zL_d, lens_d, 0.0), emo_d, weight=emo_w))
                    if scaler.is_enabled(): scaler.scale(ld / accumulation).backward()
                    else: (ld / accumulation).backward()
                if scaler.is_enabled(): scaler.unscale_(optimizer)
                disc_norm = float(nn.utils.clip_grad_norm_(disc_params, cfg.grad_clip))
                if disc_norm > disc_pre_clip:
                    disc_pre_clip = disc_norm
                    disc_clip_scale = min(1.0, cfg.grad_clip / max(disc_norm, 1e-12))
                if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
                else: optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        micro_index = 0
        task_grad_sums = None
        extra_grad_sum = None
        disc_micro_batches = []

        # ---- logging ----
        if log_now:
            with torch.no_grad():
                sc, st = U.speaker_correct(out["sid_logits"], spk, olen)
                gc, gt = U.speaker_correct(out["grl_logits"], spk, olen)
                ec, et = U.class_correct(out["emotion_logits"], emo)
                lc, lt = U.class_correct(out["emotion_grl_logits"], emo)
                pr_n, pr_d = U.ctc_errors(out["pr_logits"].float(), tgt, olen, tlen)
                gp_n, gp_d = U.ctc_errors(out["pr_grl_logits"].float(), tgt, olen, tlen)

                route_idx = out["routing_logits"].detach().argmax(dim=-1)
                if route_idx.dim() == 1:
                    n_L = int((route_idx == 0).sum())
                    n_P = int((route_idx == 1).sum())
                    route_L = (route_idx == 0).float().view(1, 1, -1)
                    route_P = (route_idx == 1).float().view(1, 1, -1)
                else:
                    n_L = int((route_idx == 0).sum(dim=-1).float().mean())
                    n_P = int((route_idx == 1).sum(dim=-1).float().mean())
                    route_L = (route_idx == 0).float().unsqueeze(1)
                    route_P = (route_idx == 1).float().unsqueeze(1)
                z_active = (out["z_t"] != 0).float()
                T = z_active.shape[1]
                fmask = (torch.arange(T, device=device).unsqueeze(0)
                         < olen.unsqueeze(1)).float()
                n_valid = fmask.sum().clamp(min=1)
                act_L = float(((z_active * route_L).sum(-1) * fmask).sum() / n_valid)
                act_P = float(((z_active * route_P).sum(-1) * fmask).sum() / n_valid)
                route_diag = model.routing.routing_diagnostics
                n_dead = int((model.sae.steps_since_fired > model.sae.dead_threshold).sum())
                dead_frac = n_dead / cfg.K
                zL_grl_frame = (cfg.grl_grad_norm_target * ramp
                                if cfg.grl_grad_norm else float("nan"))

            print(f"\n[train step={step:05d}/{cfg.stage2_steps}]  lr_sae={lr_now:.2e}  "
                  f"grl_lambda={ramp:.2f}  zL_grl_frame={zL_grl_frame:.2e}", flush=True)
            print("  factor      kept task                         adversary", flush=True)
            print(f"  phoneme     loss={l_pr.item():7.3f} PER={pr_n/max(pr_d,1):.3f}"
                  f"        loss={l_grl_p.item():7.3f} PER={gp_n/max(gp_d,1):.3f}", flush=True)
            print(f"  speaker     loss={l_sid.item():7.3f} acc={sc/max(st,1):.3f}"
                  f"        loss={l_grl.item():7.3f} acc={gc/max(gt,1):.3f}", flush=True)
            print(f"  prosody     loss={l_pros.item():7.3f}"
                  f"                  loss={l_prosgrl.item():7.3f}", flush=True)
            print(f"  emotion     loss={l_emo.item():7.3f} acc={ec/max(et,1):.3f}"
                  f"        loss={l_emogrl.item():7.3f} acc={lc/max(lt,1):.3f}", flush=True)
            print(f"  shared      recon={l_recon.item():.4f}  invariance={float(l_inv):.4f}", flush=True)

            active_total = max(act_L + act_P, 1e-12)
            print("  routing", flush=True)
            print(f"    assigned L/P={n_L}/{n_P}"
                  f"    active L/P={act_L:.0f}/{act_P:.0f}"
                  f"    active_P={100*act_P/active_total:.1f}%    dead={100*dead_frac:.1f}%", flush=True)
            print(f"    H_balance={route_diag['balance_entropy']:.3f}"
                  f"    H_unit={route_diag['unit_entropy']:.3f}"
                  f"    specialized={route_diag['specialized_frac_h_lt_0_5']:.2f}"
                  f"    margin={route_diag['top1_top2_margin']:.3f}"
                  f"    logit_std={route_diag['logit_std']:.4f}"
                  f"    tau={model.routing.tau:.3f}", flush=True)
            print("  clipping (pre-norm -> multiplier)", flush=True)
            print(f"    main={main_pre_clip:.3e} -> {main_clip_scale:.3e}"
                  f"    discriminator_max={disc_pre_clip:.3e} -> {disc_clip_scale:.3e}", flush=True)
            append_metrics(metrics_path, {
                "step": step, "split": "train", "lr": lr_now,
                "recon": float(l_recon), "pr": float(l_pr), "sid": float(l_sid),
                "grl": float(l_grl), "grl_p": float(l_grl_p),
                "prosody": float(l_pros), "prosody_grl": float(l_prosgrl),
                "emotion": float(l_emo), "emotion_grl": float(l_emogrl),
                "invariance": float(l_inv), "route_tau": float(model.routing.tau),
                "active_L": act_L, "active_P": act_P, "dead_fraction": dead_frac,
            })

            def _norm_row(norms, names):
                return "  ".join(f"{name}={norms.get(name, 0.0):.2e}" for name in names)

            coop_order = ("recon", "pr", "sid", "prosody", "emotion", "inv")
            adv_order = ("grl", "grl_p", "pros_grl", "emo_grl", "route_spec")
            if grad_diag is not None:
                coop_cos = grad_diag["coop_cosines"]
                if coop_cos:
                    sae_min_pair, sae_min_cos = min(coop_cos.items(), key=lambda item: item[1])
                else:
                    sae_min_pair, sae_min_cos = "n/a", float("nan")
                print("  SAE gradients (weighted, before clipping)", flush=True)
                print(f"    cooperative: {_norm_row(grad_diag['norms'], coop_order)}", flush=True)
                print(f"    adversarial: {_norm_row(grad_diag['norms'], adv_order)}", flush=True)
                print(f"    PCGrad: raw={grad_diag['raw_coop_norm']:.2e}"
                      f"  projected={grad_diag['projected_coop_norm']:.2e}"
                      f"  adversary_bundle={grad_diag['external_norm']:.2e}", flush=True)
                print(f"    cooperative conflicts={grad_diag['coop_conflicts']}/{len(coop_cos)}"
                      f"  strongest={sae_min_pair}:{sae_min_cos:+.2f}", flush=True)

            if router_grad_diag is not None:
                def _grad_cos(a, b):
                    av = router_grad_diag["vectors"][a].detach().float()
                    bv = router_grad_diag["vectors"][b].detach().float()
                    return float(torch.dot(av, bv) / (av.norm() * bv.norm()).clamp(min=1e-12))

                router_vectors = router_grad_diag["vectors"]
                router_adv_names = [name for name in adv_order if name in router_vectors]
                router_adv = torch.stack([router_vectors[name] for name in router_adv_names]).sum(0)

                def _cos_adv(name):
                    v = router_vectors[name].detach().float()
                    a = router_adv.detach().float()
                    return float(torch.dot(v, a) / (v.norm() * a.norm()).clamp(min=1e-12))

                router_cos = router_grad_diag["cosines"]
                if router_cos:
                    router_min_pair, router_min_cos = min(router_cos.items(), key=lambda item: item[1])
                else:
                    router_min_pair, router_min_cos = "n/a", float("nan")
                paired = (
                    f"pr<->grl_p={_grad_cos('pr', 'grl_p'):+.2f}  "
                    f"sid<->grl={_grad_cos('sid', 'grl'):+.2f}  "
                    f"pros<->pros_grl={_grad_cos('prosody', 'pros_grl'):+.2f}  "
                    f"emo<->emo_grl={_grad_cos('emotion', 'emo_grl'):+.2f}"
                )
                coop_vs_adv = "  ".join(
                    f"{name}={_cos_adv(name):+.2f}" for name in coop_order)
                push_cos = router_grad_diag.get("push_cos", {})
                coop_push = "  ".join(
                    f"{name}={push_cos.get(name, 0.0):+.2f}" for name in coop_order)
                adv_push = "  ".join(
                    f"{name}={push_cos.get(name, 0.0):+.2f}" for name in adv_order)
                print("  router gradients (weighted, before clipping)", flush=True)
                print(f"    cooperative: {_norm_row(router_grad_diag['norms'], coop_order)}", flush=True)
                print(f"    adversarial: {_norm_row(router_grad_diag['norms'], adv_order)}", flush=True)
                print(f"    paired cosines: {paired}", flush=True)
                print(f"    cooperative vs adversary bundle: {coop_vs_adv}", flush=True)
                print("    active-P push cosine (+ toward P, - toward L)", flush=True)
                print(f"      cooperative: {coop_push}", flush=True)
                print(f"      adversarial: {adv_push}", flush=True)
                print(f"      combined={router_grad_diag.get('total_push_cos', 0.0):+.2f}"
                      f"  first_order={router_grad_diag.get('total_push_effect', 0.0):+.2e}", flush=True)
                if active_p_soft_after is not None:
                    before = router_grad_diag.get("reference_value", float("nan"))
                    print(f"      actual optimizer step: soft_active_P {before:.6f} -> "
                          f"{active_p_soft_after:.6f}  delta={active_p_soft_after-before:+.2e}", flush=True)
                print(f"    all-pair negative={sum(c < 0 for c in router_cos.values())}/{len(router_cos)}"
                      f"  strongest={router_min_pair}:{router_min_cos:+.2f}", flush=True)

        # ---- eval + checkpoint ----
        if step % cfg.ckpt_every == 0 or step == cfg.stage2_steps:
            vm = evaluate(model, val_dl, device, amp_dtype, n_emotion)
            score = (vm["per"] + (1 - vm["sid_acc"]) + vm["zL_sid_acc"]
                     + (1 - vm["zP_pr_per"]) + (1 - vm["zP_emo_uar"]) + vm["zL_emo_uar"])
            print(f"\n[val step={step:05d}]", flush=True)
            print("  factor      kept representation       leakage representation", flush=True)
            print(f"  phoneme     z_L PER={vm['per']:.3f}"
                  f"              z_P PER={vm['zP_pr_per']:.3f}", flush=True)
            print(f"  speaker     z_P acc={vm['sid_acc']:.3f}"
                  f"              z_L acc={vm['zL_sid_acc']:.3f}", flush=True)
            print(f"  emotion     z_P UAR={vm['zP_emo_uar']:.3f} acc={vm['zP_emo_acc']:.3f}"
                  f"    z_L UAR={vm['zL_emo_uar']:.3f}", flush=True)
            print(f"  checkpoint score={score:.3f}", flush=True)
            append_metrics(metrics_path, {"step": step, "split": "val", **vm, "score": score})
            save_runtime(f"step{step}.pt", kind="inference", val=vm)
            if score < best:
                best = score
                save_runtime("best.pt", kind="inference", val=vm)
                print(f"  [best] disent={best:.3f} -> {ckpt_dir/'best.pt'}", flush=True)

        resume_every = int(getattr(cfg, "resume_every", 0))
        if resume_every > 0 and step % resume_every == 0:
            save_runtime("latest-resume.pt")
        if segment.reached(step):
            save_runtime("latest-resume.pt")
            print(f"[msp] segment complete at step {step}; resume from latest-resume.pt")
            return ckpt_dir / "latest-resume.pt"

    save_runtime("latest-resume.pt")
    save_runtime("final.pt", kind="inference")
    print(f"[msp] done. best disent={best:.3f}  ckpts in {ckpt_dir}")
    return ckpt_dir / "best.pt"
