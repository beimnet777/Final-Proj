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
from .heads import GELUEmotionGRLHead, GELUSpeakerGRLHead
from .checkpoints import load_sae_initialization
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


def _loss_if_present(out: dict, key: str, reference: torch.Tensor, fn):
    """Return a task loss only when its optional model output exists.

    Zero-weight baselines and adversary ablations intentionally suppress some
    heads.  The training loop must therefore treat a missing optional output as a
    zero loss instead of crashing before the first step.
    """
    if key not in out:
        return reference.new_zeros(())
    return fn(out[key])


def _set_requires_grad(params, enabled: bool) -> None:
    """Toggle a parameter group without changing optimizer membership."""
    for param in params:
        param.requires_grad_(enabled)


def _clip_group(params, max_norm: float) -> tuple[float, float]:
    """Clip one logical parameter group and return (pre-norm, scale)."""
    active = [param for param in params if param.grad is not None]
    if not active:
        return 0.0, 1.0
    pre_norm = float(nn.utils.clip_grad_norm_(active, max_norm))
    return pre_norm, min(1.0, max_norm / max(pre_norm, 1e-12))


def _gumbel_tau(step: int, total: int, tau_start: float, tau_end: float) -> float:
    """Geometric anneal of the Gumbel temperature (matches legacy)."""
    return tau_start * (tau_end / tau_start) ** (step / max(1, total))


# ---------------------------------------------------------------- schedule
def _dann_lambda(step: int, total: int, ramp_steps: int = 0) -> float:
    """Canonical DANN sigmoid ramp: 0 at start → 1 by ramp_steps/end.

    If ramp_steps > 0, lambda reaches 1.0 at that step and remains saturated.
    If ramp_steps == 0, it ramps over the whole training schedule.
    """
    ramp_steps = int(ramp_steps)
    if ramp_steps > 0:
        if step >= ramp_steps:
            return 1.0
        denom = max(1, ramp_steps)
    else:
        denom = max(1, total)
    p = max(0.0, min(1.0, step / denom))
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


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


def _calibrate_route_topk_quotas(model, train_dl, device, cfg, amp_dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate route-local active budgets after freezing learned routing.

    We do not impose a preset L/P active split.  Instead we measure, under the
    learned route assignment and the ordinary global TopK, how many selected
    units land in each route.  Those averages are rounded to integer quotas that
    sum to ``cfg.topk`` and then stored in the SAE, stabilizing the post-freeze
    active split.
    """
    if getattr(model.routing, "dynamic", False):
        raise ValueError("route-local TopK calibration supports static learned routing only")

    n_routes = int(getattr(cfg, "n_routes", 2))
    max_batches = max(1, int(getattr(cfg, "route_topk_calib_batches", 20)))
    was_training = model.training
    route_topk_was_enabled = bool(getattr(model.sae, "route_topk_enabled").item())
    if route_topk_was_enabled:
        model.sae.clear_route_topk()
    model.eval()

    sampler_state = None
    if hasattr(train_dl, "sampler") and hasattr(train_dl.sampler, "state_dict"):
        sampler_state = train_dl.sampler.state_dict()

    route_idx = model.routing.logits.detach().argmax(dim=-1).to(device)
    route_counts = torch.stack([(route_idx == r).sum() for r in range(n_routes)]).long()
    active_counts = torch.zeros(n_routes, device=device)
    frame_count = torch.zeros((), device=device)
    batches_seen = 0
    data_iter = iter(train_dl)
    for _ in range(max_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)
        audios = batch["audios"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        with _autocast(amp_dtype):
            out = model(audios, audio_lengths, stage=2, grl_lambda=0.0,
                        grl_p_lambda=0.0, grl_prosody_lambda=0.0,
                        grl_emotion_lambda=0.0, emit_emotion=False)
        z_active = (out["z_t"] != 0).float()
        _, T_ = z_active.shape[:2]
        fmask = (torch.arange(T_, device=device).unsqueeze(0)
                 < out["out_lengths"].unsqueeze(1)).float()
        valid_frames = fmask.sum()
        if valid_frames.item() <= 0:
            continue
        for r in range(n_routes):
            active_counts[r] += (
                (z_active * (route_idx == r).float()).sum(-1) * fmask
            ).sum()
        frame_count += valid_frames
        batches_seen += 1

    if sampler_state is not None and hasattr(train_dl.sampler, "load_state_dict"):
        train_dl.sampler.load_state_dict(sampler_state)
    model.train(was_training)

    if frame_count.item() <= 0 or batches_seen == 0:
        raise ValueError("route-local TopK calibration saw no valid frames")

    avg = active_counts / frame_count
    quotas = torch.floor(avg).long().cpu()
    frac = avg.cpu() - quotas.float()
    target_topk = int(getattr(cfg, "topk", int(avg.sum().round().item())))
    route_counts_cpu = route_counts.cpu()

    diff = target_topk - int(quotas.sum().item())
    if diff > 0:
        order = torch.argsort(frac, descending=True).tolist()
        while diff > 0:
            changed = False
            for idx in order:
                if diff <= 0:
                    break
                if quotas[idx].item() < route_counts_cpu[idx].item():
                    quotas[idx] += 1
                    diff -= 1
                    changed = True
            if not changed:
                raise ValueError(
                    "cannot allocate route-local TopK quotas to requested topk; "
                    f"quotas={quotas.tolist()} route_counts={route_counts_cpu.tolist()} "
                    f"target={target_topk}")
    elif diff < 0:
        order = torch.argsort(frac, descending=False).tolist()
        remaining = -diff
        while remaining > 0:
            changed = False
            for idx in order:
                if remaining <= 0:
                    break
                if quotas[idx].item() > 0:
                    quotas[idx] -= 1
                    remaining -= 1
                    changed = True
            if not changed:
                raise ValueError(f"cannot reduce route-local TopK quotas to target {target_topk}")

    model.sae.set_route_topk(route_idx.cpu(), quotas)
    print("[msp] learned-route TopK calibrated: "
          f"batches={batches_seen} avg_active={[round(float(x), 2) for x in avg.cpu()]} "
          f"quotas={quotas.tolist()} route_counts={route_counts_cpu.tolist()} "
          f"sum={int(quotas.sum().item())}", flush=True)
    return route_idx.cpu(), quotas


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
    saw_emotion_grl = False
    pros_sum = prosgrl_sum = pros_n = prosgrl_n = 0
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
        if "prosody_pred" in out:
            p_f0, p_v, p_e = U.prosody_targets_fast(audios, alen, olen)
            batch_n = int(audios.shape[0])
            pros_sum += float(U.prosody_train_loss(
                out["prosody_pred"], p_f0, p_v, p_e, olen)) * batch_n
            pros_n += batch_n
            if "prosody_grl_pred" in out:
                prosgrl_sum += float(U.prosody_train_loss(
                    out["prosody_grl_pred"], p_f0, p_v, p_e, olen)) * batch_n
                prosgrl_n += batch_n
        ep = out["emotion_logits"].argmax(-1)
        for tlab, plab in zip(emo.tolist(), ep.tolist()):
            conf_P[tlab, plab] += 1
        if "emotion_grl_logits" in out:
            saw_emotion_grl = True
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
        "zL_emo_uar": (U.uar_from_confusion(conf_L) if saw_emotion_grl else float("nan")),
        "zP_prosody_loss": pros_sum / pros_n if pros_n else float("nan"),
        "zL_prosody_loss": prosgrl_sum / prosgrl_n if prosgrl_n else float("nan"),
    }
    return m


@torch.no_grad()
def evaluate_recon(model, dl, device, amp_dtype) -> Dict[str, float]:
    model.eval()
    recon_sum = 0.0
    n_batches = 0
    for b in dl:
        audios = b["audios"].to(device)
        alen = b["audio_lengths"].to(device)
        with _autocast(amp_dtype):
            out = model(audios, alen, stage=1)
            loss = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
        recon_sum += float(loss)
        n_batches += 1
    model.train()
    return {"recon": recon_sum / max(n_batches, 1)}


# ---------------------------------------------------------------- train
def run(cfg, stage1_ckpt: Optional[str] = None) -> Path:
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)
    amp_dtype = _amp_dtype(str(getattr(cfg, "precision", "auto")))

    loaders = make_msp_dataloaders(cfg)
    tokenizer, train_dl, val_dl, test_dl = loaders[0], loaders[1], loaders[2], loaders[3]
    test_unseen_dl = loaders[4] if len(loaders) > 4 else None
    n_emotion = cfg.emotion_num_classes

    model = build_dis_model(cfg).to(device)
    if stage1_ckpt:
        audit = load_sae_initialization(model, stage1_ckpt)
        print(f"[msp] loaded {len(audit['loaded'])} SAE tensors from {stage1_ckpt} "
              f"(format={audit['source_format']} step={audit['source_step']} "
              f"shape_mismatch={len(audit['mismatched'])})")
    else:
        print("[msp] training from scratch (SAE/routing/heads from init)")
    # Speaker adversary → GELU (isolation: overrides model.grl_head without
    # editing the shared model/heads.py).
    model.grl_head = GELUSpeakerGRLHead(cfg).to(device)
    grl_norm = (f"per-frame grad norm target={cfg.grl_grad_norm_target:g}"
                if cfg.grl_grad_norm else "plain gradient reversal")
    print(f"[msp] speaker adversary: GELUSpeakerGRLHead (GELU projector, {grl_norm})")
    if hasattr(model, "emotion_grl_head"):
        model.emotion_grl_head = GELUEmotionGRLHead(cfg).to(device)
        emo_grl_norm = (
            f"per-frame grad norm target={cfg.grl_emotion_grad_norm_target:g}"
            if getattr(cfg, "grl_emotion_grad_norm", False)
            else "plain gradient reversal"
        )
        print(f"[msp] emotion adversary: GELUEmotionGRLHead (mean+std, {emo_grl_norm})")
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
    disc_params = list(model.grl_head.parameters()) + list(model.pr_grl_head.parameters())
    for optional_head in ("prosody_grl_head", "emotion_grl_head"):
        head = getattr(model, optional_head, None)
        if head is not None:
            disc_params += list(head.parameters())
    separate_disc = bool(getattr(cfg, "separate_discriminator_optimizer", False))
    separate_clip = bool(getattr(cfg, "separate_grad_clip", False))
    main_groups = [
        {"params": sae_params,  "lr": cfg.lr},
        {"params": rout_params, "lr": cfg.lr_routing},
        {"params": head_params, "lr": cfg.lr_heads},
    ]
    if not separate_disc:
        main_groups.append({"params": disc_params, "lr": cfg.lr_disc})
    optimizer = torch.optim.AdamW(
        main_groups, betas=(0.9, 0.95), weight_decay=0.0)
    disc_optimizer = (
        torch.optim.AdamW(disc_params, lr=cfg.lr_disc,
                          betas=(0.9, 0.95), weight_decay=0.0)
        if separate_disc else None
    )
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
    print(f"[msp] optimization: discriminator_optimizer={'separate' if separate_disc else 'shared'} "
          f"gradient_clip={'per-group' if separate_clip else 'global'} "
          f"pcgrad_balance={getattr(cfg, 'pcgrad_balance', 'none')}")
    print(f"[msp] weights recon={getattr(cfg, 'recon_weight', 1.0)} "
          f"α(pr)={cfg.alpha} β(sid)={cfg.beta} grl={cfg.grl_weight} "
          f"grl_p={cfg.grl_phoneme_weight} pros={cfg.prosody_weight}/{cfg.grl_prosody_weight} "
          f"emo={cfg.emotion_weight}/{cfg.grl_emotion_weight} inv={cfg.inv_weight}")
    spk_frame = (float(cfg.grl_grad_norm_target)
                 if getattr(cfg, "grl_grad_norm", False) else float("nan"))
    emo_frame = (float(getattr(cfg, "grl_emotion_grad_norm_target", float("nan")))
                 if getattr(cfg, "grl_emotion_grad_norm", False) else float("nan"))
    print(f"[msp] per-frame GRL targets: speaker={spk_frame:.2e} emotion={emo_frame:.2e}")
    if model.sae.aux_k > 0:
        print(f"[msp] AuxK on: aux_k={model.sae.aux_k} coef={cfg.aux_k_coef:g} "
              f"dead_thresh={model.sae.dead_threshold} "
              f"valid_frames={getattr(cfg, 'valid_frame_dead_count', False)}")
    pure_recon_only = (
        float(getattr(cfg, "recon_weight", 1.0)) > 0.0
        and cfg.alpha == 0.0 and cfg.beta == 0.0
        and cfg.grl_weight == 0.0 and cfg.grl_phoneme_weight == 0.0
        and cfg.prosody_weight == 0.0 and cfg.grl_prosody_weight == 0.0
        and cfg.emotion_weight == 0.0 and cfg.grl_emotion_weight == 0.0
        and cfg.inv_weight == 0.0
        and getattr(cfg, "routing_spec_weight", 0.0) == 0.0
        and getattr(cfg, "rho", 0.0) == 0.0
    )
    if pure_recon_only:
        print("[msp] pure reconstruction mode: using stage=1 forward "
              "(no routing/task/adversary heads during training)", flush=True)

    best = float("inf")
    step = 0
    resume_value = str(getattr(cfg, "resume", "none"))
    resume_path = ckpt_dir / "latest-resume.pt" if resume_value == "auto" else Path(resume_value)
    restored_from_resume = False
    if resume_value not in {"none", ""} and resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        validate_resume(saved, dataset_hash=str(getattr(cfg, "dataset_fingerprint", "")),
                        preset=str(getattr(cfg, "experiment_preset", "")), cfg=cfg)
        step, best = restore_training_state(saved, model=model, optimizer=optimizer, scaler=scaler)
        if disc_optimizer is not None:
            disc_state = saved.get("auxiliary", {}).get("disc_optimizer")
            if disc_state is None:
                raise ValueError(
                    "resume checkpoint lacks the separate discriminator optimizer state")
            disc_optimizer.load_state_dict(disc_state)
        restored_from_resume = True
        if pcgrad is not None:
            pcgrad.load_state_dict(saved.get("auxiliary", {}).get("pcgrad", {}))
        sampler_state = saved.get("auxiliary", {}).get("train_sampler")
        if sampler_state and hasattr(train_dl.sampler, "load_state_dict"):
            train_dl.sampler.load_state_dict(sampler_state)
        print(f"[msp] exact resume from {resume_path}: step={step} best={best:.4f}")
    elif resume_value not in {"none", "", "auto"}:
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

    if bool(getattr(cfg, "freeze_learned_routing_on_resume", False)):
        if not restored_from_resume:
            raise ValueError("freeze_learned_routing_on_resume requires a restored resume checkpoint")
        if getattr(model.routing, "dynamic", False):
            raise ValueError("learned-route freeze currently supports static routing only")
        model.routing.freeze_learned_routing()
        n_L, n_P, n_U = model.routing.hard_counts
        print("[msp] learned routing FROZEN after resume: "
              f"counts L/P/U={n_L}/{n_P}/{n_U}", flush=True)
        if bool(getattr(cfg, "freeze_route_topk_on_resume", False)):
            if bool(getattr(model.sae, "route_topk_enabled").item()):
                quotas = getattr(model.sae, "route_topk_quotas").detach().cpu().tolist()
                print(f"[msp] learned-route TopK already enabled from checkpoint: quotas={quotas}",
                      flush=True)
            else:
                _calibrate_route_topk_quotas(model, train_dl, device, cfg, amp_dtype)

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
                       "disc_optimizer": (disc_optimizer.state_dict()
                                          if kind == "resume" and disc_optimizer is not None
                                          else None),
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
        grad_every = max(1, int(getattr(cfg, "grad_log_every", cfg.log_every)))
        grad_now = boundary and (next_step % grad_every == 0 or next_step == 1)
        grad_diag = None
        raw_coop_grad_norms = None
        pcgrad_balance_scales = None
        adv_sae_grad_norms = None
        lr_now = lr_at(next_step)
        optimizer.param_groups[0]["lr"] = lr_now          # SAE follows the cosine
        ramp = _dann_lambda(next_step, cfg.stage2_steps,
                            int(getattr(cfg, "dann_ramp_steps", cfg.warmup_steps)))
        model.routing.tau = _gumbel_tau(next_step, cfg.stage2_steps,
                                        cfg.gumbel_tau_start, cfg.gumbel_tau_end)

        audios = b["audios"].to(device); alen = b["audio_lengths"].to(device)
        if pure_recon_only:
            with _autocast(amp_dtype):
                out = model(audios, alen, stage=1)
                olen = out["out_lengths"]
                model.sae.update_dead(
                    out["z_t"],
                    olen if getattr(cfg, "valid_frame_dead_count", False) else None,
                )
                l_recon = recon_loss(out["h_t"], out["h_hat"], olen)
                l_aux = l_recon.new_zeros(())
                if model.sae.aux_k > 0:
                    e_hat = model.sae.aux_reconstruct(out["z_pre"])
                    if e_hat is not None:
                        residual = (out["h_t"] - out["h_hat"]).detach()
                        l_aux = recon_loss(residual, e_hat, olen)
                total = (float(getattr(cfg, "recon_weight", 1.0)) * l_recon
                         + float(getattr(cfg, "aux_k_coef", 0.0)) * l_aux)
            if micro_index == 1:
                optimizer.zero_grad(set_to_none=True)
            scaled_total = total / accumulation
            if scaler.is_enabled(): scaler.scale(scaled_total).backward()
            else: scaled_total.backward()
            if not boundary:
                continue
            step = next_step
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            main_pre_clip = float(nn.utils.clip_grad_norm_(sae_params, cfg.grad_clip))
            main_clip_scale = min(1.0, cfg.grad_clip / max(main_pre_clip, 1e-12))
            if scaler.is_enabled():
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_index = 0
            task_grad_sums = None
            extra_grad_sum = None
            disc_micro_batches = []
            if log_now or grad_now:
                with torch.no_grad():
                    z_active = (out["z_t"] != 0).float()
                    T = z_active.shape[1]
                    fmask = (torch.arange(T, device=device).unsqueeze(0)
                             < olen.unsqueeze(1)).float()
                    n_valid = fmask.sum().clamp(min=1)
                    active = float((z_active.sum(-1) * fmask).sum() / n_valid)
                    n_dead = int((model.sae.steps_since_fired > model.sae.dead_threshold).sum())
                    dead_frac = n_dead / cfg.K
                print(f"\n[train {step:05d}/{cfg.stage2_steps}] "
                      f"lr={lr_now:.2e} recon={l_recon.item():.4f} "
                      f"aux={l_aux.item():.4f} "
                      f"dead={100*dead_frac:.1f}% active={active:.0f} "
                      f"clip main={main_pre_clip:.2e}->{main_clip_scale:.2e}",
                      flush=True)
                if log_now:
                    append_metrics(metrics_path, {
                        "step": step, "split": "train", "lr": lr_now,
                        "recon": float(l_recon), "active": active,
                        "aux": float(l_aux),
                        "dead_fraction": dead_frac,
                    })

            if step % cfg.ckpt_every == 0 or step == cfg.stage2_steps:
                vm = evaluate_recon(model, val_dl, device, amp_dtype)
                score = vm["recon"]
                print(f"\n[val step={step:05d}] recon={vm['recon']:.4f}", flush=True)
                append_metrics(metrics_path, {"step": step, "split": "val", **vm, "score": score})
                save_runtime(f"step{step}.pt", kind="inference", val=vm)
                if score < best:
                    best = score
                    save_runtime("best.pt", kind="inference", val=vm)
                    print(f"  [best] recon={best:.4f} -> {ckpt_dir/'best.pt'}", flush=True)

            resume_every = int(getattr(cfg, "resume_every", 0))
            if resume_every > 0 and step % resume_every == 0:
                save_runtime("latest-resume.pt")
            if segment.reached(step):
                save_runtime("latest-resume.pt")
                print(f"[msp] segment complete at step {step}; resume from latest-resume.pt")
                return ckpt_dir / "latest-resume.pt"
            continue

        tgt = b["targets"].to(device); tlen = b["target_lengths"].to(device)
        spk = b["speaker_ids"].to(device); emo = b["emotion"].to(device)
        pert = b.get("pert_audios")
        if pert is not None:
            pert = pert.to(device)

        # With a separate optimizer the discriminator acts as a fixed function
        # during the representation update: gradients still flow through it into
        # z_L/z_P, but its parameters are updated only on detached features below.
        if separate_disc:
            _set_requires_grad(disc_params, False)
        with _autocast(amp_dtype):
            out = model(audios, alen, stage=2,
                        grl_lambda=ramp, grl_p_lambda=ramp,
                        grl_prosody_lambda=ramp, grl_emotion_lambda=ramp,
                        emit_emotion=True)
            olen = out["out_lengths"]
            model.sae.update_dead(
                out["z_t"],
                olen if getattr(cfg, "valid_frame_dead_count", False) else None,
            )
            # cooperative tasks
            l_recon = recon_loss(out["h_t"], out["h_hat"], olen)
            l_aux = l_recon.new_zeros(())
            if model.sae.aux_k > 0:
                e_hat = model.sae.aux_reconstruct(out["z_pre"])
                if e_hat is not None:
                    residual = (out["h_t"] - out["h_hat"]).detach()
                    l_aux = recon_loss(residual, e_hat, olen)
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
            l_grl = _loss_if_present(
                out, "grl_logits", l_recon,
                lambda logits: U.speaker_adv_loss(logits, spk, olen))
            l_grl_p = _loss_if_present(
                out, "pr_grl_logits", l_recon,
                lambda logits: ctc_pr_loss(logits, tgt, olen, tlen))
            l_prosgrl = _loss_if_present(
                out, "prosody_grl_pred", l_recon,
                lambda pred: U.prosody_train_loss(pred, p_f0, p_v, p_e, olen))
            l_emogrl = _loss_if_present(
                out, "emotion_grl_logits", l_recon,
                lambda logits: F.cross_entropy(logits, emo, weight=emo_w))
            l_route   = route_loss(out["routing_logits"])
            l_spec    = routing_spec_loss(out["routing_logits"])

        coop_raw = {
            "recon": l_recon,
            "pr": l_pr,
            "sid": l_sid,
            "prosody": l_pros,
            "emotion": l_emo,
            "inv": l_inv,
            "aux": l_aux,
        }
        coop_weights = {
            "recon": float(getattr(cfg, "recon_weight", 1.0)),
            "pr": float(cfg.alpha),
            "sid": float(cfg.beta),
            "prosody": float(cfg.prosody_weight),
            "emotion": float(cfg.emotion_weight),
            "inv": float(cfg.inv_weight),
            "aux": float(getattr(cfg, "aux_k_coef", 0.0)),
        }
        coop = {name: coop_weights[name] * loss for name, loss in coop_raw.items()}
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
        if grad_now:
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
            balance_mode = str(getattr(cfg, "pcgrad_balance", "none"))
            managed_source = coop_raw if balance_mode == "unit" else coop
            managed = {k: v for k, v in managed_source.items() if k in coop_names}
            unmanaged = [v for k, v in coop.items() if k not in coop_names]
            shared_extra = adv + sum(unmanaged) if unmanaged else adv
            current = {name: pcgrad.flat_grad(loss).detach() / accumulation
                       for name, loss in managed.items()}
            if grad_now and adversary_losses is not None:
                adv_sae_grad_norms = {
                    name: float((pcgrad.flat_grad(loss).detach() / accumulation).float().norm().item())
                    for name, loss in adversary_losses.items()
                }
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
            projected_inputs = task_grad_sums
            if str(getattr(cfg, "pcgrad_balance", "none")) == "unit":
                raw_coop_grad_norms = {
                    name: float(grad.detach().float().norm().item())
                    for name, grad in task_grad_sums.items()
                }
                projected_inputs, pcgrad_balance_scales = pcgrad.unit_balance(
                    task_grad_sums,
                    {name: coop_weights[name] for name in task_grad_sums},
                )
            projected_coop = pcgrad.project_vectors(projected_inputs)
            if grad_now:
                grad_diag = pcgrad.vector_diagnostics(
                    projected_inputs, projected_coop, extra_grad_sum)
            pcgrad.write_(projected_coop + extra_grad_sum)
        if separate_clip:
            sae_pre_clip, sae_clip_scale = _clip_group(sae_params, cfg.grad_clip)
            route_pre_clip, route_clip_scale = _clip_group(rout_params, cfg.grad_clip)
            head_pre_clip, head_clip_scale = _clip_group(head_params, cfg.grad_clip)
            if separate_disc:
                main_disc_pre_clip, main_disc_clip_scale = 0.0, 1.0
            else:
                main_disc_pre_clip, main_disc_clip_scale = _clip_group(
                    disc_params, cfg.grad_clip)
            # Retain these names for the compact legacy log, where "main" now
            # denotes the shared SAE group. Detailed group values follow it.
            main_pre_clip, main_clip_scale = sae_pre_clip, sae_clip_scale
        else:
            main_pre_clip = float(nn.utils.clip_grad_norm_(all_params, cfg.grad_clip))
            main_clip_scale = min(1.0, cfg.grad_clip / max(main_pre_clip, 1e-12))
            sae_pre_clip = route_pre_clip = head_pre_clip = main_disc_pre_clip = main_pre_clip
            sae_clip_scale = route_clip_scale = head_clip_scale = main_disc_clip_scale = main_clip_scale
        if scaler.is_enabled():
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        if separate_disc:
            _set_requires_grad(disc_params, True)
        if grad_now and active_freq_for_router is not None and not model.routing.dynamic:
            with torch.no_grad():
                route_p_after = torch.softmax(model.routing.logits, dim=-1)[:, 1]
                active_p_soft_after = float(
                    (active_freq_for_router * route_p_after).sum()
                    / active_freq_for_router.sum().clamp(min=1e-12)
                )

        # ---- discriminator catch-up: track the moving encoder (no reversal) ----
        disc_pre_clip = 0.0
        disc_clip_scale = 1.0
        adversaries_on = any(weight > 0 for weight in (
            cfg.grl_weight, cfg.grl_phoneme_weight,
            cfg.grl_prosody_weight, cfg.grl_emotion_weight))
        disc_updates = cfg.n_disc_steps if separate_disc else max(0, cfg.n_disc_steps - 1)
        if disc_updates > 0 and adversaries_on:
            active_disc_optimizer = disc_optimizer if disc_optimizer is not None else optimizer
            for _ in range(disc_updates):
                active_disc_optimizer.zero_grad(set_to_none=True)
                for (zL_d, zP_d, lens_d, spk_d, tgt_d, tlen_d,
                     pf0_d, pv_d, pe_d, emo_d) in disc_micro_batches:
                    with _autocast(amp_dtype):
                        terms = []
                        if cfg.grl_weight > 0:
                            terms.append(U.speaker_adv_loss(
                                model.grl_head(zL_d, lens_d, 0.0), spk_d, lens_d))
                        if cfg.grl_phoneme_weight > 0:
                            terms.append(ctc_pr_loss(
                                model.pr_grl_head(zP_d, 0.0), tgt_d, lens_d, tlen_d))
                        if cfg.grl_prosody_weight > 0:
                            terms.append(U.prosody_train_loss(
                                model.prosody_grl_head(zL_d, 0.0),
                                pf0_d, pv_d, pe_d, lens_d))
                        if cfg.grl_emotion_weight > 0:
                            terms.append(F.cross_entropy(
                                model.emotion_grl_head(zL_d, lens_d, 0.0),
                                emo_d, weight=emo_w))
                        ld = sum(terms) if terms else zL_d.sum() * 0.0
                    if scaler.is_enabled(): scaler.scale(ld / accumulation).backward()
                    else: (ld / accumulation).backward()
                if scaler.is_enabled(): scaler.unscale_(active_disc_optimizer)
                disc_norm, current_disc_scale = _clip_group(disc_params, cfg.grad_clip)
                if disc_norm > disc_pre_clip:
                    disc_pre_clip = disc_norm
                    disc_clip_scale = current_disc_scale
                if scaler.is_enabled(): scaler.step(active_disc_optimizer); scaler.update()
                else: active_disc_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if disc_optimizer is not None:
                disc_optimizer.zero_grad(set_to_none=True)
        micro_index = 0
        task_grad_sums = None
        extra_grad_sum = None
        disc_micro_batches = []

        # ---- logging ----
        if log_now or grad_now:
            with torch.no_grad():
                sc, st = U.speaker_correct(out["sid_logits"], spk, olen)
                gc, gt = U.speaker_correct(out["grl_logits"], spk, olen)
                ec, et = U.class_correct(out["emotion_logits"], emo)
                if "emotion_grl_logits" in out:
                    lc, lt = U.class_correct(out["emotion_grl_logits"], emo)
                else:
                    lc = lt = 0
                pr_n, pr_d = U.ctc_errors(out["pr_logits"].float(), tgt, olen, tlen)
                if "pr_grl_logits" in out:
                    gp_n, gp_d = U.ctc_errors(out["pr_grl_logits"].float(), tgt, olen, tlen)
                else:
                    gp_n = gp_d = 0

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
                zL_emo_frame = (float(getattr(cfg, "grl_emotion_grad_norm_target", 0.0)) * ramp
                                if getattr(cfg, "grl_emotion_grad_norm", False)
                                else float("nan"))

            active_total = max(act_L + act_P, 1e-12)
            print(f"\n[train {step:05d}/{cfg.stage2_steps}] "
                  f"lr={lr_now:.2e} dann={ramp:.2f} "
                  f"zL_frame={zL_grl_frame:.2e} emo_frame={zL_emo_frame:.2e} "
                  f"recon={l_recon.item():.4f} aux={l_aux.item():.4f} "
                  f"dead={100*dead_frac:.1f}% "
                  f"active L/P={act_L:.0f}/{act_P:.0f} ({100*act_P/active_total:.1f}% P) "
                  f"assigned L/P={n_L}/{n_P}", flush=True)
            print(f"  content: z_L PER={pr_n/max(pr_d,1):.3f} | z_P adv PER={gp_n/max(gp_d,1):.3f}    "
                  f"speaker: z_P acc={sc/max(st,1):.3f} | z_L adv acc={gc/max(gt,1):.3f}", flush=True)
            print(f"  affect:  z_P emo acc={ec/max(et,1):.3f} | z_L emo-adv acc={lc/max(lt,1):.3f}    "
                  f"pros loss P/Ladv={l_pros.item():.3f}/{l_prosgrl.item():.3f}", flush=True)
            print(f"  route: Hbal={route_diag['balance_entropy']:.3f} "
                  f"Hunit={route_diag['unit_entropy']:.3f} "
                  f"spec={route_diag['specialized_frac_h_lt_0_5']:.2f} "
                  f"margin={route_diag['top1_top2_margin']:.3f} tau={model.routing.tau:.3f}    "
                  f"clip main={main_pre_clip:.2e}->{main_clip_scale:.2e} "
                  f"disc={disc_pre_clip:.2e}->{disc_clip_scale:.2e}", flush=True)
            if separate_clip:
                print("  clip groups: "
                      f"SAE={sae_pre_clip:.2e}->{sae_clip_scale:.2e} "
                      f"routing={route_pre_clip:.2e}->{route_clip_scale:.2e} "
                      f"positive_heads={head_pre_clip:.2e}->{head_clip_scale:.2e} "
                      f"main_disc={main_disc_pre_clip:.2e}->{main_disc_clip_scale:.2e}",
                      flush=True)
            if log_now:
                append_metrics(metrics_path, {
                    "step": step, "split": "train", "lr": lr_now,
                    "recon": float(l_recon), "pr": float(l_pr), "sid": float(l_sid),
                    "grl": float(l_grl), "grl_p": float(l_grl_p),
                    "prosody": float(l_pros), "prosody_grl": float(l_prosgrl),
                    "emotion": float(l_emo), "emotion_grl": float(l_emogrl),
                    "aux": float(l_aux),
                    "invariance": float(l_inv), "route_tau": float(model.routing.tau),
                    "active_L": act_L, "active_P": act_P, "dead_fraction": dead_frac,
                    "speaker_grl_frame": float(zL_grl_frame),
                    "emotion_grl_frame": float(zL_emo_frame),
                })

            def _norm_row(norms, names):
                return "  ".join(f"{name}={norms.get(name, 0.0):.2e}" for name in names)

            coop_order = ("recon", "pr", "sid", "prosody", "emotion", "inv", "aux")
            adv_order = ("grl", "grl_p", "pros_grl", "emo_grl", "route_spec")
            if grad_now and grad_diag is not None:
                coop_cos = grad_diag["coop_cosines"]
                if coop_cos:
                    sae_min_pair, sae_min_cos = min(coop_cos.items(), key=lambda item: item[1])
                else:
                    sae_min_pair, sae_min_cos = "n/a", float("nan")
                print("  SAE gradients (weighted, before clipping)", flush=True)
                print(f"    cooperative: {_norm_row(grad_diag['norms'], coop_order)}", flush=True)
                if raw_coop_grad_norms is not None:
                    print(f"    cooperative raw before unit balance: "
                          f"{_norm_row(raw_coop_grad_norms, coop_order)}", flush=True)
                    print("    unit-balance scales: "
                          f"{_norm_row(pcgrad_balance_scales or {}, coop_order)}", flush=True)
                print(f"    adversarial: {_norm_row(grad_diag['norms'], ('external_bundle',))}",
                      flush=True)
                if adv_sae_grad_norms is not None:
                    print(f"    adversarial factors: {_norm_row(adv_sae_grad_norms, adv_order)}",
                          flush=True)
                print(f"    PCGrad: raw={grad_diag['raw_coop_norm']:.2e}"
                      f"  projected={grad_diag['projected_coop_norm']:.2e}"
                      f"  adversary_bundle={grad_diag['external_norm']:.2e}", flush=True)
                print(f"    cooperative conflicts={grad_diag['coop_conflicts']}/{len(coop_cos)}"
                      f"  strongest={sae_min_pair}:{sae_min_cos:+.2f}", flush=True)

            if grad_now and router_grad_diag is not None:
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
            # Select using only heads that this preset actually trains. Composite
            # scores from different ablations therefore must not be compared
            # directly; fresh frozen probes provide that comparison.
            score = vm["per"] + (1 - vm["sid_acc"]) + (1 - vm["zP_emo_uar"])
            if cfg.grl_weight > 0:
                score += vm["zL_sid_acc"]
            if cfg.grl_phoneme_weight > 0:
                score += 1 - vm["zP_pr_per"]
            if cfg.grl_emotion_weight > 0 and math.isfinite(vm["zL_emo_uar"]):
                score += vm["zL_emo_uar"]
            print(f"\n[val step={step:05d}]", flush=True)
            print("  factor      kept representation       leakage representation", flush=True)
            print(f"  phoneme     z_L PER={vm['per']:.3f}"
                  f"              z_P PER={vm['zP_pr_per']:.3f}", flush=True)
            print(f"  speaker     z_P acc={vm['sid_acc']:.3f}"
                  f"              z_L acc={vm['zL_sid_acc']:.3f}", flush=True)
            print(f"  emotion     z_P UAR={vm['zP_emo_uar']:.3f} acc={vm['zP_emo_acc']:.3f}"
                  f"    z_L UAR={vm['zL_emo_uar']:.3f}", flush=True)
            print(f"  prosody     z_P loss={vm['zP_prosody_loss']:.3f}"
                  f"           z_L loss={vm['zL_prosody_loss']:.3f}", flush=True)
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
    final_path = save_runtime("final.pt", kind="inference")

    # The final training checkpoint is the primary reported artifact.  ``best.pt``
    # is retained as a validation diagnostic only: the composite proxy score mixes
    # heterogeneous jointly-trained heads and is not reliable enough to decide
    # which weights are allowed to see the held-out test set.
    tm = (evaluate_recon(model, test_dl, device, amp_dtype)
          if pure_recon_only else evaluate(model, test_dl, device, amp_dtype, n_emotion))
    append_metrics(metrics_path, {"step": step, "split": "test", **tm})
    final_payload = torch.load(final_path, map_location="cpu", weights_only=False)
    final_payload.setdefault("auxiliary", {})["test"] = tm
    if test_unseen_dl is not None:
        tum = (evaluate_recon(model, test_unseen_dl, device, amp_dtype)
               if pure_recon_only else evaluate(model, test_unseen_dl, device, amp_dtype, n_emotion))
        append_metrics(metrics_path, {"step": step, "split": "test_unseen", **tum})
        final_payload["auxiliary"]["test_unseen"] = tum
    atomic_torch_save(final_payload, final_path)
    mirror_file(final_path, mirror_dir)
    if metrics_path.exists():
        mirror_file(metrics_path, mirror_dir)
    if pure_recon_only:
        print(f"[test] final.pt: recon={tm['recon']:.4f}", flush=True)
        print(f"[msp] done. best recon={best:.4f}  ckpts in {ckpt_dir}")
    else:
        print(f"[test] final.pt: z_L PER={tm['per']:.3f} "
              f"z_P SID={tm['sid_acc']:.3f} z_L SID={tm['zL_sid_acc']:.3f} "
              f"z_P emo UAR={tm['zP_emo_uar']:.3f} z_L emo UAR={tm['zL_emo_uar']:.3f}",
              flush=True)
        print(f"[msp] done. best disent={best:.3f}  ckpts in {ckpt_dir}")
    return final_path
