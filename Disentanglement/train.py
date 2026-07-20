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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import DISConfig
from model import DISModel, build_dis_model
from data.dataset import make_stage1_dataloaders, make_stage2_dataloaders, _local_examples
from losses import (
    recon_loss, ctc_pr_loss, sid_ce_loss, sid_ce_loss_frames,
    route_loss, routing_spec_loss, decor_loss, ub_loss,
    inv_L_frame_cosine_loss, inv_P_stats_pool_loss,
    variance_floor_loss, effective_rank, bucket_diag,
)
from probe_robust.losses import vicreg_invariance_loss, vicreg_covariance_loss
from probe_robust.club import (
    CLUBSampled,
    no_collision_permutation,
    normalize_club_gradient,
)
from tb_logger import DISLogger
from training_runtime import (
    SegmentLimit, append_metrics, atomic_torch_save, checkpoint_payload,
    mirror_file, resolve_amp_precision, restore_training_state, validate_resume,
    accumulate_task_grads, apply_task_gradient_caps_,
)


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


def _dann_lambda(step: int, total: int, ramp_steps: int = 0) -> float:
    """DANN ramp: 0 at start → 1 at end.

    By default the ramp spans the full training schedule.  If ramp_steps > 0,
    it reaches 1.0 by that step and then stays there.
    """
    if int(ramp_steps) > 0:
        if step >= int(ramp_steps):
            return 1.0
        denom = max(1, int(ramp_steps))
    else:
        denom = max(1, total)
    p = step / denom
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


@torch.no_grad()
def _calibrate_route_topk_quotas(model, train_dl, device, cfg, use_bf16) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate learned route-local active budgets under the current global TopK.

    This is used after a learned-routing freeze.  We do not inject a preset
    240/16 split; instead, we measure how many globally selected units currently
    fall into each learned route, round that average to integer quotas that sum
    to cfg.topk, and store those quotas in the SAE.  Subsequent training/probing
    then keeps the learned route membership and the learned active split stable.
    """
    if getattr(model.routing, "dynamic", False):
        raise ValueError("route-local TopK calibration supports static learned routing only")

    n_routes = int(getattr(cfg, "n_routes", 3))
    max_batches = max(1, int(getattr(cfg, "route_topk_calib_batches", 20)))
    was_training = model.training
    route_topk_was_enabled = bool(getattr(model.sae, "route_topk_enabled").item())
    if route_topk_was_enabled:
        model.sae.clear_route_topk()
    model.eval()

    # Calibration should not consume the exact-resume sampler cursor.
    sampler_state = None
    if hasattr(train_dl, "sampler") and hasattr(train_dl.sampler, "state_dict"):
        sampler_state = train_dl.sampler.state_dict()

    route_idx = model.routing.logits.detach().argmax(dim=-1).to(device)
    route_counts = torch.stack([(route_idx == r).sum() for r in range(n_routes)]).long()
    active_counts = torch.zeros(n_routes, device=device)
    frame_count = torch.zeros((), device=device)
    batches_seen = 0
    data_iter = iter(train_dl)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16
           else torch.autocast("cuda", enabled=False))
    for _ in range(max_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)
        audios, audio_lengths = batch[0].to(device), batch[1].to(device)
        with ctx:
            out = model(audios, audio_lengths, stage=2, grl_lambda=0.0,
                        grl_p_lambda=0.0, emit_emotion=False)
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

    if was_training:
        model.train()
    else:
        model.eval()

    if frame_count.item() <= 0 or batches_seen == 0:
        raise ValueError("route-local TopK calibration saw no valid frames")

    avg = active_counts / frame_count
    quotas = torch.floor(avg).long().cpu()
    frac = (avg.cpu() - quotas.float())
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
                raise ValueError(
                    f"cannot reduce route-local TopK quotas to target {target_topk}")

    route_idx_cpu = route_idx.cpu()
    model.sae.set_route_topk(route_idx_cpu, quotas)
    print("[stage 2] learned-route TopK calibrated: "
          f"batches={batches_seen} avg_active={avg.cpu().tolist()} "
          f"quotas={quotas.tolist()} route_counts={route_counts_cpu.tolist()} "
          f"sum={int(quotas.sum().item())}")
    return route_idx_cpu, quotas


def _reset_adversary_heads_after_resume(model, optimizer, requested: str = "") -> None:
    """Reinitialize adversarial discriminator heads after exact resume.

    By default this resets all GRL/adversary heads for backwards compatibility
    with --reset_adversaries_on_resume.  If ``requested`` is non-empty, only the
    requested comma-separated heads/aliases are reset.  In all cases the SAE,
    encoder, routing partition, PR task head, and SID task head are left
    untouched.
    """
    all_adversary_names = (
        "grl_head",
        "pr_grl_head",
        "grl_head_u",
        "pr_grl_head_u",
        "prosody_grl_head",
        "prosody_grl_head_u",
        "emotion_grl_head",
    )
    aliases = {
        "speaker": ("grl_head",),
        "zl_speaker": ("grl_head",),
        "phoneme": ("pr_grl_head",),
        "zp_phoneme": ("pr_grl_head",),
        "u": ("grl_head_u", "pr_grl_head_u"),
        "prosody": ("prosody_grl_head", "prosody_grl_head_u"),
        "emotion": ("emotion_grl_head",),
        "all": all_adversary_names,
    }
    if str(requested).strip():
        expanded: list[str] = []
        for raw in str(requested).split(","):
            key = raw.strip()
            if not key:
                continue
            expanded.extend(aliases.get(key, (key,)))
        unknown = [name for name in expanded
                   if name not in all_adversary_names and not hasattr(model, name)]
        if unknown:
            raise ValueError(
                "unknown adversary head(s) requested for reset: "
                f"{unknown}; valid modules={all_adversary_names}; aliases={sorted(aliases)}")
        # Preserve order while dropping duplicates.
        seen: set[str] = set()
        adversary_names = tuple(
            name for name in expanded
            if not (name in seen or seen.add(name))
        )
    else:
        adversary_names = all_adversary_names

    def _maybe_reset(module):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    reset_names: list[str] = []
    reset_params: list[torch.nn.Parameter] = []
    for name in adversary_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        module.apply(_maybe_reset)
        reset_names.append(name)
        reset_params.extend(list(module.parameters()))

    cleared = 0
    if optimizer is not None:
        for param in reset_params:
            if param in optimizer.state:
                optimizer.state.pop(param, None)
                cleared += 1

    if reset_names:
        print("[stage 2] adversary heads RESET after resume: "
              f"heads={','.join(reset_names)} optimizer_states_cleared={cleared}")
    else:
        print("[stage 2] adversary-head reset requested, but no matching heads were present")


def _scheduled_grl_grad_norm_target(cfg, step: int) -> float:
    """Effective z_L speaker-GRL grad-norm target for this training step.

    The GRLHead owns one fixed grad_norm_target and is reused by z_U speaker
    adversaries too.  For the z_L-only decay experiment we therefore keep the
    head target fixed and scale only the z_L speaker reversal lambda by
    scheduled_target / base_target.
    """
    base = float(getattr(cfg, "grl_grad_norm_target", 1.0))
    final = float(getattr(cfg, "grl_grad_norm_final_target", -1.0))
    if final < 0.0:
        return base
    start = int(getattr(cfg, "grl_grad_norm_decay_start", 0))
    end = int(getattr(cfg, "grl_grad_norm_decay_end", 0))
    if end <= start:
        return base
    if step <= start:
        return base
    if step >= end:
        return final
    frac = (step - start) / max(1, end - start)
    return base + frac * (final - base)


def _scheduled_grl_lambda_scale(cfg, step: int) -> float:
    base = float(getattr(cfg, "grl_grad_norm_target", 1.0))
    if base <= 0.0:
        return 1.0
    return _scheduled_grl_grad_norm_target(cfg, step) / base


def _gumbel_tau(step: int, total: int, tau_start: float, tau_end: float) -> float:
    return tau_start * (tau_end / tau_start) ** (step / max(1, total))


def _count_params(model: DISModel):
    frozen    = sum(p.numel() for p in model.encoder._spear.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen, trainable


def _save_checkpoint(path, model, optimizer, scheduler, step, best_metric, *,
                     cfg=None, scaler=None, club_module=None, club_phn_module=None,
                     kind="resume", val_metrics=None, train_sampler=None,
                     pa_loader=None, pb_loader=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        auxiliary = {"val": val_metrics or {}}
        if club_module is not None:
            auxiliary["club"] = {
                "model": club_module.state_dict(),
                "optimizer": club_module.optimizer.state_dict(),
            }
        if club_phn_module is not None:
            auxiliary["club_phoneme"] = {
                "model": club_phn_module.state_dict(),
                "optimizer": club_phn_module.optimizer.state_dict(),
            }
        if train_sampler is not None and hasattr(train_sampler, "state_dict"):
            auxiliary["train_sampler"] = train_sampler.state_dict()
        if pa_loader is not None and hasattr(pa_loader.dataset, "rng"):
            auxiliary["pair_alpha_rng"] = pa_loader.dataset.rng.getstate()
        if pb_loader is not None and hasattr(pb_loader.dataset, "rng"):
            auxiliary["pair_beta_rng"] = pb_loader.dataset.rng.getstate()
        payload = checkpoint_payload(
            model=model, optimizer=optimizer if kind == "resume" else None,
            scheduler=scheduler if kind == "resume" else None,
            scaler=scaler if kind == "resume" else None,
            step=step, best_metric=best_metric, cfg=cfg,
            auxiliary=auxiliary,
            dataset_hash=str(getattr(cfg, "dataset_fingerprint", "")),
            preset=str(getattr(cfg, "experiment_preset", "")), kind=kind,
        )
        atomic_torch_save(payload, path)
        mirror = Path(cfg.drive_mirror) if str(getattr(cfg, "drive_mirror", "")) else None
        mirror_file(path, mirror)
        metrics = Path(cfg.checkpoint_dir) / "metrics.jsonl"
        if metrics.exists(): mirror_file(metrics, mirror)
        return
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
         "pr_head.", "sid_head.", "grl_head.", "pr_grl_head.",
         "prosody_head.", "prosody_grl_head.", "prosody_grl_head_u.",
         "emotion_head.", "emotion_grl_head.", "encoder._spear."))]
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


# ---------------------------------------------------------------- adversary readouts
# An adversary's CE/CTC loss alone is hard to read (e.g. speaker CE 4.5 vs chance
# ln(251)=5.52 — is that "removed"?).  The interpretable readout is how much of the
# factor the adversary can still EXTRACT: speaker top-1 accuracy (chance 1/S) for the
# speaker adversaries, greedy PER (chance ~1.0) for the phoneme CTC adversaries.
# Both helpers return raw counts so eval can accumulate exactly over the val set.

def _is_branch_logits(logits) -> bool:
    return isinstance(logits, (tuple, list))


def _speaker_adv_loss(logits, speaker_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Speaker adversary CE for pooled, frame-level, or branched readouts.

    Branched adversaries average branch losses so enabling an extra readout does
    not silently multiply the configured GRL weight.
    """
    if _is_branch_logits(logits):
        losses = [_speaker_adv_loss(branch, speaker_ids, lengths) for branch in logits]
        return torch.stack(losses).mean()
    if logits.dim() == 3:
        return sid_ce_loss_frames(logits, speaker_ids, lengths)
    return sid_ce_loss(logits, speaker_ids)


@torch.no_grad()
def _speaker_correct(logits, speaker_ids: torch.Tensor, lengths: torch.Tensor):
    """(#correct, #total) top-1 speaker hits for an adversary head.

    Handles frame-level (B, T, S), pooled (B, S), and branched heads.  For
    branched heads we report the strongest branch on this batch; averaged branch
    accuracy can hide the leakage path the adversary is supposed to cover.
    """
    if _is_branch_logits(logits):
        branch_counts = [_speaker_correct(branch, speaker_ids, lengths) for branch in logits]
        return max(branch_counts, key=lambda ct: ct[0] / max(ct[1], 1))
    if logits.dim() == 3:
        B, T, _ = logits.shape
        pred = logits.argmax(dim=-1)                                        # (B, T)
        mask = (torch.arange(T, device=logits.device).unsqueeze(0)
                < lengths.unsqueeze(1))
        tgt  = speaker_ids.unsqueeze(1).expand(B, T)
        return int(((pred == tgt) & mask).sum().item()), int(mask.sum().item())
    pred = logits.argmax(dim=-1)                                            # (B,)
    return int((pred == speaker_ids).sum().item()), int(speaker_ids.numel())


def _random_speaker_targets(
    speaker_ids: torch.Tensor,
    num_speakers: int,
    seed: int,
    step: int,
) -> torch.Tensor:
    """Deterministic random targets for the shuffled-speaker control."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 104729 * int(step))
    targets = torch.randint(
        0, num_speakers, speaker_ids.shape, generator=gen, device="cpu"
    )
    return targets.to(speaker_ids.device)


@torch.no_grad()
def _class_correct(logits: torch.Tensor, labels: torch.Tensor):
    pred = logits.argmax(dim=-1)
    return int((pred == labels).sum().item()), int(labels.numel())


def _cap_loss_by_scaling(loss: torch.Tensor, cap: float):
    """Scale a loss down to at most `cap` without zeroing its gradient direction."""
    cap = float(cap or 0.0)
    if cap <= 0:
        return loss, loss.detach().new_tensor(1.0)
    detached = loss.detach().clamp_min(1e-8)
    scale = torch.clamp(detached.new_tensor(cap) / detached, max=1.0)
    return loss * scale, scale


@torch.no_grad()
def _ctc_errors(logits: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor):
    """(edit-distance, ref-length) for a CTC adversary head via greedy decode."""
    preds = _greedy_ctc_decode(logits, input_lengths)
    num = den = 0
    for i, pred_ids in enumerate(preds):
        ref = targets[i, :target_lengths[i]].tolist()
        num += _edit_distance(pred_ids, ref)
        den += len(ref)
    return num, den


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
    # adversary readouts (speaker acc / phoneme PER) — accumulated only when the
    # corresponding adversary head is present in the model's output.
    grl_c, grl_t        = 0, 0    # speaker adversary on z_L
    grlu_c, grlu_t      = 0, 0    # speaker adversary on z_U
    grlp_n, grlp_d      = 0, 0    # phoneme adversary on z_P
    grlpu_n, grlpu_d    = 0, 0    # phoneme adversary on z_U
    n = 0
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)

    for audios, audio_lengths, targets, target_lengths, speaker_ids in val_dl:
        audios, audio_lengths = audios.to(device), audio_lengths.to(device)
        targets, target_lengths = targets.to(device), target_lengths.to(device)
        speaker_ids = speaker_ids.to(device)

        with ctx:
            out = model(audios, audio_lengths, stage=2, grl_lambda=0.0, emit_emotion=False)
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

        # adversary readouts — how much factor each adversary can still extract
        c, t = _speaker_correct(out["grl_logits"], speaker_ids, out["out_lengths"])
        grl_c += c; grl_t += t
        if "grl_u_logits" in out:
            c, t = _speaker_correct(out["grl_u_logits"], speaker_ids, out["out_lengths"])
            grlu_c += c; grlu_t += t
        if "pr_grl_logits" in out:
            num, den = _ctc_errors(out["pr_grl_logits"], targets, out["out_lengths"], target_lengths)
            grlp_n += num; grlp_d += den
        if "pr_grl_u_logits" in out:
            num, den = _ctc_errors(out["pr_grl_u_logits"], targets, out["out_lengths"], target_lengths)
            grlpu_n += num; grlpu_d += den

    model.train()
    metrics = {
        "recon":   r_total   / max(n, 1),
        "pr":      pr_total  / max(n, 1),
        "per":     per_num   / max(per_den, 1),
        "sid_acc": sid_correct / max(sid_total, 1),
        "grl_acc": grl_c     / max(grl_t, 1),     # speaker still readable from z_L (chance 1/S)
    }
    if grlp_d  > 0: metrics["grl_p_per"]   = grlp_n  / grlp_d   # phoneme still readable from z_P
    if grlu_t  > 0: metrics["grl_u_acc"]   = grlu_c  / grlu_t   # speaker still readable from z_U
    if grlpu_d > 0: metrics["grl_p_u_per"] = grlpu_n / grlpu_d  # phoneme still readable from z_U
    return metrics


@torch.no_grad()
def _eval_emotion(model, val_dl, device, use_bf16) -> Dict[str, float]:
    model.eval()
    loss_total, n_batches = 0.0, 0
    correct, total = 0, 0
    grl_loss_total, grl_batches = 0.0, 0
    grl_correct, grl_total = 0, 0
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)

    for audios, audio_lengths, labels in val_dl:
        audios = audios.to(device, non_blocking=True)
        audio_lengths = audio_lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with ctx:
            out = model(audios, audio_lengths, stage=2, grl_lambda=0.0,
                        grl_p_lambda=0.0, grl_emotion_lambda=0.0,
                        emit_emotion=True)
            loss = F.cross_entropy(out["emotion_logits"], labels)
            if "emotion_grl_logits" in out:
                grl_loss = F.cross_entropy(out["emotion_grl_logits"], labels)
            else:
                grl_loss = None
        loss_total += loss.item()
        n_batches += 1
        c, t = _class_correct(out["emotion_logits"], labels)
        correct += c; total += t
        if grl_loss is not None:
            grl_loss_total += grl_loss.item()
            grl_batches += 1
            c, t = _class_correct(out["emotion_grl_logits"], labels)
            grl_correct += c; grl_total += t

    model.train()
    metrics = {
        "emotion_loss": loss_total / max(n_batches, 1),
        "emotion_acc": correct / max(total, 1),
    }
    if grl_batches > 0:
        metrics["emotion_zL_loss"] = grl_loss_total / grl_batches
        metrics["emotion_zL_acc"] = grl_correct / max(grl_total, 1)
    return metrics


@torch.no_grad()
def _eval_club_estimators(model, loader, device, use_bf16,
                          club_module=None, club_phn_module=None) -> Dict[str, float]:
    """Held-out q_phi fit diagnostics; these are not leakage-probe results."""
    if club_module is None and club_phn_module is None:
        return {}
    was_training = model.training; model.eval()
    totals = {"speaker_loss": 0.0, "speaker_correct": 0, "speaker_n": 0,
              "phoneme_loss": 0.0, "phoneme_correct": 0, "phoneme_n": 0}
    for batch in loader:
        audios, lengths, _, _, speakers = batch[:5]
        audios, lengths, speakers = audios.to(device), lengths.to(device), speakers.to(device)
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16
               else torch.autocast("cuda", enabled=False))
        with ctx:
            out = model(audios, lengths, stage=2, grl_lambda=0.0, emit_emotion=False)
            if club_module is not None:
                z, olen = out["z_L"], out["out_lengths"]
                mask = (torch.arange(z.shape[1], device=device)[None] < olen[:, None]).float().unsqueeze(-1)
                n = olen.float().clamp_min(1).unsqueeze(1)
                mean = (z * mask).sum(1) / n
                var = (((z - mean[:, None]) ** 2) * mask).sum(1) / n
                logits = club_module.classifier(torch.cat([mean, (var + 1e-5).sqrt()], -1))
                totals["speaker_loss"] += float(F.cross_entropy(logits, speakers, reduction="sum"))
                totals["speaker_correct"] += int((logits.argmax(-1) == speakers).sum())
                totals["speaker_n"] += speakers.numel()
            if club_phn_module is not None:
                z, olen = out["z_P"], out["out_lengths"]
                mask = torch.arange(z.shape[1], device=device)[None] < olen[:, None]
                zf = z[mask]; labels = out["pr_logits"].argmax(-1)[mask]
                logits = club_phn_module.classifier(zf)
                totals["phoneme_loss"] += float(F.cross_entropy(logits, labels, reduction="sum"))
                totals["phoneme_correct"] += int((logits.argmax(-1) == labels).sum())
                totals["phoneme_n"] += labels.numel()
    if was_training: model.train()
    result = {}
    if totals["speaker_n"]:
        result.update(club_val_ce=totals["speaker_loss"] / totals["speaker_n"],
                      club_val_acc=totals["speaker_correct"] / totals["speaker_n"])
    if totals["phoneme_n"]:
        result.update(club_phn_val_ce=totals["phoneme_loss"] / totals["phoneme_n"],
                      club_phn_val_acc=totals["phoneme_correct"] / totals["phoneme_n"])
    return result


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
                           eff_grl_weight: float = -1.0, grl_p_lam=None,
                           shuffle_grl_labels: bool = False,
                           club_module=None, club_phn_module=None) -> None:
    """Per-loss gradient norms on SAE params — used for weight calibration."""
    device = torch.device(cfg.device)
    audios, audio_lengths, targets, target_lengths, speaker_ids = batch[:5]   # invariance batch has a 6th (perturbed) elem
    audios        = audios.to(device)
    audio_lengths = audio_lengths.to(device)
    targets       = targets.to(device)
    target_lengths = target_lengths.to(device)
    speaker_ids   = speaker_ids.to(device)
    adversary_speaker_ids = (
        _random_speaker_targets(speaker_ids, cfg.num_speakers, cfg.seed, step)
        if shuffle_grl_labels else speaker_ids
    )

    sae_params = [p for p in model.sae.parameters() if p.requires_grad]
    shared_param = model.sae.enc_weight
    shared_idx = next(i for i, p in enumerate(sae_params) if p is shared_param)

    def _grad_vec(loss, retain):
        """Return (encoder-gradient vector, full-SAE gradient norm).

        Historical norm values cover all SAE parameters. Cosines use only the
        shared encoder matrix, avoiding unequal vectors when reconstruction also
        updates the decoder.
        """
        grads = torch.autograd.grad(loss, sae_params, retain_graph=retain,
                                    allow_unused=True, create_graph=False)
        norm_parts = [g.detach().float().pow(2).sum() for g in grads if g is not None]
        full_norm = float(torch.stack(norm_parts).sum().sqrt().item()) if norm_parts else 0.0
        shared_grad = grads[shared_idx]
        shared_vec = (shared_grad.detach().float().flatten()
                      if shared_grad is not None else None)
        return shared_vec, full_norm

    def _norm(loss, retain):
        return _grad_vec(loss, retain)[1]

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
    with ctx:
        _no_routing     = getattr(cfg, 'no_routing', False)
        _projection     = getattr(cfg, 'projection_disentanglement', False)
        _routing_active = (not _no_routing and not _projection and
                           not getattr(cfg, 'freeze_learned_routing_on_resume', False) and
                           any(p.requires_grad for p in model.routing.parameters()))

        out     = model(audios, audio_lengths, stage=2, grl_lambda=grl_lam,
                        grl_p_lambda=grl_p_lam, emit_emotion=False)
        l_recon = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
        l_pr    = ctc_pr_loss(out["pr_logits"], targets, out["out_lengths"], target_lengths)
        l_sid   = sid_ce_loss(out["sid_logits"], speaker_ids)
        l_route = (route_loss(model.routing.logits) if _routing_active
                   else l_recon.new_zeros(()))

        raw: Dict[str, float] = {}
        # Probe-robust runs (grl_weight=0): GRL forward + grad measurement is
        # not informative — the head receives no gradient and is essentially
        # an untrained random projection. Skip its diagnostics entirely.
        _grl_on   = (cfg.grl_weight > 0)
        _grl_p_w  = getattr(cfg, 'grl_phoneme_weight', 0.0)
        _grl_p_on = (_grl_p_w > 0)

        l_grl = l_recon.new_zeros(())
        if _grl_on:
            l_grl = _speaker_adv_loss(out["grl_logits"], adversary_speaker_ids, out["out_lengths"])
        l_grl_p_gn = (ctc_pr_loss(out["pr_grl_logits"], targets, out["out_lengths"], target_lengths)
                      if (_grl_p_on and "pr_grl_logits" in out)
                      else l_recon.new_zeros(()))

        # --- PER-FRAME gradient each adversary delivers to its block (reversal
        # strength factored out, lam=1) — exposes the pooled-vs-dense dilution.
        # Computed only when the corresponding adversary is actually on.
        def _per_frame_grad(loss, z):
            g = torch.autograd.grad(loss, z, retain_graph=True, allow_unused=True)[0]
            if g is None:
                return 0.0
            pf = g.float().norm(dim=-1)                                   # (B, T)
            Tg = pf.shape[1]
            m  = (torch.arange(Tg, device=z.device).unsqueeze(0) < out["out_lengths"].unsqueeze(1)).float()
            return float((pf * m).sum() / m.sum().clamp(min=1))
        pf_grl = 0.0
        if _grl_on:
            sp_lg  = model.grl_head(out["z_L"], out["out_lengths"], 1.0)
            L_sp   = _speaker_adv_loss(sp_lg, adversary_speaker_ids, out["out_lengths"])
            pf_grl = _per_frame_grad(L_sp, out["z_L"])
        pf_grl_p = 0.0
        if _grl_p_on and hasattr(model, 'pr_grl_head'):
            ph_lg  = model.pr_grl_head(out["z_P"], 1.0)
            L_ph   = ctc_pr_loss(ph_lg, targets, out["out_lengths"], target_lengths)
            pf_grl_p = _per_frame_grad(L_ph, out["z_P"])

        loss_terms = [
            ("recon",    l_recon,              True),
            ("pr",       l_pr,                 True),
            ("sid",      l_sid,                True),
        ]
        if _grl_on:
            loss_terms.append(("grl", l_grl, True))
        if _grl_p_on:
            # With dann_full_discriminator the weight already lives in the lambda.
            _dann_fix = getattr(cfg, 'dann_full_discriminator', False)
            loss_terms.append(("grl_p", l_grl_p_gn if _dann_fix else _grl_p_w * l_grl_p_gn, True))

        # ---- probe_robust: CLUB gradient measurement on SAE encoder ----
        # Recompute the same z_L stats-pool / z_P frame batches the main loop
        # uses, call mi_bound at current q_phi state, take encoder gradient.
        # No inner_step here (we are measuring, not training q_phi).
        _club_on     = (club_module     is not None)
        _club_phn_on = (club_phn_module is not None
                        and step >= int(getattr(cfg, 'club_phoneme_warmup_steps', 0)))
        if _club_on:
            _eff_w_meas, _ = _effective_club_scaling(cfg, step)
            _zL    = out["z_L"]
            if bool(getattr(cfg, "club_grad_norm", False)):
                _zL = normalize_club_gradient(
                    _zL,
                    target=float(cfg.club_grad_norm_target),
                    weight=_eff_w_meas,
                )
            _olen2 = out["out_lengths"]
            _Tcl   = _zL.shape[1]
            _fm    = (torch.arange(_Tcl, device=device).unsqueeze(0)
                      < _olen2.unsqueeze(1)).float().unsqueeze(-1)
            _n     = _olen2.float().clamp(min=1).unsqueeze(1)
            _mean  = (_zL * _fm).sum(1) / _n
            _var2  = (((_zL - _mean.unsqueeze(1)) ** 2) * _fm).sum(1) / _n
            _std   = (_var2 + 1e-5).sqrt()
            _z_pool = torch.cat([_mean, _std], dim=-1)
            if bool(getattr(cfg, "club_no_collision_negatives", False)):
                _neg = no_collision_permutation(speaker_ids)
                l_club_gn = club_module.mi_bound(
                    _z_pool, speaker_ids,
                    negative_labels=speaker_ids[_neg],
                )
            else:
                l_club_gn = club_module.mi_bound(_z_pool, speaker_ids)
            loss_terms.append(("club", _eff_w_meas * l_club_gn, True))
        if _club_phn_on:
            _zP    = out["z_P"]
            _olen3 = out["out_lengths"]
            _Tp    = _zP.shape[1]
            _fmP   = (torch.arange(_Tp, device=device).unsqueeze(0)
                      < _olen3.unsqueeze(1))
            _zP_flat = _zP[_fmP]
            with torch.no_grad():
                _phn_pseudo = out["pr_logits"].argmax(dim=-1)
            _y_phn = _phn_pseudo[_fmP]
            l_club_phn_gn = club_phn_module.mi_bound(_zP_flat, _y_phn)
            loss_terms.append(("club_phn",
                               float(cfg.club_phoneme_weight) * l_club_phn_gn, True))
        if _routing_active:
            loss_terms.append(("route", cfg.rho * l_route, False))

        # ---- per-task gradient on the routing masks (m_L, m_P) ----
        # Shows which mask each task is pushing on: PR/CLUB should pressure m_L,
        # SID/GRL_p should pressure m_P. Cross-bucket pressure (e.g. PR on m_P,
        # or CLUB on m_P) reveals leakage of the mechanism into the wrong bucket.
        # Computed BEFORE the SAE-grad loop with retain_graph=True so the
        # existing SAE retain pattern (last term frees) stays correct.
        _mL_t = out.get("m_L"); _mP_t = out.get("m_P")
        mask_grads: Dict[str, tuple] = {}
        if _routing_active and _mL_t is not None and _mP_t is not None:
            for name, loss, _retain in loss_terms:
                try:
                    g_pair = torch.autograd.grad(
                        loss, [_mL_t, _mP_t],
                        retain_graph=True, allow_unused=True, create_graph=False,
                    )
                    gL_n = (float(g_pair[0].detach().float().norm().item())
                            if g_pair[0] is not None else 0.0)
                    gP_n = (float(g_pair[1].detach().float().norm().item())
                            if g_pair[1] is not None else 0.0)
                    mask_grads[name] = (gL_n, gP_n)
                except RuntimeError:
                    # Loss doesn't depend on masks (e.g. recon path bypassing
                    # the masked z) — record zeros rather than crash.
                    mask_grads[name] = (0.0, 0.0)

        flat_vecs: Dict[str, torch.Tensor] = {}
        for name, loss, retain in loss_terms:
            v, n = _grad_vec(loss, retain)
            raw[name] = n
            if v is not None and n > 1e-12:
                flat_vecs[name] = v
        if not _routing_active:
            raw["route"] = 0.0
        if not _grl_on:
            raw["grl"] = 0.0
        if not _grl_p_on:
            raw["grl_p"] = 0.0

        # Pairwise cosines between per-loss gradients on the shared SAE encoder.
        # cos>0: tasks agree on which params to move; cos<0: they fight on the
        # same params and the sum-then-step optimizer cancels signal.
        cos_pairs: Dict[str, float] = {}
        names_in_order = [n for n in ("recon", "pr", "sid", "grl", "grl_p",
                                       "club", "club_phn", "route")
                          if n in flat_vecs]
        for i, a in enumerate(names_in_order):
            va = flat_vecs[a]
            na = va.norm().clamp(min=1e-12)
            for b in names_in_order[i + 1:]:
                vb = flat_vecs[b]
                nb = vb.norm().clamp(min=1e-12)
                cos_pairs[f"{a}_vs_{b}"] = float((va @ vb / (na * nb)).item())

    grl_w  = eff_grl_weight if eff_grl_weight >= 0 else cfg.grl_weight
    norms = {
        "recon":        raw["recon"],
        "pr_raw":       raw["pr"],
        "pr_weighted":  raw["pr"]  * cfg.alpha,
        "sid_raw":      raw["sid"],
        "sid_weighted": raw["sid"] * cfg.beta,
    }
    if _grl_on:
        norms["grl"]   = raw["grl"] * grl_w
    if _grl_p_on:
        norms["grl_p"] = raw.get("grl_p", 0.0)
    if _club_on:
        # raw["club"] already has cfg.club_weight baked in (we passed the
        # weighted loss into loss_terms above).
        norms["club"] = raw.get("club", 0.0)
    if _club_phn_on:
        norms["club_phn"] = raw.get("club_phn", 0.0)
    if _routing_active:
        norms["route"] = raw["route"]

    recon_n = norms["recon"]
    lines   = [f"  [grad_norms @{step}]"]
    for k, v in norms.items():
        ratio = v / recon_n if recon_n > 1e-8 else float("nan")
        lines.append(f"    {k:<16s}  |g|={v:.5f}  ratio={ratio:.3f}x recon")
    # per-frame adversary gradient density — only meaningful when an adversary
    # is actually on. Probe-robust runs (CLUB / VICReg, grl_weight=0) skip it.
    if _grl_on or _grl_p_on:
        ratio_pf = (pf_grl_p / pf_grl) if pf_grl > 1e-12 else float("nan")
        lines.append(f"    per-frame |dL/dz[t]|:  grl(z_L)={pf_grl:.5f}   grl_p(z_P)={pf_grl_p:.5f}"
                     f"   (grl_p/grl = {ratio_pf:.1f}x  <- pooled-vs-dense dilution)")
    if cos_pairs:
        lines.append(f"    [grad_cos @{step}]  shared=sae.enc_weight; cos<0 = conflict")
        for pair, c in cos_pairs.items():
            tag = "  conflict" if c < -0.05 else ("  aligned" if c > 0.05 else "")
            lines.append(f"      cos({pair:<22s}) = {c:+.3f}{tag}")
    if mask_grads:
        # Per-task gradient |dL/dm| on each routing mask.
        # tilt = (|gL| - |gP|) / (|gL| + |gP|); +1 = all pressure on m_L,
        # -1 = all pressure on m_P, 0 = symmetric.  Healthy routing:
        # PR & CLUB tilt -> +1 (pushing m_L); SID & GRL_p tilt -> -1.
        lines.append(f"    [mask_grad @{step}]  |dL/dm_L| / |dL/dm_P|  (tilt: +1=all on L, -1=all on P)")
        for name, (gL, gP) in mask_grads.items():
            denom = (gL + gP)
            tilt = ((gL - gP) / denom) if denom > 1e-12 else 0.0
            tag = ""
            if   tilt > +0.30: tag = "  -> L"
            elif tilt < -0.30: tag = "  -> P"
            lines.append(f"      {name:<16s} mL={gL:.5f}  mP={gP:.5f}  tilt={tilt:+.2f}{tag}")
    print("\n".join(lines))
    norms["grl_perframe"] = pf_grl
    norms["grl_p_perframe"] = pf_grl_p
    tb.log_grad_norms(step, norms)
    if cos_pairs:
        tb.log_grad_cosines(step, cos_pairs)
    # Per-task mask gradient magnitudes to TB (one scalar per task per mask)
    for name, (gL, gP) in mask_grads.items():
        tb.log_grad_norms(step, {
            f"mask_grad/{name}_mL": gL,
            f"mask_grad/{name}_mP": gP,
        })


def _masked_mse(a: torch.Tensor, b: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid frames (a, b: (B, T, D))."""
    T = a.shape[1]
    mask = (torch.arange(T, device=a.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).float().unsqueeze(-1)
    return (((a - b) ** 2) * mask).sum() / mask.sum().clamp(min=1) / a.shape[-1]


def _stats_pool(z: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Masked mean+std pooling used by speaker CLUB and its diagnostics."""
    T = z.shape[1]
    mask = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)
            ).to(z.dtype).unsqueeze(-1)
    n = lengths.to(z.dtype).clamp(min=1).unsqueeze(1)
    mean = (z * mask).sum(1) / n
    var = (((z - mean.unsqueeze(1)) ** 2) * mask).sum(1) / n
    return torch.cat([mean, (var + 1e-5).sqrt()], dim=-1)


def _current_grad_norm(params) -> float:
    parts = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
    return float(torch.stack(parts).sum().sqrt().item()) if parts else 0.0


def _effective_club_scaling(cfg, step: int) -> tuple[float, int]:
    """Return (effective_club_weight, effective_inner_steps) for this step.

    During the CLUB warmup phase the encoder-facing weight is held at 0 so an
    under-fit q_phi cannot push z_L into arbitrary directions, and q_phi trains
    with `club_pretrain_inner_steps` inner CE steps per accumulation boundary
    so it approximates p(speaker|z_L) before we descend its bound. After
    warmup, both values fall back to the configured `club_weight` and
    `club_inner_steps`. Matches Cheng 2020 Algorithm 1's guidance to converge
    q_phi before optimising against its output.
    """
    warmup = int(getattr(cfg, "club_warmup_steps", 0))
    if warmup > 0 and int(step) < warmup:
        boosted = int(getattr(cfg, "club_pretrain_inner_steps",
                              getattr(cfg, "club_inner_steps", 3)))
        return 0.0, boosted
    return float(cfg.club_weight), int(cfg.club_inner_steps)


def _club_full_diagnostics(
    *, model, step: int, objective_terms: Dict[str, torch.Tensor],
    routing_params, out, club_module, raw_club_loss: torch.Tensor,
    normalized_club_loss: torch.Tensor,
    controlled_negative_collision: float,
) -> list[str]:
    """Expensive opt-in evidence for a one-shot CLUB calibration run.

    All objective gradients are measured on the graph used by the actual
    optimizer step. CLUB's raw and normalized paths use the same negative
    labels, so their difference is solely the gradient transform.
    """
    enc = model.sae.enc_weight
    params = [enc] + list(routing_params)
    vectors: Dict[str, torch.Tensor] = {}
    rows = []
    for name, loss in objective_terms.items():
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            continue
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True, create_graph=False)
        g_enc = grads[0]
        enc_norm = float(g_enc.detach().float().norm().item()) if g_enc is not None else 0.0
        route_parts = [g.detach().float().pow(2).sum() for g in grads[1:] if g is not None]
        route_norm = (float(torch.stack(route_parts).sum().sqrt().item())
                      if route_parts else 0.0)
        rows.append((name, float(loss.detach().float().item()), enc_norm, route_norm))
        if g_enc is not None and enc_norm > 1e-12:
            vectors[name] = g_enc.detach().float().flatten()

    def _frame_summary(loss: torch.Tensor):
        grad = torch.autograd.grad(loss, out["z_L"], retain_graph=True,
                                   allow_unused=True, create_graph=False)[0]
        if grad is None:
            return (0.0,) * 5
        norms = grad.detach().float().norm(dim=-1)
        T = norms.shape[1]
        valid = torch.arange(T, device=norms.device).unsqueeze(0) < out["out_lengths"].unsqueeze(1)
        values = norms[valid]
        if values.numel() == 0:
            return (0.0,) * 5
        qs = torch.quantile(values, values.new_tensor([0.5, 0.9]))
        return (float(values.mean()), float(qs[0]), float(qs[1]),
                float(values.max()), float((values <= 1e-12).float().mean()))

    raw_stats = _frame_summary(raw_club_loss)
    normalized_stats = _frame_summary(normalized_club_loss)
    lines = [f"  [club_full_diag @{step}] objective=weighted_current_microbatch"]
    lines.append("    objective             value       |g_enc|    |g_route|")
    for name, value, enc_norm, route_norm in rows:
        lines.append(f"    {name:<20s} {value:+10.5f}  {enc_norm:10.5f}  {route_norm:10.5f}")
    if "club" in vectors:
        for name, vec in vectors.items():
            if name == "club":
                continue
            cosine = float(F.cosine_similarity(vectors["club"], vec, dim=0).item())
            lines.append(f"    cos(club,{name})={cosine:+.4f}")
    lines.append("    club_frame_grad       mean       p50       p90       max      zero_frac")
    lines.append("    raw                " + " ".join(f"{x:9.6f}" for x in raw_stats))
    lines.append("    normalized         " + " ".join(f"{x:9.6f}" for x in normalized_stats))
    lines.append(
        f"    club_controlled_bound: raw={float(raw_club_loss.detach()):+.6f} "
        f"normalized={float(normalized_club_loss.detach()):+.6f} "
        "(forward values must match)")
    club_diag = getattr(club_module, "last_diagnostics", {})
    if club_diag:
        lines.append(
            "    club_actual_bound_batch: "
            f"logq_pos={club_diag['positive_log_q']:+.5f} "
            f"logq_neg={club_diag['negative_log_q']:+.5f} "
            f"collision={club_diag['negative_label_collision']:.3f} "
            f"pred_H={club_diag['prediction_entropy']:.4f} "
            f"class_cov={club_diag['label_class_coverage']:.4f}")
    lines.append(
        f"    club_controlled_negatives: collision={controlled_negative_collision:.3f} "
        "policy=roll(1)")

    qdiag = club_module.last_inner_diagnostics
    if qdiag:
        lines.append(
            "    q_phi same_effective_batch: "
            f"pre_ce={qdiag['pre_ce']:.5f} pre_acc={qdiag['pre_acc']:.3f} "
            f"pre_H={qdiag['pre_entropy']:.4f} post_ce={qdiag['post_ce']:.5f} "
            f"post_acc={qdiag['post_acc']:.3f} post_H={qdiag['post_entropy']:.4f} "
            f"B={int(qdiag['batch_size'])} uniq={int(qdiag['unique_classes'])} "
            f"maj={qdiag['majority_fraction']:.3f}")
        lines.append(
            "    q_phi logits: "
            f"pre_std={qdiag['pre_logit_std']:.5f} pre_absmax={qdiag['pre_logit_absmax']:.5f} "
            f"post_std={qdiag['post_logit_std']:.5f} "
            f"post_absmax={qdiag['post_logit_absmax']:.5f}")
        lines.append(
            "    q_phi optimizer: "
            f"inner_steps={int(qdiag['inner_steps'])} lr={qdiag['lr']:.3e} "
            f"last_grad={qdiag['last_grad_norm']:.5f} param={qdiag['parameter_norm']:.5f} "
            f"update={qdiag['update_norm']:.5f} update/param="
            f"{qdiag['update_norm'] / max(qdiag['parameter_norm'], 1e-12):.3e}")

    with torch.no_grad():
        valid = (torch.arange(out["z_L"].shape[1], device=out["z_L"].device).unsqueeze(0)
                 < out["out_lengths"].unsqueeze(1))
        enc_rows = model.sae.enc_weight.detach().float().norm(dim=1)
        dec_cols = model.sae.dec_weight.detach().float().norm(dim=0)
        zL = out["z_L"].detach().float()[valid]
        zP = out["z_P"].detach().float()[valid]
        poolL = _stats_pool(out["z_L"].detach().float(), out["out_lengths"])
        lines.append(
            "    geometry: "
            f"enc_row_norm={enc_rows.mean():.4f}±{enc_rows.std():.4f} "
            f"dec_col_norm={dec_cols.mean():.4f}±{dec_cols.std():.4f} "
            f"zL_rms={zL.pow(2).mean().sqrt():.5f} zP_rms={zP.pow(2).mean().sqrt():.5f} "
            f"zL_absmax={zL.abs().max():.4f} zP_absmax={zP.abs().max():.4f} "
            f"poolL_norm={poolL.norm(dim=1).mean():.4f}±{poolL.norm(dim=1).std():.4f}")
        mL, mP = out.get("m_L"), out.get("m_P")
        if mL is not None and mP is not None:
            lines.append(
                "    routing_values: "
                f"mL_mean={mL.detach().float().mean():.4f} "
                f"mL_std={mL.detach().float().std():.4f} "
                f"mL_min={mL.detach().float().min():.4f} "
                f"mL_max={mL.detach().float().max():.4f} "
                f"mP_mean={mP.detach().float().mean():.4f}")
    return lines


def _invariance_loss(zL: torch.Tensor, zLp: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Scale-normalized per-frame invariance: the FRACTION of z_L's energy that
    changes under the speaker perturbation, averaged over valid frames.

        r_t = ||z_L[t] - z_L'[t]||^2 / (0.5(||z_L[t]||^2 + ||z_L'[t]||^2) + eps)

    r in [0, ~2]: 0 = perfectly invariant, 1 = orthogonal.  Being scale-invariant
    (independent of z_L magnitude and of the K_L zero-padding), a small interpretable
    weight works — unlike the raw MSE whose K_L normalization diluted the gradient.
    """
    T = min(zL.shape[1], zLp.shape[1])
    zL, zLp = zL[:, :T], zLp[:, :T]
    diff = (zL - zLp).pow(2).sum(-1)                                  # (B, T)
    den  = 0.5 * (zL.pow(2).sum(-1) + zLp.pow(2).sum(-1)) + 1e-6
    r    = diff / den                                                 # (B, T)
    mask = (torch.arange(T, device=zL.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    return (r * mask).sum() / mask.sum().clamp(min=1)


def _interp1d(x: torch.Tensor, T: int) -> torch.Tensor:
    """Linear-resample a 1-D contour to length T."""
    L = x.shape[0]
    if L == T:
        return x
    if L < 2:
        return x.new_full((T,), float(x.mean()) if L else 0.0)
    return F.interpolate(x.view(1, 1, L), size=T, mode="linear", align_corners=True).view(T)


@torch.no_grad()
def _prosody_targets_fast(audios, audio_lengths, out_lengths, sr: int = 16_000):
    """Per-frame prosody targets aligned to the SAE frame grid, computed on the fly
    (no caching): log-F0 via torchaudio NCCF pitch (fast, batched-capable) + log
    frame-RMS energy.  Returns (f0, voiced, energy), each (B, Tmax).  F0 is raw
    (not speaker-normalized) so z_P can serve both SID and prosody."""
    import torchaudio.functional as AF
    B = audios.shape[0]
    Tmax = int(out_lengths.max().item()) if B else 0
    dev = audios.device
    f0o = torch.zeros(B, Tmax, device=dev)
    vo  = torch.zeros(B, Tmax, device=dev)
    eo  = torch.zeros(B, Tmax, device=dev)
    frame, hop = 400, 160
    # Pitch/energy extraction must run in fp32 (NCCF/FFT) — disable the bf16 autocast.
    ac_ctx = torch.autocast("cuda" if audios.is_cuda else "cpu", enabled=False)
    aud = audios.float()
    with ac_ctx:
      for i in range(B):
        n  = int(audio_lengths[i].item())
        Ti = int(out_lengths[i].item())
        if Ti <= 0 or n < frame:
            continue
        w = aud[i, :n]
        loge = w.unfold(0, frame, hop).pow(2).mean(-1).clamp_min(1e-8).log()   # (Lf,)
        try:
            f0 = AF.detect_pitch_frequency(
                w.unsqueeze(0), sr, frame_time=hop / sr,
                win_length=30, freq_low=65, freq_high=400).squeeze(0)
        except Exception:
            f0 = torch.zeros_like(loge)
        voiced = ((f0 >= 65.0) & (f0 <= 400.0)).float()
        logf0  = torch.where(voiced.bool(), f0.clamp_min(1.0).log(), torch.zeros_like(f0))
        f0o[i, :Ti] = _interp1d(logf0,  Ti)
        vo[i, :Ti]  = (_interp1d(voiced, Ti) > 0.5).float()
        eo[i, :Ti]  = _interp1d(loge,   Ti)
    return f0o, vo, eo


def _prosody_train_loss(pred, f0, voiced, energy, lengths) -> torch.Tensor:
    """Masked MSE: F0 on voiced frames, energy on all valid frames (pred: (B,T,2)).

    The padded pred frame-dim can exceed the target Tmax (=out_lengths.max()) by a
    frame, so align both to the common T (extra frames are padding, masked anyway).
    """
    T = min(pred.shape[1], f0.shape[1])
    pred = pred[:, :T]
    valid = (torch.arange(T, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    vmask = voiced[:, :T] * valid
    f0e = ((pred[..., 0] - f0[:, :T]) ** 2 * vmask).sum() / vmask.sum().clamp(min=1)
    ee  = ((pred[..., 1] - energy[:, :T]) ** 2 * valid).sum() / valid.sum().clamp(min=1)
    return f0e + ee


@torch.no_grad()
def _init_bias_geometric_median(model, batch, device, use_bf16, iters: int = 20) -> None:
    """Init b_pre to the geometric median of a batch of h_t (Gao et al. A.1)."""
    audios, audio_lengths = batch[0].to(device), batch[1].to(device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
    with ctx:
        h_t, out_lengths = model.encoder(audios, audio_lengths)
    T = h_t.shape[1]
    mask = (torch.arange(T, device=device).unsqueeze(0) < out_lengths.unsqueeze(1))
    X = h_t[mask].float()                                      # (N, D) valid frames
    gm = X.mean(0)
    for _ in range(iters):                                     # Weiszfeld iterations
        w = 1.0 / (X - gm).norm(dim=1).clamp(min=1e-6)
        gm = (X * w.unsqueeze(1)).sum(0) / w.sum()
    model.sae.b_pre.data.copy_(gm.to(model.sae.b_pre.dtype))


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

    precision = str(getattr(cfg, "precision", "auto"))
    use_bf16, use_fp16 = resolve_amp_precision(precision)
    scaler   = torch.amp.GradScaler("cuda", enabled=use_fp16)

    if getattr(cfg, 'geom_median_bias', False):
        _init_bias_geometric_median(model, next(iter(train_dl)), device, use_bf16)
        print("[stage 1] b_pre ← geometric median of a data sample")
    if model.sae.aux_k > 0:
        print(f"[stage 1] AuxK on: aux_k={model.sae.aux_k}  coef={cfg.aux_k_coef}  "
              f"dead_thresh={model.sae.dead_threshold} steps  "
              f"valid_frame_dead_count={getattr(cfg,'valid_frame_dead_count',False)}  "
              f"renorm_dec={getattr(cfg,'renorm_decoder',False)}")

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
            # Track dead latents in every SAE run; AuxK is an optional intervention.
            model.sae.update_dead(
                out["z_t"],
                out["out_lengths"] if getattr(cfg, "valid_frame_dead_count", False) else None,
            )
            # AuxK dead-latent revival (Gao): model the recon residual with dead latents.
            l_aux = None
            if model.sae.aux_k > 0:
                e_hat = model.sae.aux_reconstruct(out["z_pre"])
                if e_hat is not None:
                    resid = (out["h_t"] - out["h_hat"]).detach()         # residual target
                    l_aux = _masked_mse(resid, e_hat, out["out_lengths"])
                    loss  = loss + cfg.aux_k_coef * l_aux

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

        if getattr(cfg, 'renorm_decoder', False):
            model.sae.normalize_decoder()
        scheduler.step()

        if step % cfg.log_every == 0 or step == 1:
            lr_now  = optimizer.param_groups[0]["lr"]
            density = (out["z_pre"] > 0).float().mean().item()
            recon_v = l_recon.item()
            log_dict = {"recon": recon_v, "total": loss.item()}
            n_dead = int((model.sae.steps_since_fired > model.sae.dead_threshold).sum())
            dead_frac = n_dead / cfg.K
            log_dict["dead_frac"] = dead_frac
            dead_str = f"  dead={n_dead}/{cfg.K} ({100*dead_frac:.1f}%)"
            if l_decor is not None:
                decor_v = l_decor.item()
                log_dict["decor"]          = decor_v
                log_dict["decor_weighted"] = cfg.decor_weight * decor_v
                print(f"  step {step:>6d}/{cfg.total_steps}  recon={recon_v:.4f}  "
                      f"decor={decor_v:.4f} (w={cfg.decor_weight * decor_v:.4f})  "
                      f"total={loss.item():.4f}  lr={lr_now:.2e}{dead_str}")
            else:
                aux_str = dead_str
                if model.sae.aux_k > 0:
                    aux_v  = l_aux.item() if l_aux is not None else 0.0
                    aux_str = f"  aux={aux_v:.4f}{aux_str}"
                    log_dict["aux"] = aux_v
                print(f"  step {step:>6d}/{cfg.total_steps}  recon={recon_v:.4f}  lr={lr_now:.2e}{aux_str}")
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

def _build_dual_inv_loaders(cfg):
    """Build pair-alpha + pair-beta dataloaders for dual-invariance training.

    Returns (pa_loader, pb_loader) or (None, None) if disabled.
    """
    if not getattr(cfg, 'dual_invariance', False):
        return None, None
    from data.parallel_datasets import (ARCTICIndex, LibrispeechChapterIndex,
                                         PairAlphaDataset, PairBetaDataset, collate_pairs)
    from torch.utils.data import DataLoader

    arctic_idx = ARCTICIndex(cfg.arctic_root)
    if len(arctic_idx) == 0:
        print(f"[dual_inv] WARN: ARCTIC empty at {cfg.arctic_root} (pair-alpha will rely on perturb only)")
        arctic_idx = None

    if not getattr(cfg, 'local_data', False):
        raise RuntimeError("dual_invariance requires --local_data (LibriSpeech on disk)")
    libri_examples = _local_examples(cfg, "train.100", n=None)
    if not libri_examples:
        raise RuntimeError(f"dual_invariance: no LibriSpeech examples found under {cfg.librispeech_root}")
    libri_chapters = LibrispeechChapterIndex(libri_examples)
    if len(libri_chapters) == 0:
        raise RuntimeError("dual_invariance: LibriSpeech chapter index empty (need ≥2 utts per chapter)")

    perturb_kwargs = {
        "f0_range":     (cfg.inv_f0_low,     cfg.inv_f0_high),
        "formant_range": (cfg.inv_formant_low, cfg.inv_formant_high),
    }
    weights_alpha = {"arctic": cfg.pair_alpha_arctic_w, "perturb": cfg.pair_alpha_pert_w}
    pa_ds = PairAlphaDataset(arctic_idx, libri_examples, cfg.sample_rate,
                              weights_alpha, perturb_kwargs=perturb_kwargs,
                              rng_seed=cfg.seed, epoch_size=10**9)
    pb_ds = PairBetaDataset(libri_chapters, cfg.sample_rate,
                             rng_seed=cfg.seed + 1, epoch_size=10**9)
    pa = DataLoader(pa_ds, batch_size=cfg.pairs_alpha_per_step,
                    num_workers=cfg.num_workers, collate_fn=collate_pairs, shuffle=False)
    pb = DataLoader(pb_ds, batch_size=cfg.pairs_beta_per_step,
                    num_workers=cfg.num_workers, collate_fn=collate_pairs, shuffle=False)
    return pa, pb


def run_stage2(cfg: DISConfig, stage1_ckpt: Optional[Path]) -> Path:
    """Full disentanglement training.  Optionally loads SAE from stage1_ckpt."""
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    tokenizer, train_dl, val_dl, _test_dl = make_stage2_dataloaders(cfg)
    emo_train_dl = emo_val_dl = emo_test_dl = None
    if getattr(cfg, 'emotion', False):
        from data.iemocap_emotion import make_iemocap_emotion_dataloaders
        emo_train_dl, emo_val_dl, emo_test_dl = make_iemocap_emotion_dataloaders(cfg)

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
    if getattr(cfg, 'grl_stats_pool', False):
        extra_str += "  grl_stats_pool=True"
    if getattr(cfg, 'grl_attention_pool', False):
        extra_str += "  grl_attention_pool=True"
    if getattr(cfg, 'grl_dense_context', False):
        extra_str += f"  grl_dense_context=True(k={cfg.grl_context_kernel})"
    if getattr(cfg, 'grl_robust_sid', False):
        extra_str += f"  grl_robust_sid=True(act={cfg.grl_robust_activation})"
    if getattr(cfg, 'grl_linear_stats', False):
        extra_str += "  grl_linear_stats=True"
    if getattr(cfg, 'grl_linear_mean', False):
        extra_str += "  grl_linear_mean=True"
    if getattr(cfg, 'grl_grad_norm', False):
        extra_str += f"  grl_grad_norm={cfg.grl_grad_norm_target}"
        if float(getattr(cfg, "grl_grad_norm_final_target", -1.0)) >= 0.0:
            extra_str += (
                f"→{cfg.grl_grad_norm_final_target}"
                f"@{cfg.grl_grad_norm_decay_start}-{cfg.grl_grad_norm_decay_end}"
            )
    if getattr(cfg, 'instance_norm_zL', False):
        extra_str += "  instance_norm_zL=True"
    if getattr(cfg, 'dann_full_discriminator', False):
        extra_str += "  dann_full_disc=True"
    print(f"[stage 2] α={cfg.alpha}  β={cfg.beta}  grl={cfg.grl_weight}  ρ={cfg.rho}{delay_str}{extra_str}")
    schedule_steps = int(getattr(cfg, "stage2_schedule_steps", 0) or cfg.stage2_steps)
    if schedule_steps < cfg.stage2_steps:
        raise ValueError("stage2_schedule_steps must be 0 or >= stage2_steps")
    print(f"[stage 2] steps={cfg.stage2_steps}  schedule_steps={schedule_steps}  "
          f"batch={cfg.batch_size}  grad_clip={cfg.grad_clip}")
    print("[stage 2] best_ckpt selection: per + (1 - sid_acc) + grl_acc + "
          "(1 - grl_p_per)  — in-training head proxies, NOT a probe; "
          "run diag_probe/ for the authoritative leakage signal.")

    model.train()
    use_bf16_init = cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if getattr(cfg, 'geom_median_bias', False):
        _b = next(iter(train_dl))
        _init_bias_geometric_median(model, (_b[0], _b[1]), device, use_bf16_init)
        print("[stage 2] b_pre ← geometric median of a data sample")
    if model.sae.aux_k > 0:
        print(f"[stage 2] AuxK on: aux_k={model.sae.aux_k}  coef={cfg.aux_k_coef}  "
              f"dead_thresh={model.sae.dead_threshold}  "
              f"valid_frame_dead_count={getattr(cfg,'valid_frame_dead_count',False)}  "
              f"renorm_dec={getattr(cfg,'renorm_decoder',False)}")
    if hasattr(model, 'grl_head_u'):
        print(f"[stage 2] z_U adversaries on: grl_u={cfg.grl_u_weight}  grl_p_u={cfg.grl_phoneme_u_weight}")
    if hasattr(model, 'prosody_head'):
        print(f"[stage 2] prosody on: prosody_weight={cfg.prosody_weight}  "
              f"anti-prosody grl_L={cfg.grl_prosody_weight}  grl_U={cfg.grl_prosody_u_weight}")
    if hasattr(model, 'emotion_head'):
        print(f"[stage 2] emotion on: emotion_weight={cfg.emotion_weight}  "
              f"anti-emotion grl_L={cfg.grl_emotion_weight}  every={cfg.emotion_every} "
              f"aux_clip={cfg.emotion_aux_loss_clip}  iemocap_fold={cfg.iemocap_fold}")

    projection_mode = getattr(cfg, 'projection_disentanglement', False)

    # routing logits may be frozen (fixed_routing); filter to avoid optimizer warnings
    routing_params = ([] if projection_mode
                      else [p for p in model.routing.parameters() if p.requires_grad])
    projection_params = []
    for _m in ("proj_L", "proj_P", "up_L", "up_P", "proj_U", "up_U"):
        if hasattr(model, _m):
            projection_params.extend(getattr(model, _m).parameters())
    # Adversary discriminators get their own (optionally higher) lr so they can
    # track the moving encoder; task heads stay weak at lr_heads.
    u_adv_params = (list(model.grl_head_u.parameters()) + list(model.pr_grl_head_u.parameters())
                    if hasattr(model, 'grl_head_u') else [])
    # Prosody: task head trains at lr_heads; anti-prosody adversaries at lr_disc.
    prosody_task_params = list(model.prosody_head.parameters()) if hasattr(model, 'prosody_head') else []
    prosody_adv_params  = []
    if hasattr(model, 'prosody_grl_head'):
        prosody_adv_params += list(model.prosody_grl_head.parameters())
    if hasattr(model, 'prosody_grl_head_u'):
        prosody_adv_params += list(model.prosody_grl_head_u.parameters())
    emotion_task_params = list(model.emotion_head.parameters()) if hasattr(model, 'emotion_head') else []
    emotion_adv_params = list(model.emotion_grl_head.parameters()) if hasattr(model, 'emotion_grl_head') else []
    disc_params  = (list(model.grl_head.parameters()) + list(model.pr_grl_head.parameters())
                    + u_adv_params + prosody_adv_params + emotion_adv_params)
    lr_disc_eff  = cfg.lr_disc if getattr(cfg, 'lr_disc', 0.0) > 0 else cfg.lr_heads
    lr_sid_eff   = (cfg.lr_sid_head if getattr(cfg, 'lr_sid_head', 0.0) > 0
                    else cfg.lr_heads)
    param_groups = [
        {"params": list(model.sae.parameters()),         "lr": cfg.lr},
        {"params": routing_params,                       "lr": cfg.lr_routing},
        {"params": (list(model.pr_head.parameters()) +
                    prosody_task_params +
                    emotion_task_params),                "lr": cfg.lr_heads},
        {"params": list(model.sid_head.parameters()),    "lr": lr_sid_eff},
        {"params": disc_params,                          "lr": lr_disc_eff},
    ]
    if projection_params:
        param_groups.insert(2, {"params": projection_params, "lr": cfg.lr_heads})
    vib_params = [model.vib_logvar] if hasattr(model, 'vib_logvar') else []
    if vib_params:
        param_groups.append({"params": vib_params, "lr": cfg.lr})
    optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg.warmup_steps, schedule_steps, cfg.lr, cfg.lr_min)

    precision = str(getattr(cfg, "precision", "auto"))
    use_bf16, use_fp16 = resolve_amp_precision(precision)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    resolved_precision = "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32"
    print(f"[stage 2] precision requested={precision} resolved={resolved_precision}")

    from datetime import datetime
    tb = DISLogger(cfg.runs_dir / "tb", run_name=f"stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_metric    = float("inf")
    best_ckpt      = cfg.checkpoint_dir / "stage2_best.pt"
    train_iter     = iter(train_dl)

    # Dual-invariance pair loaders (None when disabled)
    pa_loader, pb_loader = _build_dual_inv_loaders(cfg)
    dual_inv_on    = pa_loader is not None and pb_loader is not None
    pa_iter        = iter(pa_loader) if pa_loader is not None else None
    pb_iter        = iter(pb_loader) if pb_loader is not None else None
    inv_L_w        = float(getattr(cfg, 'inv_L_weight',   0.0)) if dual_inv_on else 0.0
    inv_P_w        = float(getattr(cfg, 'inv_P_weight',   0.0)) if dual_inv_on else 0.0
    inv_var_w      = float(getattr(cfg, 'inv_var_weight', 0.0)) if dual_inv_on else 0.0
    inv_var_g      = float(getattr(cfg, 'inv_var_gamma',  1.0))
    inv_L_frames   = int(getattr(cfg, 'inv_L_interp_frames', 200))
    if dual_inv_on:
        n_routes_eff = int(getattr(cfg, 'n_routes', 3))
        hard_route   = bool(getattr(cfg, 'hard_gumbel_routing', False))
        print(f"[stage 2] dual-invariance ON: inv_L_w={inv_L_w}  inv_P_w={inv_P_w}  "
              f"inv_var_w={inv_var_w}  gamma={inv_var_g}  interp_frames={inv_L_frames}")
        print(f"[stage 2] routing: n_routes={n_routes_eff}  hard_gumbel={hard_route}  "
              f"tau {cfg.gumbel_tau_start} -> {cfg.gumbel_tau_end}")
    # ---- probe_robust: VICReg-full + CLUB MI-min init ----
    vicreg_full_on = bool(getattr(cfg, 'vicreg_full', False))
    if vicreg_full_on:
        print(f"[stage 2] VICReg-full ON: per-frame L2 inv (frame-aligned pairs only); "
              f"cov_weight={cfg.vicreg_cov_weight}")
    club_module = None
    if bool(getattr(cfg, 'club_enabled', False)):
        # mean+std pool of z_L over time -> 2*K-dim input (x-vector tradition)
        club_module = CLUBSampled(
            in_dim=2 * int(cfg.K),
            num_classes=int(cfg.num_speakers),
            hidden=int(cfg.club_hidden),
            lr=float(cfg.club_lr),
            projection_dim=int(getattr(cfg, "club_projection_dim", 0)),
        ).to(device)
        _club_gn = (f"  grad_norm_target={cfg.club_grad_norm_target}"
                    if bool(getattr(cfg, 'club_grad_norm', False)) else "  grad_norm=off")
        _club_proj = int(getattr(cfg, "club_projection_dim", 0))
        _club_proj_msg = (f"  projection_dim={_club_proj}" if _club_proj > 0
                          else "  projection=off")
        _club_warm = int(getattr(cfg, "club_warmup_steps", 0))
        _club_pre_k = int(getattr(cfg, "club_pretrain_inner_steps",
                                  cfg.club_inner_steps))
        _club_warm_msg = (
            f"  warmup_steps={_club_warm}  pretrain_inner_steps={_club_pre_k}"
            if _club_warm > 0 else "  warmup=off"
        )
        _club_noc = bool(getattr(cfg, "club_no_collision_negatives", False))
        _club_noc_msg = "  negatives=no_collision" if _club_noc else "  negatives=randperm"
        print(f"[stage 2] CLUB ON: weight={cfg.club_weight}  inner_steps={cfg.club_inner_steps}  "
              f"lr={cfg.club_lr}  hidden={cfg.club_hidden}  in_dim={2 * cfg.K}  "
              f"num_speakers={cfg.num_speakers}  pool=mean+std(z_L)"
              f"{_club_gn}{_club_proj_msg}{_club_warm_msg}{_club_noc_msg}")
    club_full_diagnostics = bool(getattr(cfg, "club_full_diagnostics", False))
    club_diagnostics_every = int(getattr(cfg, "club_diagnostics_every", 100))
    if club_full_diagnostics:
        if club_module is None:
            raise ValueError("club_full_diagnostics requires club_enabled")
        if club_diagnostics_every <= 0:
            raise ValueError("club_diagnostics_every must be positive")
        print("[stage 2] CLUB FULL DIAGNOSTICS ON: "
              f"every={club_diagnostics_every} optimizer steps; expensive; "
              "q_phi pre/post, raw/delivered frame gradients, objective encoder/routing "
              "gradients, clipping, optimizer groups, geometry")

    # ---- probe_robust: phoneme CLUB on z_P (frame-level) ----
    club_phn_module = None
    if bool(getattr(cfg, 'club_phoneme_enabled', False)):
        # Per-frame z_P -> K-dim input. Targets are pr_head argmax (pseudo
        # labels). 74 phoneme classes by default. Warmup gates the loss until
        # pr_head is no longer random.
        club_phn_module = CLUBSampled(
            in_dim=int(cfg.K),
            num_classes=int(cfg.vocab_size),
            hidden=int(cfg.club_phoneme_hidden),
            lr=float(cfg.club_phoneme_lr),
        ).to(device)
        print(f"[stage 2] CLUB-phn ON: weight={cfg.club_phoneme_weight}  "
              f"inner_steps={cfg.club_phoneme_inner_steps}  "
              f"lr={cfg.club_phoneme_lr}  hidden={cfg.club_phoneme_hidden}  "
              f"in_dim={cfg.K}  num_classes={cfg.vocab_size}  "
              f"warmup_steps={cfg.club_phoneme_warmup_steps}  "
              f"target=pr_head.argmax  pool=per-frame(z_P)")

    # Restore only after auxiliary CLUB estimators exist. Legacy checkpoints are
    # intentionally inference/stage-init only; exact continuation requires v2.
    start_step = 0
    resume_value = str(getattr(cfg, "resume", "none"))
    resume_path = cfg.checkpoint_dir / "latest-resume.pt" if resume_value == "auto" else Path(resume_value)
    if resume_value not in {"none", ""} and resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=False)
        validate_resume(saved, dataset_hash=str(getattr(cfg, "dataset_fingerprint", "")),
                        preset=str(getattr(cfg, "experiment_preset", "")), cfg=cfg)
        start_step, best_metric = restore_training_state(
            saved, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        aux = saved.get("auxiliary", {})
        for module, key in ((club_module, "club"), (club_phn_module, "club_phoneme")):
            if module is not None:
                if key not in aux:
                    raise ValueError(f"resume checkpoint is missing required {key} estimator state")
                module.load_state_dict(aux[key]["model"])
                module.optimizer.load_state_dict(aux[key]["optimizer"])
        if aux.get("train_sampler") and hasattr(train_dl.sampler, "load_state_dict"):
            train_dl.sampler.load_state_dict(aux["train_sampler"])
        if pa_loader is not None and aux.get("pair_alpha_rng"):
            pa_loader.dataset.rng.setstate(aux["pair_alpha_rng"])
        if pb_loader is not None and aux.get("pair_beta_rng"):
            pb_loader.dataset.rng.setstate(aux["pair_beta_rng"])
        train_iter = iter(train_dl)
        pa_iter = iter(pa_loader) if pa_loader is not None else None
        pb_iter = iter(pb_loader) if pb_loader is not None else None
        print(f"[stage 2] exact resume from {resume_path}: step={start_step} best={best_metric:.4f}")
    elif resume_value not in {"none", "", "auto"}:
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    freeze_learned_routing = bool(
        getattr(cfg, "freeze_learned_routing_on_resume", False))
    if freeze_learned_routing:
        if start_step <= 0:
            raise ValueError(
                "freeze_learned_routing_on_resume requires a restored stage-2 resume checkpoint")
        if (getattr(cfg, "no_routing", False)
                or getattr(cfg, "fixed_blocks", False)
                or getattr(cfg, "fixed_routing", False)
                or projection_mode):
            raise ValueError(
                "freeze_learned_routing_on_resume requires ordinary learned routing "
                "(not fixed blocks, fixed routing, no-routing, or projection mode)")
        model.routing.freeze_learned_routing()
        n_l, n_p, n_u = model.routing.hard_counts
        print("[stage 2] learned routing FROZEN after resume: "
              f"step={start_step} deterministic=True hard_counts={n_l}/{n_p}/{n_u} "
              "routing_optimizer_updates=off")
        if bool(getattr(cfg, "freeze_route_topk_on_resume", False)):
            if bool(getattr(model.sae, "route_topk_enabled").item()):
                quotas = getattr(model.sae, "route_topk_quotas").detach().cpu().tolist()
                print("[stage 2] learned-route TopK already enabled from checkpoint: "
                      f"quotas={quotas}")
            else:
                _calibrate_route_topk_quotas(model, train_dl, device, cfg, use_bf16)
                # The calibration intentionally restores sampler state; refresh
                # the iterator so continuation starts cleanly from that state.
                train_iter = iter(train_dl)
    reset_requested = str(getattr(cfg, "reset_adversary_heads_on_resume", "")).strip()
    if bool(getattr(cfg, "reset_adversaries_on_resume", False)) or reset_requested:
        if start_step <= 0:
            raise ValueError(
                "adversary-head reset on resume requires a restored stage-2 resume checkpoint")
        _reset_adversary_heads_after_resume(model, optimizer, reset_requested)
    segment = SegmentLimit(start_step, int(getattr(cfg, "segment_steps", 0)),
                           float(getattr(cfg, "max_runtime_minutes", 0.0)))
    metrics_path = cfg.checkpoint_dir / "metrics.jsonl"
    accumulation = max(1, int(getattr(cfg, "gradient_accumulation_steps", 1)))
    if accumulation > 1:
        print(f"[stage 2] microbatch={cfg.batch_size} accumulation={accumulation} "
              f"effective_batch={cfg.batch_size * accumulation}")
    no_routing     = getattr(cfg, 'no_routing', False)
    fixed_blocks   = getattr(cfg, 'fixed_blocks', False)
    n_routes       = getattr(cfg, 'n_routes', 3)
    routing_active = (not no_routing and not projection_mode and not fixed_blocks
                      and not freeze_learned_routing and bool(routing_params))
    grl_p_weight   = getattr(cfg, 'grl_phoneme_weight', 0.0)
    dann_fix       = getattr(cfg, 'dann_full_discriminator', False)
    n_disc_steps   = max(1, int(getattr(cfg, 'n_disc_steps', 1)))
    vib_w          = getattr(cfg, 'vib_zL_weight', 0.0)
    vib_ramp_end   = getattr(cfg, 'vib_zL_ramp_end', 0)
    grl_u_weight   = getattr(cfg, 'grl_u_weight', 0.0)          # speaker adv on z_U
    grl_p_u_weight = getattr(cfg, 'grl_phoneme_u_weight', 0.0)  # phoneme adv on z_U
    invariance_on  = getattr(cfg, 'invariance', False)
    inv_w          = getattr(cfg, 'inv_weight', 0.0)
    inv_ramp_end   = getattr(cfg, 'inv_ramp_end', 0)
    if invariance_on:
        print(f"[stage 2] invariance ON: inv_weight={inv_w}  ramp_end={inv_ramp_end} "
              f"(z_L speaker-invariance via perturbed-pair consistency)")
    shuffle_grl_labels = bool(getattr(cfg, 'shuffle_grl_speaker_labels', False))
    if shuffle_grl_labels:
        print("[stage 2] NEGATIVE CONTROL: speaker adversaries use deterministic "
              "random targets resampled each batch; z_P SID uses true labels")
    prosody_on     = hasattr(model, 'prosody_head')
    prosody_w      = getattr(cfg, 'prosody_weight', 0.0)            # z_P prosody task
    grl_pros_w     = getattr(cfg, 'grl_prosody_weight', 0.0)        # anti-prosody on z_L
    grl_pros_u_w   = getattr(cfg, 'grl_prosody_u_weight', 0.0)      # anti-prosody on z_U
    emotion_on     = hasattr(model, 'emotion_head') and emo_train_dl is not None
    emotion_w      = getattr(cfg, 'emotion_weight', 0.0)
    grl_emo_w      = getattr(cfg, 'grl_emotion_weight', 0.0)
    emotion_every  = max(1, int(getattr(cfg, 'emotion_every', 8)))
    emotion_grl_ramp_end = int(getattr(cfg, 'emotion_grl_ramp_end', 0))
    emotion_aux_clip = float(getattr(cfg, 'emotion_aux_loss_clip', 0.0))
    emo_train_iter = iter(emo_train_dl) if emotion_on else None
    aux_k_on       = model.sae.aux_k > 0
    ub_w           = getattr(cfg, 'ub_weight', 0.0)
    ub_ramp_start  = getattr(cfg, 'ub_ramp_start', 0)
    ub_ramp_end    = getattr(cfg, 'ub_ramp_end', 0)
    u_l2_w         = getattr(cfg, 'projection_u_l2', 0.0)   # L2 penalty on residual z_U
    spec_w         = getattr(cfg, 'routing_spec_weight', 0.0)  # per-unit routing specialization
    routing_clip_params = [] if projection_mode else list(model.routing.parameters())
    all_params     = (list(model.sae.parameters()) + routing_clip_params + projection_params +
                      list(model.pr_head.parameters()) + list(model.sid_head.parameters()) +
                      list(model.grl_head.parameters()) + list(model.pr_grl_head.parameters()) +
                      u_adv_params + prosody_task_params + prosody_adv_params +
                      emotion_task_params + emotion_adv_params +
                      ([model.vib_logvar] if hasattr(model, 'vib_logvar') else []))

    # Task-level adversarial clipping operates only where utility and adversary
    # gradients actually meet.  Exclude the decoder and adversary heads: the
    # decoder receives no GRL gradient, while discriminator learning must remain
    # full strength.  Include b_pre because it participates in SAE encoding.
    adversarial_cap_on = bool(getattr(cfg, 'adversarial_task_grad_cap', False))
    adversarial_cap_ratios = {
        'grl': float(getattr(cfg, 'grl_shared_grad_cap_ratio', 2.0)),
        'grl_p': float(getattr(cfg, 'grl_p_shared_grad_cap_ratio', 1.0)),
    }
    if adversarial_cap_on and any(value <= 0 for value in adversarial_cap_ratios.values()):
        raise ValueError("adversarial shared-gradient cap ratios must be positive")
    _cap_candidates = [model.sae.enc_weight, model.sae.b_pre] + routing_clip_params + projection_params
    adversarial_cap_params = []
    _cap_seen = set()
    for _param in _cap_candidates:
        if _param.requires_grad and id(_param) not in _cap_seen:
            adversarial_cap_params.append(_param)
            _cap_seen.add(id(_param))
    if adversarial_cap_on:
        print("[stage 2] adversarial task-gradient cap ON: "
              f"speaker<={adversarial_cap_ratios['grl']:.2f}x reference  "
              f"phoneme<={adversarial_cap_ratios['grl_p']:.2f}x reference  "
              "scope=shared SAE encoder/routing; discriminator heads uncapped")

    # ---- GradNorm: learn the managed task weights online (replaces fixed weights) ----
    gradnorm_on = bool(getattr(cfg, 'gradnorm', False))
    gn_every    = max(1, int(getattr(cfg, 'gradnorm_every', 1)))
    gn_ctrl     = None
    if gradnorm_on:
        from gradnorm import GradNormController
        gn_names = [t.strip() for t in getattr(cfg, 'gradnorm_tasks', 'recon,pr,sid').split(',') if t.strip()]
        gn_ctrl  = GradNormController(gn_names, model.sae.enc_weight,
                                      alpha=cfg.gradnorm_alpha, lr=cfg.gradnorm_lr, device=str(device))
        print(f"[stage 2] GradNorm ON: tasks={gn_names}  alpha={cfg.gradnorm_alpha}  "
              f"lr={cfg.gradnorm_lr}  every={gn_every}  shared=sae.enc_weight")

    first_micro = start_step * accumulation + 1
    final_micro = cfg.stage2_steps * accumulation
    club_pool_buffer = []
    club_label_buffer = []
    club_phn_pool_buffer = []
    club_phn_label_buffer = []
    discriminator_buffer = []
    adversarial_grad_buffers = {}
    last_adversarial_cap_stats = {}
    for micro_step in range(first_micro, final_micro + 1):
        step = (micro_step - 1) // accumulation + 1
        micro_in_group = (micro_step - 1) % accumulation
        accumulation_boundary = (micro_in_group == accumulation - 1)
        club_diagnostics_due = (club_full_diagnostics and accumulation_boundary
                                and (step == 1 or step % club_diagnostics_every == 0))
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        # ---- temperature + DANN ramp
        model.routing.tau = _gumbel_tau(step, schedule_steps,
                                        cfg.gumbel_tau_start, cfg.gumbel_tau_end)
        # Delayed linear ramp for the IB capacity penalty (specialize first, then prune).
        if ub_w > 0 and ub_ramp_end > ub_ramp_start:
            eff_ub_w = ub_w * min(1.0, max(0.0, (step - ub_ramp_start) / (ub_ramp_end - ub_ramp_start)))
        else:
            eff_ub_w = ub_w
        # VIB KL ramp (let z_L form before compressing)
        eff_vib_w = (vib_w * min(1.0, step / vib_ramp_end)) if (vib_w > 0 and vib_ramp_end > 0) else vib_w
        grl_active        = (cfg.grl_delay_steps == 0 or step >= cfg.grl_delay_steps)
        ramp              = (_dann_lambda(step, schedule_steps, getattr(cfg, "dann_ramp_steps", 0))
                             if grl_active else 0.0)
        grl_target_scale  = (_scheduled_grl_lambda_scale(cfg, step)
                             if getattr(cfg, "grl_grad_norm", False) else 1.0)
        if dann_fix:
            # Canonical DANN: heads train at full strength; the per-adversary
            # weights act only on the reversed (encoder-side) gradient via lambda.
            grl_lam          = cfg.grl_weight * ramp * grl_target_scale
            grl_p_lam        = grl_p_weight * ramp
            eff_grl_weight   = 1.0
            eff_grl_p_weight = 1.0 if grl_p_weight > 0 else 0.0
        else:
            grl_lam          = ramp * grl_target_scale
            grl_p_lam        = None
            eff_grl_weight   = cfg.grl_weight if grl_active else 0.0
            eff_grl_p_weight = grl_p_weight
        # z_U adversaries (anti-speaker + anti-phoneme): reversal ramps with the rest;
        # discriminators train at full strength (dann), so eff weight on their loss = 1.
        grl_u_lam   = grl_u_weight   * ramp
        grl_p_u_lam = grl_p_u_weight * ramp
        # anti-prosody adversaries (z_L / z_U): reversal ramps with the rest
        grl_pros_lam   = grl_pros_w   * ramp
        # Invariance weight ramp (let z_L form content before stripping speaker)
        eff_inv_w = (inv_w * min(1.0, step / inv_ramp_end)) if (invariance_on and inv_ramp_end > 0) else inv_w
        grl_pros_u_lam = grl_pros_u_w * ramp
        emo_grl_ramp = (min(1.0, step / max(1, emotion_grl_ramp_end))
                        if emotion_grl_ramp_end > 0 else 1.0)
        grl_emo_lam = grl_emo_w * ramp * emo_grl_ramp
        run_emotion_aux = emotion_on and (step % emotion_every == 0)

        if accumulation_boundary and step % cfg.grad_log_every == 0:
            _log_grad_norms_stage2(model, batch, cfg, step, tb, use_bf16, grl_lam,
                                   eff_grl_weight, grl_p_lam, shuffle_grl_labels,
                                   club_module=club_module,
                                   club_phn_module=club_phn_module)

        audios, audio_lengths, targets, target_lengths, speaker_ids = batch[:5]
        pert_audios    = batch[5].to(device, non_blocking=True) if len(batch) > 5 else None
        audios         = audios.to(device, non_blocking=True)
        audio_lengths  = audio_lengths.to(device, non_blocking=True)
        targets        = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        speaker_ids    = speaker_ids.to(device, non_blocking=True)
        adversary_speaker_ids = (
            _random_speaker_targets(speaker_ids, cfg.num_speakers, cfg.seed, step)
            if shuffle_grl_labels else speaker_ids
        )
        emo_audios = emo_lengths = emo_labels = None
        if run_emotion_aux:
            try:
                emo_batch = next(emo_train_iter)
            except StopIteration:
                emo_train_iter = iter(emo_train_dl)
                emo_batch = next(emo_train_iter)
            emo_audios, emo_lengths, emo_labels = emo_batch
            emo_audios = emo_audios.to(device, non_blocking=True)
            emo_lengths = emo_lengths.to(device, non_blocking=True)
            emo_labels = emo_labels.to(device, non_blocking=True)

        if micro_in_group == 0:
            optimizer.zero_grad(set_to_none=True)
            adversarial_grad_buffers = {}
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else
               torch.autocast("cuda", dtype=torch.float16) if use_fp16 else
               torch.autocast("cuda", enabled=False))
        with ctx:
            out     = model(audios, audio_lengths, stage=2, grl_lambda=grl_lam,
                            grl_p_lambda=grl_p_lam,
                            grl_u_lambda=grl_u_lam, grl_p_u_lambda=grl_p_u_lam,
                            grl_prosody_lambda=grl_pros_lam,
                            grl_prosody_u_lambda=grl_pros_u_lam,
                            emit_emotion=False)
            # Invariance: z_L of the speaker-perturbed copy must match z_L of the
            # original (frame-aligned) — a dense per-frame speaker-removal signal.
            l_inv = out["z_L"].new_zeros(())
            if invariance_on and pert_audios is not None:
                out_p = model(pert_audios, audio_lengths, stage=2, grl_lambda=0.0,
                              emit_emotion=False)
                l_inv = _invariance_loss(out["z_L"], out_p["z_L"], out["out_lengths"])
            l_recon = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
            l_pr    = ctc_pr_loss(out["pr_logits"], targets, out["out_lengths"], target_lengths)
            l_sid   = sid_ce_loss(out["sid_logits"], speaker_ids)
            l_grl   = _speaker_adv_loss(out["grl_logits"], adversary_speaker_ids, out["out_lengths"])
            l_route    = (route_loss(out["routing_logits"])
                          if routing_active else l_recon.new_zeros(()))
            # Per-unit specialization: minimise mean unit routing entropy (with
            # route_loss this maximises MI(feature; route) — decisive + balanced).
            l_spec     = (routing_spec_loss(out["routing_logits"])
                          if (routing_active and spec_w > 0) else l_recon.new_zeros(()))
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
            # VIB KL on z_L
            l_vib      = (out["vib_kl"] if "vib_kl" in out else l_recon.new_zeros(()))
            # z_U adversaries: anti-speaker + anti-phoneme (dann → eff weight 1.0)
            l_grl_u    = (_speaker_adv_loss(out["grl_u_logits"], adversary_speaker_ids, out["out_lengths"])
                          if (grl_u_weight > 0 and "grl_u_logits" in out)
                          else l_recon.new_zeros(()))
            l_grl_p_u  = (ctc_pr_loss(out["pr_grl_u_logits"], targets, out["out_lengths"], target_lengths)
                          if (grl_p_u_weight > 0 and "pr_grl_u_logits" in out)
                          else l_recon.new_zeros(()))
            eff_grl_u_w   = 1.0 if grl_u_weight   > 0 else 0.0
            eff_grl_p_u_w = 1.0 if grl_p_u_weight > 0 else 0.0
            # Prosody: per-frame F0/energy regression on z_P + anti-prosody adversaries.
            l_pros = l_pros_grl = l_pros_grl_u = l_recon.new_zeros(())
            if prosody_on and "prosody_pred" in out:
                p_f0, p_v, p_e = _prosody_targets_fast(audios, audio_lengths, out["out_lengths"])
                l_pros = _prosody_train_loss(out["prosody_pred"], p_f0, p_v, p_e, out["out_lengths"])
                if "prosody_grl_pred" in out:
                    l_pros_grl = _prosody_train_loss(out["prosody_grl_pred"], p_f0, p_v, p_e, out["out_lengths"])
                if "prosody_grl_u_pred" in out:
                    l_pros_grl_u = _prosody_train_loss(out["prosody_grl_u_pred"], p_f0, p_v, p_e, out["out_lengths"])
            eff_grl_pros_w   = 1.0 if grl_pros_w   > 0 else 0.0
            eff_grl_pros_u_w = 1.0 if grl_pros_u_w > 0 else 0.0
            # Track dead latents regardless of whether AuxK revival is enabled.
            model.sae.update_dead(
                out["z_t"],
                out["out_lengths"] if getattr(cfg, "valid_frame_dead_count", False) else None,
            )
            # AuxK dead-latent revival (Gao): model the recon residual with dead latents
            l_aux = l_recon.new_zeros(())
            if aux_k_on:
                e_hat = model.sae.aux_reconstruct(out["z_pre"])
                if e_hat is not None:
                    resid = (out["h_t"] - out["h_hat"]).detach()
                    l_aux = _masked_mse(resid, e_hat, out["out_lengths"])
            # GradNorm-managed weights override the fixed ones for listed tasks;
            # unmanaged tasks keep their cfg/eff weights.
            if gradnorm_on:
                _gw = gn_ctrl.weights()
                m_recon  = _gw.get('recon',   1.0)
                m_pr     = _gw.get('pr',      cfg.alpha)
                m_sid    = _gw.get('sid',     cfg.beta)
                m_grl    = _gw.get('grl',     eff_grl_weight)
                m_grl_p  = _gw.get('grl_p',   eff_grl_p_weight)
                m_grl_u  = _gw.get('grl_u',   eff_grl_u_w)
                m_grl_pu = _gw.get('grl_p_u', eff_grl_p_u_w)
                m_aux    = _gw.get('aux',     cfg.aux_k_coef)
            else:
                m_recon, m_pr, m_sid = 1.0, cfg.alpha, cfg.beta
                m_grl, m_grl_p, m_grl_u, m_grl_pu = (eff_grl_weight, eff_grl_p_weight,
                                                     eff_grl_u_w, eff_grl_p_u_w)
                m_aux = cfg.aux_k_coef
            # ---- Dual-invariance: pair-alpha (L) + pair-beta (P) + variance floor ----
            l_inv_L = l_recon.new_zeros(())
            l_inv_P = l_recon.new_zeros(())
            l_var   = l_recon.new_zeros(())
            if dual_inv_on:
                try:
                    pa = next(pa_iter)
                except StopIteration:
                    pa_iter = iter(pa_loader); pa = next(pa_iter)
                try:
                    pb = next(pb_iter)
                except StopIteration:
                    pb_iter = iter(pb_loader); pb = next(pb_iter)
                # Pair α (z_L invariance)
                out_pa_a = model(pa["audio_a"].to(device, non_blocking=True),
                                  pa["len_a"].to(device,   non_blocking=True),
                                  stage=2, grl_lambda=0.0, emit_emotion=False)
                out_pa_b = model(pa["audio_b"].to(device, non_blocking=True),
                                  pa["len_b"].to(device,   non_blocking=True),
                                  stage=2, grl_lambda=0.0, emit_emotion=False)
                if vicreg_full_on:
                    # VICReg per-frame L2 — assumes frame-aligned pairs (use only
                    # with pair_alpha_pert_w=1.0, no ARCTIC: ARCTIC pairs would
                    # require time-alignment we deliberately removed).
                    l_inv_L = vicreg_invariance_loss(
                        out_pa_a["z_L"], out_pa_b["z_L"],
                        out_pa_a["out_lengths"], out_pa_b["out_lengths"],
                    )
                else:
                    l_inv_L = inv_L_frame_cosine_loss(
                        out_pa_a["z_L"], out_pa_a["out_lengths"],
                        out_pa_b["z_L"], out_pa_b["out_lengths"],
                        target_frames=inv_L_frames,
                    )
                # Pair β (z_P invariance)
                out_pb_a = model(pb["audio_a"].to(device, non_blocking=True),
                                  pb["len_a"].to(device,   non_blocking=True),
                                  stage=2, grl_lambda=0.0, emit_emotion=False)
                out_pb_b = model(pb["audio_b"].to(device, non_blocking=True),
                                  pb["len_b"].to(device,   non_blocking=True),
                                  stage=2, grl_lambda=0.0, emit_emotion=False)
                l_inv_P = inv_P_stats_pool_loss(
                    out_pb_a["z_P"], out_pb_a["out_lengths"],
                    out_pb_b["z_P"], out_pb_b["out_lengths"],
                )
                # Variance floor on main-batch z_L, z_P, weighted by routing mask
                # so we only penalise the dims the router put in this bucket
                # (otherwise hard routing makes ~half of dims mechanically zero
                # and the loss falsely flags collapse).
                if inv_var_w > 0:
                    _mL = out.get("m_L"); _mP = out.get("m_P")
                    l_var = (variance_floor_loss(out["z_L"], out["out_lengths"],
                                                  gamma=inv_var_g, weight=_mL) +
                             variance_floor_loss(out["z_P"], out["out_lengths"],
                                                  gamma=inv_var_g, weight=_mP))

            # ---- probe_robust: VICReg covariance regulariser on bucket dims ----
            l_cov = l_recon.new_zeros(())
            if vicreg_full_on and dual_inv_on and float(cfg.vicreg_cov_weight) > 0:
                _mL = out.get("m_L"); _mP = out.get("m_P")
                l_cov = (vicreg_covariance_loss(out["z_L"], out["out_lengths"], mask_dim=_mL) +
                         vicreg_covariance_loss(out["z_P"], out["out_lengths"], mask_dim=_mP))

            # ---- probe_robust: CLUB MI-min on (stats_pool(z_L), speaker_id) ----
            # bf16 throughout (matches the rest of the training pass — no fp32
            # island, no precision-mismatch artifacts). The q_phi LayerNorm at
            # the input normalises z_pool to unit variance per-example so that
            # default-init Linear pre-activations land at O(1) rather than
            # O(1e-3); that's what makes bf16 OK for this head despite its
            # 10240-d sparse input.
            l_club = l_recon.new_zeros(())
            raw_club_diag_loss = l_recon.new_zeros(())
            normalized_club_diag_loss = l_recon.new_zeros(())
            controlled_negative_collision = float("nan")
            club_ce = float('nan')
            club_acc = float('nan')
            eff_club_w, eff_club_inner = _effective_club_scaling(cfg, step)
            if club_module is not None:
                _zL_raw = out["z_L"]
                _zL = _zL_raw
                if bool(getattr(cfg, 'club_grad_norm', False)):
                    # This branch is used only by CLUB. Other objectives retain
                    # the original z_L and therefore keep their own gradients.
                    _amp_scale = float(scaler.get_scale()) if use_fp16 else 1.0
                    _zL = normalize_club_gradient(
                        _zL,
                        target=float(cfg.club_grad_norm_target),
                        weight=eff_club_w,
                        accumulation=accumulation,
                        amp_scale=_amp_scale,
                    )
                _olen = out["out_lengths"]
                _z_pool = _stats_pool(_zL, _olen)
                club_pool_buffer.append(_z_pool.detach())
                club_label_buffer.append(speaker_ids.detach())
                if accumulation_boundary:
                    club_ce, club_acc = club_module.inner_step(
                        torch.cat(club_pool_buffer), torch.cat(club_label_buffer),
                        k=eff_club_inner,
                        capture_diagnostics=club_diagnostics_due,
                    )
                if bool(getattr(cfg, "club_no_collision_negatives", False)):
                    _neg_perm = no_collision_permutation(speaker_ids)
                    l_club = club_module.mi_bound(
                        _z_pool, speaker_ids,
                        negative_labels=speaker_ids[_neg_perm],
                    )
                else:
                    l_club = club_module.mi_bound(_z_pool, speaker_ids)
                if club_diagnostics_due:
                    # Fixed negatives make raw-vs-normalized gradients exactly
                    # comparable and do not perturb the training RNG stream.
                    _negative_labels = speaker_ids.roll(1)
                    controlled_negative_collision = float(
                        (_negative_labels == speaker_ids).float().mean().item())
                    _raw_pool = _stats_pool(_zL_raw, _olen)
                    _raw_bound = club_module.mi_bound(
                        _raw_pool, speaker_ids, negative_labels=_negative_labels,
                        update_diagnostics=False)
                    _diag_normalized_z = normalize_club_gradient(
                        _zL_raw,
                        target=float(cfg.club_grad_norm_target),
                        weight=eff_club_w,
                        accumulation=accumulation,
                        # This separate autograd probe is not GradScaler-scaled;
                        # report the effective post-unscale gradient.
                        amp_scale=1.0,
                    )
                    _normalized_pool = _stats_pool(_diag_normalized_z, _olen)
                    _normalized_bound = club_module.mi_bound(
                        _normalized_pool, speaker_ids, negative_labels=_negative_labels,
                        update_diagnostics=False)
                    raw_club_diag_loss = (eff_club_w * _raw_bound / accumulation)
                    normalized_club_diag_loss = (
                        eff_club_w * _normalized_bound / accumulation)

            # ---- probe_robust: phoneme CLUB MI-min on (z_P per-frame, phoneme) ----
            # Symmetric to the speaker CLUB but operates frame-wise: input is
            # the raw per-frame z_P (no pooling — phoneme labels are inherently
            # frame-level), targets are the pr_head's argmax frame predictions
            # used as pseudo-labels (we have CTC sequence targets, not forced
            # frame alignments; pr_head's own argmax is the closest tractable
            # frame label and aligns with what a phoneme probe would extract).
            # Warmup gate: pr_head needs to stabilise before its argmax is
            # meaningful — until then, q_phi_phn would be chasing random labels.
            l_club_phn = l_recon.new_zeros(())
            club_phn_ce  = float('nan')
            club_phn_acc = float('nan')
            club_phn_pseudo_entropy = float('nan')
            club_phn_pseudo_confidence = float('nan')
            club_phn_pseudo_coverage = float('nan')
            if (club_phn_module is not None
                    and step >= int(cfg.club_phoneme_warmup_steps)):
                _zP = out["z_P"]
                _olen2 = out["out_lengths"]
                _Bp, _Tp, _Kp = _zP.shape
                _fm2 = (torch.arange(_Tp, device=device).unsqueeze(0)
                        < _olen2.unsqueeze(1))                      # (B, T) bool
                _zP_flat = _zP[_fm2]                                # (N_valid, K)
                with torch.no_grad():
                    _pr_probs = out["pr_logits"].float().softmax(dim=-1)
                    _phn_pseudo = _pr_probs.argmax(dim=-1)          # (B, T)
                _y_phn = _phn_pseudo[_fm2]                          # (N_valid,)
                with torch.no_grad():
                    _valid_probs = _pr_probs[_fm2]
                    club_phn_pseudo_entropy = float(
                        -(_valid_probs * _valid_probs.clamp_min(1e-12).log()).sum(-1).mean())
                    club_phn_pseudo_confidence = float(_valid_probs.max(-1).values.mean())
                    club_phn_pseudo_coverage = float(
                        _y_phn.unique().numel() / max(1, int(cfg.vocab_size)))
                # Bound estimator memory deterministically while preserving the
                # class/time distribution across the valid frame sequence.
                _cap = max(1, 8192 // accumulation)
                if _zP_flat.shape[0] > _cap:
                    _pick = torch.linspace(0, _zP_flat.shape[0] - 1, _cap,
                                           device=device).long()
                    _q_z, _q_y = _zP_flat[_pick], _y_phn[_pick]
                else:
                    _q_z, _q_y = _zP_flat, _y_phn
                club_phn_pool_buffer.append(_q_z.detach())
                club_phn_label_buffer.append(_q_y.detach())
                if accumulation_boundary:
                    club_phn_ce, club_phn_acc = club_phn_module.inner_step(
                        torch.cat(club_phn_pool_buffer), torch.cat(club_phn_label_buffer),
                        k=int(cfg.club_phoneme_inner_steps),
                    )
                l_club_phn = club_phn_module.mi_bound(_zP_flat, _y_phn)

            # ---- IEMOCAP auxiliary emotion/prosody batch (8 Libri : 1 IEMOCAP by default) ----
            l_emo = l_emo_grl = l_emo_pros = l_emo_pros_grl = l_recon.new_zeros(())
            l_emo_aux_raw = l_emo_aux = l_recon.new_zeros(())
            emo_aux_scale = l_recon.detach().new_tensor(1.0)
            emo_acc = emo_grl_acc = None
            eff_grl_emo_w = 1.0 if grl_emo_w > 0 else 0.0
            if run_emotion_aux and emo_audios is not None:
                out_emo = model(
                    emo_audios, emo_lengths, stage=2,
                    grl_lambda=0.0,
                    grl_p_lambda=0.0,
                    grl_u_lambda=0.0,
                    grl_p_u_lambda=0.0,
                    grl_prosody_lambda=grl_pros_lam,
                    grl_prosody_u_lambda=0.0,
                    grl_emotion_lambda=grl_emo_lam,
                    emit_emotion=True,
                )
                l_emo = F.cross_entropy(out_emo["emotion_logits"], emo_labels)
                ec, et = _class_correct(out_emo["emotion_logits"], emo_labels)
                emo_acc = ec / max(et, 1)
                if "emotion_grl_logits" in out_emo:
                    l_emo_grl = F.cross_entropy(out_emo["emotion_grl_logits"], emo_labels)
                    gc, gt = _class_correct(out_emo["emotion_grl_logits"], emo_labels)
                    emo_grl_acc = gc / max(gt, 1)
                if prosody_on and "prosody_pred" in out_emo:
                    e_f0, e_v, e_e = _prosody_targets_fast(
                        emo_audios, emo_lengths, out_emo["out_lengths"])
                    l_emo_pros = _prosody_train_loss(
                        out_emo["prosody_pred"], e_f0, e_v, e_e, out_emo["out_lengths"])
                    if "prosody_grl_pred" in out_emo:
                        l_emo_pros_grl = _prosody_train_loss(
                            out_emo["prosody_grl_pred"], e_f0, e_v, e_e, out_emo["out_lengths"])
                l_emo_aux_raw = (
                    emotion_w       * l_emo
                    + eff_grl_emo_w * l_emo_grl
                    + prosody_w     * l_emo_pros
                    + eff_grl_pros_w * l_emo_pros_grl
                )
                l_emo_aux, emo_aux_scale = _cap_loss_by_scaling(l_emo_aux_raw, emotion_aux_clip)

            total      = (m_recon            * l_recon
                          + m_pr             * l_pr
                          + m_sid            * l_sid
                          + m_grl            * l_grl
                          + m_grl_p          * l_grl_p
                          + m_grl_u          * l_grl_u
                          + m_grl_pu         * l_grl_p_u
                          + prosody_w        * l_pros
                          + eff_grl_pros_w   * l_pros_grl
                          + eff_grl_pros_u_w * l_pros_grl_u
                          + eff_inv_w        * l_inv
                          + inv_L_w          * l_inv_L
                          + inv_P_w          * l_inv_P
                          + inv_var_w        * l_var
                          + float(cfg.vicreg_cov_weight) * l_cov
                          + eff_club_w                   * l_club
                          + float(cfg.club_phoneme_weight) * l_club_phn
                          + l_emo_aux
                          + cfg.rho          * l_route
                          + spec_w           * l_spec
                          + eff_ub_w         * l_ub
                          + u_l2_w           * l_u
                          + eff_vib_w        * l_vib
                          + m_aux            * l_aux)

        full_diagnostic_lines = []
        if club_diagnostics_due:
            _objective_terms = {
                "recon": m_recon * l_recon / accumulation,
                "pr": m_pr * l_pr / accumulation,
                "sid": m_sid * l_sid / accumulation,
                "grl_p": m_grl_p * l_grl_p / accumulation,
                "inv_L": inv_L_w * l_inv_L / accumulation,
                "inv_P": inv_P_w * l_inv_P / accumulation,
                "variance": inv_var_w * l_var / accumulation,
                "covariance": float(cfg.vicreg_cov_weight) * l_cov / accumulation,
                "club": eff_club_w * l_club / accumulation,
                "route_balance": cfg.rho * l_route / accumulation,
                "route_specialize": spec_w * l_spec / accumulation,
            }
            full_diagnostic_lines = _club_full_diagnostics(
                model=model, step=step, objective_terms=_objective_terms,
                routing_params=routing_params, out=out, club_module=club_module,
                raw_club_loss=raw_club_diag_loss,
                normalized_club_loss=normalized_club_diag_loss,
                controlled_negative_collision=controlled_negative_collision,
            )

        # Isolate each adversarial contribution on the shared representation
        # before total.backward destroys the graph.  Dividing here exactly
        # matches gradient accumulation's effective-batch scaling.
        if adversarial_cap_on:
            _cap_losses = {}
            if cfg.grl_weight > 0 and grl_lam != 0.0:
                _cap_losses['grl'] = m_grl * l_grl
            if grl_p_weight > 0 and grl_p_lam is not None and grl_p_lam != 0.0:
                _cap_losses['grl_p'] = m_grl_p * l_grl_p
            for _name, _loss in _cap_losses.items():
                _grads = torch.autograd.grad(
                    _loss / accumulation,
                    adversarial_cap_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                adversarial_grad_buffers[_name] = accumulate_task_grads(
                    adversarial_grad_buffers.get(_name), _grads)

        discriminator_buffer.append({
            "zL": out["z_L"].detach(), "zP": out["z_P"].detach(),
            "zU": out["z_U"].detach() if "z_U" in out else None,
            "lengths": out["out_lengths"].detach(),
            "speakers": adversary_speaker_ids.detach(), "targets": targets.detach(),
            "target_lengths": target_lengths.detach(),
            "prosody": ((p_f0.detach(), p_v.detach(), p_e.detach())
                        if prosody_on and "prosody_pred" in out else None),
        })

        # GradNorm weight update (retains the graph so total.backward() still works).
        if gradnorm_on and accumulation_boundary and step % gn_every == 0:
            _gn_losses = {'recon': l_recon, 'pr': l_pr, 'sid': l_sid, 'grl': l_grl,
                          'grl_p': l_grl_p, 'grl_u': l_grl_u, 'grl_p_u': l_grl_p_u,
                          'aux': l_aux}
            gn_ctrl.update({n: _gn_losses[n] for n in gn_ctrl.names})

        scaled_total = total / accumulation
        if not use_fp16:
            scaled_total.backward()
            if not accumulation_boundary:
                continue
        else:
            scaler.scale(scaled_total).backward()
            if not accumulation_boundary:
                continue
            scaler.unscale_(optimizer)

        if adversarial_cap_on and adversarial_grad_buffers:
            last_adversarial_cap_stats = apply_task_gradient_caps_(
                adversarial_cap_params,
                adversarial_grad_buffers,
                adversarial_cap_ratios,
            )
        else:
            last_adversarial_cap_stats = {}

        if club_diagnostics_due:
            _group_names = (["sae", "routing", "task_heads", "sid_head", "adversaries"]
                            if not projection_params else
                            ["sae", "routing", "projection", "task_heads", "sid_head", "adversaries"])
            if vib_params:
                _group_names.append("vib")
            _preclip_groups = [
                (name, _current_grad_norm(group["params"]), float(group["lr"]))
                for name, group in zip(_group_names, optimizer.param_groups)
            ]
            _update_params = {
                "sae_encoder": [model.sae.enc_weight],
                "sae_decoder": [model.sae.dec_weight],
                "sae_bias": [model.sae.b_pre],
                "routing": list(routing_params),
            }
            _before_update = {
                name: [p.detach().clone() for p in params]
                for name, params in _update_params.items()
            }
        preclip_norm = nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
        if club_diagnostics_due:
            _preclip = float(preclip_norm)
            _clip_scale = min(1.0, float(cfg.grad_clip) / max(_preclip, 1e-12))
            full_diagnostic_lines.append(
                f"    optimizer_global: preclip={_preclip:.5f} "
                f"clip_limit={float(cfg.grad_clip):.5f} applied_scale={_clip_scale:.6f}")
            for _name, _norm, _lr in _preclip_groups:
                full_diagnostic_lines.append(
                    f"    optimizer_group { _name:<12s}: preclip={_norm:.5f} lr={_lr:.3e}")
        if not use_fp16:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if club_diagnostics_due:
            for _name, _params in _update_params.items():
                _delta_sq = [(p.detach().float() - old.float()).pow(2).sum()
                             for p, old in zip(_params, _before_update[_name])]
                _param_sq = [p.detach().float().pow(2).sum() for p in _params]
                _delta = float(torch.stack(_delta_sq).sum().sqrt()) if _delta_sq else 0.0
                _param = float(torch.stack(_param_sq).sum().sqrt()) if _param_sq else 0.0
                full_diagnostic_lines.append(
                    f"    optimizer_update {_name:<12s}: delta={_delta:.6f} "
                    f"param={_param:.5f} delta/param={_delta / max(_param, 1e-12):.3e}")
            print("\n".join(full_diagnostic_lines))

        club_pool_buffer.clear(); club_label_buffer.clear()
        club_phn_pool_buffer.clear(); club_phn_label_buffer.clear()

        if getattr(cfg, 'renorm_decoder', False):
            model.sae.normalize_decoder()

        # ---- Extra discriminator catch-up steps (GAN n_critic) ----
        # Reuse THIS batch's detached z_L/z_P (no extra encoder forward) to take a
        # few more gradient steps on the adversary heads, so they track the moving
        # encoder instead of stalling at chance.  lam=0: z is detached anyway, this
        # is pure discriminator learning (no reversal to the encoder).
        if n_disc_steps > 1:
            pros_adv = prosody_on and (grl_pros_w > 0 or grl_pros_u_w > 0)
            for _ in range(n_disc_steps - 1):
                optimizer.zero_grad(set_to_none=True)
                for cached in discriminator_buffer:
                    zL_d, zP_d, zU_d = cached["zL"], cached["zP"], cached["zU"]
                    lens_d = cached["lengths"]
                    with ctx:
                        sp = model.grl_head(zL_d, lens_d, 0.0)
                        l_d = _speaker_adv_loss(sp, cached["speakers"], lens_d)
                        if grl_p_weight > 0:
                            ph = model.pr_grl_head(zP_d, 0.0)
                            l_d = l_d + ctc_pr_loss(
                                ph, cached["targets"], lens_d, cached["target_lengths"])
                        if hasattr(model, 'grl_head_u') and zU_d is not None:
                            spu = model.grl_head_u(zU_d, lens_d, 0.0)
                            l_d = l_d + _speaker_adv_loss(spu, cached["speakers"], lens_d)
                            phu = model.pr_grl_head_u(zU_d, 0.0)
                            l_d = l_d + ctc_pr_loss(
                                phu, cached["targets"], lens_d, cached["target_lengths"])
                        if pros_adv and cached["prosody"] is not None:
                            pf0_d, pv_d, pe_d = cached["prosody"]
                            if grl_pros_w > 0:
                                l_d = l_d + _prosody_train_loss(
                                    model.prosody_grl_head(zL_d, 0.0), pf0_d, pv_d, pe_d, lens_d)
                            if grl_pros_u_w > 0 and zU_d is not None:
                                l_d = l_d + _prosody_train_loss(
                                    model.prosody_grl_head_u(zU_d, 0.0), pf0_d, pv_d, pe_d, lens_d)
                    if not use_fp16: (l_d / accumulation).backward()
                    else: scaler.scale(l_d / accumulation).backward()
                if use_fp16: scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(disc_params, cfg.grad_clip)
                if not use_fp16: optimizer.step()
                else: scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
        discriminator_buffer.clear()

        scheduler.step()

        if step % cfg.log_every == 0 or step == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            if fixed_blocks:
                n_L, n_P, n_U = cfg.K_L, cfg.K_P, cfg.K_U
                routing_diag = {}
                entropy = float('nan')
            elif not no_routing and not projection_mode:
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
            if getattr(cfg, "grl_grad_norm", False):
                losses["grl_grad_norm_target_eff"] = float(
                    _scheduled_grl_grad_norm_target(cfg, step)
                )
            if emotion_on:
                losses.update({
                    "emotion": l_emo.item(),
                    "emotion_grl": l_emo_grl.item(),
                    "emotion_pros": l_emo_pros.item(),
                    "emotion_pros_grl": l_emo_pros_grl.item(),
                    "emotion_aux_raw": l_emo_aux_raw.item(),
                    "emotion_aux": l_emo_aux.item(),
                    "emotion_aux_scale": float(emo_aux_scale.item()),
                    "emotion_ran": 1.0 if run_emotion_aux else 0.0,
                })
                if emo_acc is not None:
                    losses["emotion_acc"] = emo_acc
                if emo_grl_acc is not None:
                    losses["emotion_grl_acc"] = emo_grl_acc

            with torch.no_grad():
                z_active = (out["z_t"] != 0).float()                 # (B, T, K)
                B_, T_   = z_active.shape[:2]
                fmask    = (torch.arange(T_, device=z_active.device).unsqueeze(0)
                            < out["out_lengths"].unsqueeze(1)).float()
                n_valid  = fmask.sum().clamp(min=1)
                hard_idx = None
                if fixed_blocks:
                    hard_idx = model.block_idx                           # (K,) fixed
                elif not no_routing and not projection_mode:
                    hard_idx = model.routing.logits.argmax(dim=-1)       # (K,)
                if hard_idx is not None:
                    act_L = ((z_active * (hard_idx == 0).float()).sum(-1) * fmask).sum() / n_valid
                    act_P = ((z_active * (hard_idx == 1).float()).sum(-1) * fmask).sum() / n_valid
                    act_U = ((z_active * (hard_idx == 2).float()).sum(-1) * fmask).sum() / n_valid
                else:
                    act_L = act_P = act_U = z_active.new_tensor(float('nan'))

                # Adversary readouts on this batch (loss is hard to read; accuracy/PER
                # say directly how much each adversary still extracts).
                gc, gt   = _speaker_correct(out["grl_logits"], adversary_speaker_ids, out["out_lengths"])
                grl_acc  = gc / max(gt, 1)
                grl_true_acc = None
                if shuffle_grl_labels:
                    tc, tt = _speaker_correct(out["grl_logits"], speaker_ids, out["out_lengths"])
                    grl_true_acc = tc / max(tt, 1)
                grl_p_per = grl_u_acc = grl_p_u_per = None
                if "pr_grl_logits" in out:
                    num, den  = _ctc_errors(out["pr_grl_logits"], targets, out["out_lengths"], target_lengths)
                    grl_p_per = num / max(den, 1)
                if "grl_u_logits" in out:
                    uc, ut    = _speaker_correct(out["grl_u_logits"], adversary_speaker_ids, out["out_lengths"])
                    grl_u_acc = uc / max(ut, 1)
                if "pr_grl_u_logits" in out:
                    num, den    = _ctc_errors(out["pr_grl_u_logits"], targets, out["out_lengths"], target_lengths)
                    grl_p_u_per = num / max(den, 1)

            # Probe-robust runs (grl_weight=0) suppress GRL diagnostics from
            # TB too — they reflect a phantom head not being trained.
            if cfg.grl_weight > 0:
                losses["grl_acc"] = grl_acc
                if grl_true_acc is not None: losses["grl_true_acc"] = grl_true_acc
            if grl_p_weight > 0 and grl_p_per is not None:
                losses["grl_p_per"] = grl_p_per
            if grl_u_weight   > 0 and grl_u_acc   is not None: losses["grl_u_acc"]   = grl_u_acc
            if grl_p_u_weight > 0 and grl_p_u_per is not None: losses["grl_p_u_per"] = grl_p_u_per

            cap_str = ""
            if adversarial_cap_on and last_adversarial_cap_stats:
                for _key, _value in last_adversarial_cap_stats.items():
                    losses[f"adv_cap/{_key}"] = float(_value)
                _ref = last_adversarial_cap_stats.get('reference_norm', float('nan'))
                _pieces = []
                for _name in ('grl', 'grl_p'):
                    if f'{_name}_raw_norm' in last_adversarial_cap_stats:
                        _raw = last_adversarial_cap_stats[f'{_name}_raw_norm']
                        _capped = last_adversarial_cap_stats[f'{_name}_capped_norm']
                        _scale = last_adversarial_cap_stats[f'{_name}_scale']
                        _pieces.append(
                            f"{_name}={_raw:.3f}->{_capped:.3f}(x{_scale:.3f})")
                cap_str = f"  cap[ref={_ref:.3f} {' '.join(_pieces)}]"

            grl_p_str = ""
            if grl_p_weight > 0:
                grl_p_str = f"  grl_p={l_grl_p.item():.4f}"
                if grl_p_per is not None:
                    grl_p_str += f"(per={grl_p_per:.3f})"
            ub_str    = f"  ub={l_ub.item():.4f}"        if ub_w > 0        else ""
            u_str = ""
            if grl_u_weight > 0 or grl_p_u_weight > 0:
                u_str = f"  grlU={l_grl_u.item():.3f}/{l_grl_p_u.item():.3f}"
                _ua = f"{grl_u_acc:.3f}"   if grl_u_acc   is not None else "na"
                _up = f"{grl_p_u_per:.3f}" if grl_p_u_per is not None else "na"
                u_str += f"(acc={_ua},per={_up})"
            pros_str  = ""
            if prosody_on:
                pros_str = f"  pros={l_pros.item():.4f}"
                if grl_pros_w > 0 or grl_pros_u_w > 0:
                    pros_str += f"  grlPr={l_pros_grl.item():.3f}/{l_pros_grl_u.item():.3f}"
            emo_str = ""
            if emotion_on:
                if run_emotion_aux:
                    _ea = f"{emo_acc:.3f}" if emo_acc is not None else "na"
                    _ega = f"{emo_grl_acc:.3f}" if emo_grl_acc is not None else "na"
                    emo_str = (f"  emo={l_emo.item():.3f}(acc={_ea})"
                               f"  grlE={l_emo_grl.item():.3f}(acc={_ega})"
                               f"  emoAux={l_emo_aux_raw.item():.3f}x{float(emo_aux_scale.item()):.2f}")
                else:
                    emo_str = "  emo=skip"
            inv_str   = f"  inv={l_inv.item():.4f}" if invariance_on else ""
            dual_inv_str = ""
            if dual_inv_on:
                with torch.no_grad():
                    # Bucket-restricted diagnostics: only the dims the router
                    # assigned to this view (avoids false collapse alarms when
                    # half of dims are mechanically zero in hard routing).
                    _mL = out.get("m_L"); _mP = out.get("m_P")
                    _maskL = (_mL > 0.5) if _mL is not None else None
                    _maskP = (_mP > 0.5) if _mP is not None else None
                    diag_L = bucket_diag(out["z_L"], out["out_lengths"], _maskL, gamma=inv_var_g)
                    diag_P = bucket_diag(out["z_P"], out["out_lengths"], _maskP, gamma=inv_var_g)
                    _eL = effective_rank(out["z_L"], out["out_lengths"], max_frames=1024)
                    _eP = effective_rank(out["z_P"], out["out_lengths"], max_frames=1024)
                    # Pair-source mix: count last pa batch's source tags
                    _srcs = pa.get("sources", []) if isinstance(pa, dict) else []
                    _n = max(1, len(_srcs))
                    _f_arctic = sum(1 for s in _srcs if s == "arctic") / _n
                    _f_pert   = sum(1 for s in _srcs if s == "perturb") / _n
                losses["inv_L"]                 = l_inv_L.item()
                losses["inv_P"]                 = l_inv_P.item()
                losses["inv_var"]               = l_var.item()
                losses["var/zL_p10_std"]        = diag_L["p10_std"]
                losses["var/zP_p10_std"]        = diag_P["p10_std"]
                losses["var/zL_frac_blw_g"]     = diag_L["frac_blw_g"]
                losses["var/zP_frac_blw_g"]     = diag_P["frac_blw_g"]
                losses["var/zL_k_active"]       = diag_L["k_active"]
                losses["var/zP_k_active"]       = diag_P["k_active"]
                losses["inv/zL_utt_norm_mean"]  = diag_L["utt_norm_mean"]
                losses["inv/zL_utt_norm_std"]   = diag_L["utt_norm_std"]
                losses["inv/zP_utt_norm_mean"]  = diag_P["utt_norm_mean"]
                losses["inv/zP_utt_norm_std"]   = diag_P["utt_norm_std"]
                losses["probe_robust/cov"]      = float(l_cov.item())
                losses["probe_robust/club"]     = float(l_club.item())
                if bool(getattr(cfg, 'club_grad_norm', False)):
                    losses["probe_robust/club_grad_norm_target"] = float(
                        cfg.club_grad_norm_target)
                    losses["probe_robust/club_grad_norm_delivered"] = float(
                        eff_club_w * cfg.club_grad_norm_target)
                losses["probe_robust/q_phi_ce"] = float(club_ce) if club_ce == club_ce else 0.0
                losses["probe_robust/q_phi_acc"] = float(club_acc) if club_acc == club_acc else 0.0
                if club_module is not None:
                    for _name, _value in club_module.last_diagnostics.items():
                        losses[f"probe_robust/club_{_name}"] = _value
                if bool(getattr(cfg, 'club_phoneme_enabled', False)):
                    losses["probe_robust/club_phn"]     = float(l_club_phn.item())
                    losses["probe_robust/q_phi_phn_ce"] = float(club_phn_ce) if club_phn_ce == club_phn_ce else 0.0
                    losses["probe_robust/q_phi_phn_acc"] = float(club_phn_acc) if club_phn_acc == club_phn_acc else 0.0
                    losses["probe_robust/phn_pseudo_entropy"] = club_phn_pseudo_entropy
                    losses["probe_robust/phn_pseudo_confidence"] = club_phn_pseudo_confidence
                    losses["probe_robust/phn_pseudo_coverage"] = club_phn_pseudo_coverage
                    if club_phn_module is not None:
                        for _name, _value in club_phn_module.last_diagnostics.items():
                            losses[f"probe_robust/club_phn_{_name}"] = _value
                losses["eff_rank/zL"]           = _eL
                losses["eff_rank/zP"]           = _eP
                losses["pair_mix/alpha_arctic"] = _f_arctic
                losses["pair_mix/alpha_pert"]   = _f_pert
                _club_phn_str = (
                    f"  clubP={l_club_phn.item():+.4f}"
                    f"  q_phi_phn[ce={club_phn_ce:.3f},acc={club_phn_acc:.3f}]"
                    if bool(getattr(cfg, 'club_phoneme_enabled', False)) else ""
                )
                dual_inv_str = (
                    f"  inv_L={l_inv_L.item():.4f}  inv_P={l_inv_P.item():.4f}"
                    f"  var={l_var.item():.4f}"
                    f"  cov={l_cov.item():.4f}"
                    f"  club={l_club.item():+.4f}  q_phi[ce={club_ce:.3f},acc={club_acc:.3f}]"
                    f"{_club_phn_str}"
                    f"  k[L/P]={diag_L['k_active']}/{diag_P['k_active']}"
                    f"  Zσ10[L/P]={diag_L['p10_std']:.2f}/{diag_P['p10_std']:.2f}"
                    f"  blw[L/P]={diag_L['frac_blw_g']:.2f}/{diag_P['frac_blw_g']:.2f}"
                    f"  uN[L/P]={diag_L['utt_norm_mean']:.2f}±{diag_L['utt_norm_std']:.2f}/"
                    f"{diag_P['utt_norm_mean']:.2f}±{diag_P['utt_norm_std']:.2f}"
                    f"  eR[L/P]={_eL:.0f}/{_eP:.0f}"
                    f"  mix[arc/pert]={_f_arctic:.2f}/{_f_pert:.2f}"
                )
            vib_str   = f"  vib={l_vib.item():.4f}(w={eff_vib_w:.1e})" if vib_w > 0 else ""
            n_dead = int((model.sae.steps_since_fired > model.sae.dead_threshold).sum())
            dead_frac = n_dead / cfg.K
            aux_str = f"  dead={100*dead_frac:.1f}%"
            losses["dead_frac"] = dead_frac
            if aux_k_on:
                aux_str = f"  aux={l_aux.item():.4f}{aux_str}"
            gn_str = ""
            if gradnorm_on:
                _gw = gn_ctrl.weights()
                gn_str = "  gn=[" + " ".join(f"{k}:{v:.2f}" for k, v in _gw.items()) + "]"
                losses.update({f"w_{k}": v for k, v in _gw.items()})
            # Suppress GRL log fields when the GRL mechanism is off
            # (probe-robust / CLUB runs): the head forwards still run but
            # contribute zero gradient, and printing their CE/acc clutters
            # the log with a phantom adversary.
            grl_str = ""
            if cfg.grl_weight > 0:
                grl_true_str = (f",true={grl_true_acc:.3f}"
                                if grl_true_acc is not None else "")
                grl_str = f"  grl={l_grl.item():.4f}(acc={grl_acc:.3f}{grl_true_str})"
            print(
                f"  step {step:>6d}/{cfg.stage2_steps}"
                f"  recon={l_recon.item():.4f}"
                f"  pr={l_pr.item():.4f}"
                f"  sid={l_sid.item():.4f}"
                f"{grl_str}{grl_p_str}{cap_str}{u_str}{pros_str}{emo_str}{inv_str}{dual_inv_str}{vib_str}{aux_str}{ub_str}{gn_str}"
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
            append_metrics(metrics_path, {"step": step, "split": "train", **losses})
            tb.log_routing(step, n_L, n_P, n_U, entropy, routing_diag)
            tb.log_sae(step, density)

        if step % cfg.ckpt_every == 0 or step == cfg.stage2_steps:
            val_metrics = _eval_stage2(model, val_dl, device, use_bf16)
            val_metrics.update(_eval_club_estimators(
                model, val_dl, device, use_bf16, club_module, club_phn_module))
            if emotion_on and emo_val_dl is not None:
                emo_val_metrics = _eval_emotion(model, emo_val_dl, device, use_bf16)
                val_metrics.update(emo_val_metrics)
            # Per-bucket × per-task val readout (PR=PER↓, SID=acc↑), read straight off
            # the model heads — NOT a probe.  Build heads: z_L PR (pr_head), z_P SID
            # (sid_head).  Adversary heads (co-adapted proxy): z_L SID (grl_head),
            # z_P PR (pr_grl_head), z_U PR/SID (pr_grl_u_head / grl_u_head).
            # Probe-robust runs (grl_weight=0 / grl_phoneme_weight=0) skip the
            # adversary readouts — those heads are untrained random projections
            # in that mode and their accuracy/PER is not a leakage signal.
            # Authoritative cross-leakage measurements come from diag_probe/.
            if cfg.grl_weight > 0:
                zL_str = f"z_L PR={val_metrics['per']:.3f} SID={val_metrics['grl_acc']:.3f}"
            else:
                zL_str = f"z_L PR={val_metrics['per']:.3f}"
            if grl_p_weight > 0 and "grl_p_per" in val_metrics:
                zP_str = f"z_P PR={val_metrics['grl_p_per']:.3f} SID={val_metrics['sid_acc']:.3f}"
            else:
                zP_str = f"z_P SID={val_metrics['sid_acc']:.3f}"
            bucket_str = f"  | {zL_str}  | {zP_str}"
            if "grl_u_acc" in val_metrics:
                zU_pr = f"PR={val_metrics['grl_p_u_per']:.3f} " if "grl_p_u_per" in val_metrics else ""
                bucket_str += f"  | z_U {zU_pr}SID={val_metrics['grl_u_acc']:.3f}"
            print(
                f"  [val] step={step}"
                f"  recon={val_metrics['recon']:.4f}"
                f"  pr={val_metrics['pr']:.4f}"
                f"{bucket_str}"
            )
            if "emotion_acc" in val_metrics:
                emo_bucket = f"z_P emotion={val_metrics['emotion_acc']:.3f}"
                if "emotion_zL_acc" in val_metrics:
                    emo_bucket += f"  z_L emotion={val_metrics['emotion_zL_acc']:.3f}"
                print(f"  [iemocap val] step={step}  {emo_bucket}")
            tb.log_val(step, val_metrics)
            append_metrics(metrics_path, {"step": step, "split": "val", **val_metrics})
            # Selection criterion.  recon-best is undertrained (lowest before the
            # task/adversary losses reshape z_t).  The previous criterion was
            # per + (1 - sid_acc) — only the two *main-task* head outputs.  That
            # was shown in the June 23 2026 pending-sweep analysis to select
            # checkpoints whose final probe-recoverable z_L SID disagreed
            # wildly with the val-time number (e.g. seed 7: val 0.008, final
            # 0.704), because the criterion never looked at the adversary heads
            # at all.  Include them when present so the selection at least
            # *sees* the in-training leakage proxies:
            #   z_L PR        ↓  (per)            phoneme in z_L                main task
            #   z_P SID       ↑  (sid_acc)        speaker in z_P                main task
            #   z_L SID       ↓  (grl_acc)        speaker leakage into z_L      adversary
            #   z_P PR        ↑  (grl_p_per)      phoneme leakage into z_P      adversary
            # All four terms enter as "lower is better".  Missing adversaries
            # default to neutral (0) so runs without that head are not penalised
            # and the criterion remains backwards-compatible with old configs.
            # Still a coarse proxy, NOT a held-out probe: the diagnostic probe
            # in diag_probe/ is the only authoritative signal.  Per-step
            # checkpoints are still saved so recon-best is recoverable.
            disent_score = (
                val_metrics["per"]
                + (1.0 - val_metrics["sid_acc"])
                + val_metrics.get("grl_acc", 0.0)
                + (1.0 - val_metrics.get("grl_p_per", 1.0))
            )
            if disent_score < best_metric:
                best_metric = disent_score
                _save_checkpoint(best_ckpt, model, optimizer, scheduler, step, best_metric,
                                 cfg=cfg, kind="inference", val_metrics=val_metrics)
                parts = (f"PER={val_metrics['per']:.3f} "
                         f"sid={val_metrics['sid_acc']:.3f} "
                         f"grl_acc={val_metrics.get('grl_acc', float('nan')):.3f} "
                         f"grl_p_per={val_metrics.get('grl_p_per', float('nan')):.3f}")
                print(f"  ✓ best checkpoint (disent={best_metric:.4f}  {parts}) → {best_ckpt}")
            _save_checkpoint(cfg.checkpoint_dir / f"stage2_step{step}.pt",
                             model, optimizer, scheduler, step, best_metric,
                             cfg=cfg, kind="inference", val_metrics=val_metrics)
            tb.flush()

        resume_every = int(getattr(cfg, "resume_every", 0))
        if resume_every > 0 and step % resume_every == 0:
            _save_checkpoint(cfg.checkpoint_dir / "latest-resume.pt", model, optimizer,
                             scheduler, step, best_metric, cfg=cfg, scaler=scaler,
                             club_module=club_module, club_phn_module=club_phn_module,
                             train_sampler=train_dl.sampler, pa_loader=pa_loader, pb_loader=pb_loader)
        if segment.reached(step):
            _save_checkpoint(cfg.checkpoint_dir / "latest-resume.pt", model, optimizer,
                             scheduler, step, best_metric, cfg=cfg, scaler=scaler,
                             club_module=club_module, club_phn_module=club_phn_module,
                             train_sampler=train_dl.sampler, pa_loader=pa_loader, pb_loader=pb_loader)
            tb.close()
            print(f"[stage 2] segment complete at step {step}; resume from latest-resume.pt")
            return cfg.checkpoint_dir / "latest-resume.pt"

    tb.close()
    _save_checkpoint(cfg.checkpoint_dir / "latest-resume.pt", model, optimizer,
                     scheduler, cfg.stage2_steps, best_metric, cfg=cfg, scaler=scaler,
                     club_module=club_module, club_phn_module=club_phn_module,
                     train_sampler=train_dl.sampler, pa_loader=pa_loader, pb_loader=pb_loader)
    _save_checkpoint(cfg.checkpoint_dir / "final.pt", model, optimizer, scheduler,
                     cfg.stage2_steps, best_metric, cfg=cfg, kind="inference")
    print(f"\n[stage 2] done.  Best disent_score={best_metric:.4f}  → {best_ckpt}")
    return best_ckpt
