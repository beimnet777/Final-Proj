#!/usr/bin/env python3
"""Train fresh frozen probes on MSP representations.

Unlike the jointly trained heads, these probes cannot shape the representation.
Validation selects each probe independently and the held-out test split is read
only after training, providing an MSP-native leakage measurement for z_t/z_L/z_P.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DISConfig
from losses import ctc_pr_loss
from model import build_dis_model
from training_runtime import atomic_json_dump

from .checkpoints import checkpoint_model_state
from .data import make_msp_dataloaders
from .heads import GELUSpeakerGRLHead
from . import utils as U


def _pool_stats(z, lengths):
    t = z.shape[1]
    mask = (torch.arange(t, device=z.device)[None, :] < lengths[:, None]).unsqueeze(-1)
    count = mask.sum(1).clamp(min=1)
    mean = (z * mask).sum(1) / count
    var = (((z - mean[:, None]) ** 2) * mask).sum(1) / count
    return torch.cat((mean, var.clamp_min(1e-8).sqrt()), dim=-1)


class ProbeSet(nn.Module):
    def __init__(self, dims, vocab_size, num_speakers, num_emotions, tasks):
        super().__init__()
        self.tasks = tuple(tasks)
        self.pr = nn.ModuleDict()
        self.sid = nn.ModuleDict()
        self.emotion = nn.ModuleDict()
        self.prosody = nn.ModuleDict()
        if "pr" in self.tasks:
            self.pr.update({s: nn.Linear(d, vocab_size) for s, d in dims.items()})
        if "sid" in self.tasks:
            self.sid.update({s: nn.Linear(2 * d, num_speakers) for s, d in dims.items()})
        if "emotion" in self.tasks:
            self.emotion.update({
                s: nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, num_emotions))
                for s, d in dims.items()
            })
        if "prosody" in self.tasks:
            self.prosody.update({
                s: nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 2))
                for s, d in dims.items()
            })


def _representations(model, batch, device, sources):
    audio = batch["audios"].to(device)
    lengths = batch["audio_lengths"].to(device)
    with torch.no_grad():
        out = model(audio, lengths, stage=2, grl_lambda=0.0, grl_p_lambda=0.0,
                    grl_prosody_lambda=0.0, grl_emotion_lambda=0.0,
                    emit_emotion=False)
    return {s: out[s].detach().float() for s in sources}, out["out_lengths"]


class _PearsonAcc:
    def __init__(self) -> None:
        self.n = self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        m = mask.bool()
        x = x[m].double()
        y = y[m].double()
        self.n += x.numel()
        self.sx += x.sum().item()
        self.sy += y.sum().item()
        self.sxx += (x * x).sum().item()
        self.syy += (y * y).sum().item()
        self.sxy += (x * y).sum().item()

    def value(self) -> float:
        if self.n < 2:
            return float("nan")
        cov = self.sxy - self.sx * self.sy / self.n
        vx = self.sxx - self.sx * self.sx / self.n
        vy = self.syy - self.sy * self.sy / self.n
        if vx <= 0 or vy <= 0:
            return float("nan")
        return float(cov / ((vx * vy) ** 0.5))


@torch.no_grad()
def evaluate(model, probes, dl, device, sources, tasks):
    probes.eval()
    totals = {}
    for source in sources:
        totals[source] = {}
        if "pr" in tasks:
            totals[source].update({"pr_n": 0, "pr_d": 0})
        if "sid" in tasks:
            totals[source].update({"sid_c": 0, "sid_n": 0})
        if "emotion" in tasks:
            totals[source]["conf"] = torch.zeros(model.cfg.emotion_num_classes,
                                                 model.cfg.emotion_num_classes)
        if "prosody" in tasks:
            totals[source].update({"prosody_loss_sum": 0.0, "prosody_n": 0,
                                   "f0_corr": _PearsonAcc(), "energy_corr": _PearsonAcc()})
    for batch in dl:
        reps, olen = _representations(model, batch, device, sources)
        if "pr" in tasks:
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
        if "sid" in tasks:
            speakers = batch["speaker_ids"].to(device)
        if "emotion" in tasks:
            emotions = batch["emotion"].to(device)
        if "prosody" in tasks:
            audios = batch["audios"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            f0, voiced, energy = U.prosody_targets_fast(audios, audio_lengths, olen)
            T_targets = f0.shape[1]
            valid = (torch.arange(T_targets, device=device).unsqueeze(0)
                     < olen.unsqueeze(1)).float()
        for source, z in reps.items():
            if "pr" in tasks:
                pr_logits = probes.pr[source](z)
                n, d = U.ctc_errors(pr_logits, targets, olen, target_lengths)
                totals[source]["pr_n"] += n
                totals[source]["pr_d"] += d
            if "sid" in tasks or "emotion" in tasks:
                pooled = _pool_stats(z, olen)
            if "sid" in tasks:
                sid_pred = probes.sid[source](pooled).argmax(-1)
                valid_speakers = speakers >= 0
                totals[source]["sid_c"] += int((sid_pred[valid_speakers] == speakers[valid_speakers]).sum())
                totals[source]["sid_n"] += int(valid_speakers.sum())
            if "emotion" in tasks:
                emo_pred = probes.emotion[source](pooled).argmax(-1)
                for truth, pred in zip(emotions.cpu().tolist(), emo_pred.cpu().tolist()):
                    totals[source]["conf"][truth, pred] += 1
            if "prosody" in tasks:
                pred = probes.prosody[source](z)[:, :T_targets]
                batch_n = int(z.shape[0])
                loss = U.prosody_train_loss(pred, f0, voiced, energy, olen)
                totals[source]["prosody_loss_sum"] += float(loss) * batch_n
                totals[source]["prosody_n"] += batch_n
                totals[source]["f0_corr"].update(pred[..., 0], f0, voiced * valid)
                totals[source]["energy_corr"].update(pred[..., 1], energy, valid)
    result = {}
    for source, values in totals.items():
        row = {}
        if "pr" in tasks:
            row["pr_per"] = values["pr_n"] / max(values["pr_d"], 1)
        if "sid" in tasks:
            row["sid_acc"] = values["sid_c"] / max(values["sid_n"], 1)
        if "emotion" in tasks:
            conf = values["conf"]
            row["emotion_uar"] = U.uar_from_confusion(conf)
            row["emotion_acc"] = float(conf.diag().sum() / conf.sum().clamp(min=1))
        if "prosody" in tasks:
            row["prosody_loss"] = values["prosody_loss_sum"] / max(values["prosody_n"], 1)
            row["prosody_f0_corr"] = values["f0_corr"].value()
            row["prosody_energy_corr"] = values["energy_corr"].value()
        result[source] = row
    probes.train()
    return result


def _better(task, value, best):
    return value < best if task in {"pr", "prosody"} else value > best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--audio_root", required=True)
    p.add_argument("--transcripts", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sources", default="z_t,z_L,z_P",
                   help="comma-separated representation sources to probe")
    p.add_argument("--tasks", default="pr,sid,emotion",
                   help="comma-separated probe tasks: pr,sid,emotion,prosody")
    p.add_argument("--output", type=Path, default=Path("msp_probe_results.json"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_sources = {"z_t", "z_L", "z_P"}
    valid_tasks = {"pr", "sid", "emotion", "prosody"}
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    bad_sources = sorted(set(sources) - valid_sources)
    bad_tasks = sorted(set(tasks) - valid_tasks)
    if bad_sources:
        raise ValueError(f"unknown probe source(s): {bad_sources}; expected {sorted(valid_sources)}")
    if bad_tasks:
        raise ValueError(f"unknown probe task(s): {bad_tasks}; expected {sorted(valid_tasks)}")
    if not sources or not tasks:
        raise ValueError("--sources and --tasks must each select at least one item")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = DISConfig()
    for key, value in payload.get("analysis_config", {}).items():
        setattr(cfg, key, value)
    cfg.msp_manifest = args.manifest
    cfg.msp_audio_root = args.audio_root
    cfg.msp_transcripts = args.transcripts
    cfg.batch_size = args.batch_size
    cfg.eval_batch_size = args.eval_batch
    cfg.num_workers = args.num_workers
    cfg.invariance = False
    cfg.device = str(device)

    tokenizer, train_dl, val_dl, test_dl, *extra = make_msp_dataloaders(cfg)
    model = build_dis_model(cfg).to(device)
    model.grl_head = GELUSpeakerGRLHead(cfg).to(device)
    state = checkpoint_model_state(payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    non_encoder_missing = [k for k in missing if "_spear." not in k]
    if non_encoder_missing or unexpected:
        raise ValueError(f"checkpoint/model mismatch: missing={non_encoder_missing[:8]} "
                         f"unexpected={unexpected[:8]}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    first = next(iter(val_dl))
    reps, _ = _representations(model, first, device, sources)
    dims = {source: z.shape[-1] for source, z in reps.items()}
    probes = ProbeSet(dims, tokenizer.vocab_size, cfg.num_speakers,
                      cfg.emotion_num_classes, tasks).to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=args.lr, weight_decay=1e-4)
    emotion_weights = (U.emotion_class_weights(train_dl.dataset.rows,
                                               cfg.emotion_num_classes, device)
                       if "emotion" in tasks else None)

    best = {(source, task): (float("inf") if task in {"pr", "prosody"} else -float("inf"))
            for source in dims for task in tasks}
    best_states = {}
    iterator = iter(train_dl)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_dl); batch = next(iterator)
        reps, olen = _representations(model, batch, device, sources)
        if "pr" in tasks:
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
        if "sid" in tasks:
            speakers = batch["speaker_ids"].to(device)
        if "emotion" in tasks:
            emotions = batch["emotion"].to(device)
        if "prosody" in tasks:
            audios = batch["audios"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            f0, voiced, energy = U.prosody_targets_fast(audios, audio_lengths, olen)
        loss = torch.zeros((), device=device)
        for source, z in reps.items():
            if "pr" in tasks:
                loss = loss + ctc_pr_loss(probes.pr[source](z), targets, olen, target_lengths)
            if "sid" in tasks or "emotion" in tasks:
                pooled = _pool_stats(z, olen)
            if "sid" in tasks:
                loss = loss + F.cross_entropy(probes.sid[source](pooled), speakers)
            if "emotion" in tasks:
                loss = loss + F.cross_entropy(probes.emotion[source](pooled), emotions,
                                              weight=emotion_weights)
            if "prosody" in tasks:
                loss = loss + U.prosody_train_loss(probes.prosody[source](z), f0, voiced, energy, olen)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.val_every == 0 or step == args.steps:
            metrics = evaluate(model, probes, val_dl, device, sources, tasks)
            print(f"[probe step={step}] {json.dumps(metrics, sort_keys=True)}", flush=True)
            for source, row in metrics.items():
                values = {}
                if "pr" in tasks:
                    values["pr"] = row["pr_per"]
                if "sid" in tasks:
                    values["sid"] = row["sid_acc"]
                if "emotion" in tasks:
                    values["emotion"] = row["emotion_uar"]
                if "prosody" in tasks:
                    values["prosody"] = row["prosody_loss"]
                for task, value in values.items():
                    key = (source, task)
                    if _better(task, value, best[key]):
                        best[key] = value
                        best_states[key] = copy.deepcopy(
                            getattr(probes, task)[source].state_dict())

    for (source, task), state_dict in best_states.items():
        getattr(probes, task)[source].load_state_dict(state_dict)
    result = {"checkpoint": str(args.checkpoint), "validation_selection": {
        f"{source}.{task}": value for (source, task), value in best.items()},
        "sources": list(sources),
        "tasks": list(tasks),
        "test": evaluate(model, probes, test_dl, device, sources, tasks)}
    if extra:
        result["test_unseen"] = evaluate(model, probes, extra[0], device, sources, tasks)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
