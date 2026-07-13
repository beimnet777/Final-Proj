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
    def __init__(self, dims, vocab_size, num_speakers, num_emotions):
        super().__init__()
        self.pr = nn.ModuleDict({s: nn.Linear(d, vocab_size) for s, d in dims.items()})
        self.sid = nn.ModuleDict({s: nn.Linear(2 * d, num_speakers) for s, d in dims.items()})
        self.emotion = nn.ModuleDict({
            s: nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, num_emotions))
            for s, d in dims.items()
        })


def _representations(model, batch, device):
    audio = batch["audios"].to(device)
    lengths = batch["audio_lengths"].to(device)
    with torch.no_grad():
        out = model(audio, lengths, stage=2, grl_lambda=0.0, grl_p_lambda=0.0,
                    grl_prosody_lambda=0.0, grl_emotion_lambda=0.0,
                    emit_emotion=False)
    return {s: out[s].detach().float() for s in ("z_t", "z_L", "z_P")}, out["out_lengths"]


@torch.no_grad()
def evaluate(model, probes, dl, device):
    probes.eval()
    totals = {s: {"pr_n": 0, "pr_d": 0, "sid_c": 0, "sid_n": 0,
                  "conf": torch.zeros(model.cfg.emotion_num_classes,
                                      model.cfg.emotion_num_classes)}
              for s in probes.pr}
    for batch in dl:
        reps, olen = _representations(model, batch, device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        speakers = batch["speaker_ids"].to(device)
        emotions = batch["emotion"].to(device)
        for source, z in reps.items():
            pr_logits = probes.pr[source](z)
            n, d = U.ctc_errors(pr_logits, targets, olen, target_lengths)
            totals[source]["pr_n"] += n; totals[source]["pr_d"] += d
            pooled = _pool_stats(z, olen)
            sid_pred = probes.sid[source](pooled).argmax(-1)
            valid = speakers >= 0
            totals[source]["sid_c"] += int((sid_pred[valid] == speakers[valid]).sum())
            totals[source]["sid_n"] += int(valid.sum())
            emo_pred = probes.emotion[source](pooled).argmax(-1)
            for truth, pred in zip(emotions.cpu().tolist(), emo_pred.cpu().tolist()):
                totals[source]["conf"][truth, pred] += 1
    result = {}
    for source, values in totals.items():
        conf = values["conf"]
        result[source] = {
            "pr_per": values["pr_n"] / max(values["pr_d"], 1),
            "sid_acc": values["sid_c"] / max(values["sid_n"], 1),
            "emotion_uar": U.uar_from_confusion(conf),
            "emotion_acc": float(conf.diag().sum() / conf.sum().clamp(min=1)),
        }
    probes.train()
    return result


def _better(task, value, best):
    return value < best if task == "pr" else value > best


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
    p.add_argument("--output", type=Path, default=Path("msp_probe_results.json"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    reps, _ = _representations(model, first, device)
    dims = {source: z.shape[-1] for source, z in reps.items()}
    probes = ProbeSet(dims, tokenizer.vocab_size, cfg.num_speakers,
                      cfg.emotion_num_classes).to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=args.lr, weight_decay=1e-4)
    emotion_weights = U.emotion_class_weights(train_dl.dataset.rows,
                                               cfg.emotion_num_classes, device)

    best = {(source, task): (float("inf") if task == "pr" else -float("inf"))
            for source in dims for task in ("pr", "sid", "emotion")}
    best_states = {}
    iterator = iter(train_dl)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_dl); batch = next(iterator)
        reps, olen = _representations(model, batch, device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        speakers = batch["speaker_ids"].to(device)
        emotions = batch["emotion"].to(device)
        loss = torch.zeros((), device=device)
        for source, z in reps.items():
            loss = loss + ctc_pr_loss(probes.pr[source](z), targets, olen, target_lengths)
            pooled = _pool_stats(z, olen)
            loss = loss + F.cross_entropy(probes.sid[source](pooled), speakers)
            loss = loss + F.cross_entropy(probes.emotion[source](pooled), emotions,
                                          weight=emotion_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.val_every == 0 or step == args.steps:
            metrics = evaluate(model, probes, val_dl, device)
            print(f"[probe step={step}] {json.dumps(metrics, sort_keys=True)}", flush=True)
            for source, row in metrics.items():
                values = {"pr": row["pr_per"], "sid": row["sid_acc"],
                          "emotion": row["emotion_uar"]}
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
        "test": evaluate(model, probes, test_dl, device)}
    if extra:
        result["test_unseen"] = evaluate(model, probes, extra[0], device)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
