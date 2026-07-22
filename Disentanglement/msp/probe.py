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
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from config import DISConfig
from losses import ctc_pr_loss
from model import build_dis_model
from training_runtime import atomic_json_dump
from data.dataset import CharTokenizer

from .checkpoints import checkpoint_model_state
from .data import EMOTION_NAMES, make_msp_dataloaders
from .heads import GELUSpeakerGRLHead
from . import utils as U

from diag_probe import probe_runner as base_probe


class _FrameLinearCTCProbe(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, proj_dim)
        self.linear = nn.Linear(proj_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.projector(x))


class _FrameMLPCTCProbe(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, proj_dim)
        self.linear = nn.Linear(proj_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.relu(self.projector(x)))


def _make_pr_probe(source_dim: int, vocab_size: int, args) -> nn.Module:
    if args.pr_probe_arch == "linear":
        return _FrameLinearCTCProbe(source_dim, vocab_size,
                                    proj_dim=args.pr_probe_proj_dim)
    if args.pr_probe_arch == "mlp":
        return _FrameMLPCTCProbe(source_dim, vocab_size,
                                 proj_dim=args.pr_probe_proj_dim)
    if args.pr_probe_arch == "direct":
        return nn.Linear(source_dim, vocab_size)
    raise ValueError(f"unknown PR probe arch: {args.pr_probe_arch}")


def _pool_stats(z, lengths):
    t = z.shape[1]
    mask = (torch.arange(t, device=z.device)[None, :] < lengths[:, None]).unsqueeze(-1)
    count = mask.sum(1).clamp(min=1)
    mean = (z * mask).sum(1) / count
    var = (((z - mean[:, None]) ** 2) * mask).sum(1) / count
    return torch.cat((mean, var.clamp_min(1e-8).sqrt()), dim=-1)


def _make_asr_probe(source_dim: int, vocab_size: int, args) -> nn.Module:
    if args.asr_probe_arch == "lstm":
        return base_probe._ASRLSTMProbe(
            source_dim, vocab_size,
            proj_dim=args.asr_probe_proj_dim,
            lstm_hidden=args.asr_lstm_hidden,
            num_layers=args.asr_lstm_layers,
            dropout=args.asr_probe_dropout,
            time_mask_param=args.asr_time_mask_param,
            freq_mask_param=args.asr_freq_mask_param,
        )
    if args.asr_probe_arch == "linear":
        return base_probe._ASRProbe(
            source_dim, vocab_size,
            proj_dim=args.asr_probe_proj_dim,
            dropout=args.asr_probe_dropout,
        )
    if args.asr_probe_arch == "direct":
        return base_probe._PRProbeDirect(source_dim, vocab_size)
    if args.asr_probe_arch == "mlp":
        return base_probe._PRProbeMLP(source_dim, vocab_size)
    raise ValueError(f"unknown ASR probe arch: {args.asr_probe_arch}")


class ProbeSet(nn.Module):
    def __init__(self, dims, vocab_size, asr_vocab_size, num_speakers,
                 num_emotions, tasks, args):
        super().__init__()
        self.tasks = tuple(tasks)
        self.pr = nn.ModuleDict()
        self.asr = nn.ModuleDict()
        self.sid = nn.ModuleDict()
        self.emotion = nn.ModuleDict()
        self.prosody = nn.ModuleDict()
        if "pr" in self.tasks:
            self.pr.update({s: _make_pr_probe(d, vocab_size, args)
                            for s, d in dims.items()})
        if "asr" in self.tasks:
            self.asr.update({s: _make_asr_probe(d, asr_vocab_size, args)
                             for s, d in dims.items()})
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


def _char_targets(texts, tokenizer: CharTokenizer, device: torch.device):
    encoded = [tokenizer.encode(t) for t in texts]
    # CTCLoss cannot consume zero-length references. The MSP dataset already
    # drops non-verbal transcripts, but this guard makes the ASR probe robust to
    # odd transcript cleanup edge cases.
    encoded = [e if e.numel() else torch.tensor([tokenizer.char_to_id[" "]])
               for e in encoded]
    targets = pad_sequence(encoded, batch_first=True, padding_value=0).to(device)
    lengths = torch.tensor([int(e.numel()) for e in encoded],
                           dtype=torch.long, device=device)
    return targets, lengths


def _cosine_eer(embeddings: torch.Tensor, labels: torch.Tensor,
                max_pairs: int, seed: int) -> dict:
    valid = labels >= 0
    embeddings = embeddings[valid].float()
    labels = labels[valid].long()
    if embeddings.numel() == 0 or labels.numel() < 2:
        return {"sv_eer": float("nan"), "sv_pos_pairs": 0, "sv_neg_pairs": 0}

    embeddings = F.normalize(embeddings, dim=-1, eps=1e-8)
    by_spk = {}
    for idx, speaker in enumerate(labels.tolist()):
        by_spk.setdefault(int(speaker), []).append(idx)
    pos_speakers = [s for s, idxs in by_spk.items() if len(idxs) >= 2]
    all_speakers = list(by_spk)
    if not pos_speakers or len(all_speakers) < 2:
        return {"sv_eer": float("nan"), "sv_pos_pairs": 0, "sv_neg_pairs": 0}

    rng = random.Random(seed)
    target_each = max(1, max_pairs // 2)
    scores = []
    same = []

    for _ in range(target_each):
        s = rng.choice(pos_speakers)
        i, j = rng.sample(by_spk[s], 2)
        scores.append(float((embeddings[i] * embeddings[j]).sum().item()))
        same.append(1)

    for _ in range(target_each):
        s1, s2 = rng.sample(all_speakers, 2)
        i = rng.choice(by_spk[s1])
        j = rng.choice(by_spk[s2])
        scores.append(float((embeddings[i] * embeddings[j]).sum().item()))
        same.append(0)

    scores_np = np.asarray(scores, dtype=np.float64)
    same_np = np.asarray(same, dtype=np.int64)
    pos = scores_np[same_np == 1]
    neg = scores_np[same_np == 0]
    thresholds = np.unique(scores_np)
    best = (float("inf"), float("nan"), float("nan"), float("nan"))
    for thr in thresholds:
        far = float((neg >= thr).mean()) if neg.size else float("nan")
        frr = float((pos < thr).mean()) if pos.size else float("nan")
        gap = abs(far - frr)
        if gap < best[0]:
            best = (gap, 0.5 * (far + frr), far, frr)
    return {
        "sv_eer": best[1],
        "sv_far_at_eer": best[2],
        "sv_frr_at_eer": best[3],
        "sv_pos_pairs": int(pos.size),
        "sv_neg_pairs": int(neg.size),
        "sv_same_score_mean": float(pos.mean()) if pos.size else float("nan"),
        "sv_diff_score_mean": float(neg.mean()) if neg.size else float("nan"),
    }


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
def evaluate(model, probes, dl, device, sources, tasks, asr_tokenizer=None,
             sv_max_pairs: int = 20000, sv_seed: int = 42,
             include_emotion_details: bool = False):
    probes.eval()
    asr_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True) \
        if "asr" in tasks else None
    totals = {}
    for source in sources:
        totals[source] = {}
        if "pr" in tasks:
            totals[source].update({"pr_n": 0, "pr_d": 0})
        if "asr" in tasks:
            totals[source].update({"asr_refs": [], "asr_hyps": [],
                                   "asr_loss_sum": 0.0, "asr_loss_n": 0})
        if "sid" in tasks:
            totals[source].update({"sid_c": 0, "sid_n": 0})
        if "sv" in tasks:
            totals[source].update({"sv_embs": [], "sv_labels": []})
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
        if "asr" in tasks:
            asr_targets, asr_target_lengths = _char_targets(batch["texts"], asr_tokenizer, device)
        if "sid" in tasks:
            speakers = batch["speaker_ids"].to(device)
        if "sv" in tasks:
            sv_speakers = batch["speaker_ids"].to(device)
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
            if "asr" in tasks:
                log_probs = probes.asr[source](z)
                hyps = [
                    base_probe._normalize_asr_text(h, asr_tokenizer)
                    for h in base_probe._greedy_pr_decode(
                        log_probs.cpu(), olen.cpu(), asr_tokenizer
                    )
                ]
                refs = [base_probe._normalize_asr_text(t, asr_tokenizer)
                        for t in batch["texts"]]
                totals[source]["asr_hyps"].extend(hyps)
                totals[source]["asr_refs"].extend(refs)
                loss = asr_loss(log_probs.permute(1, 0, 2), asr_targets,
                                olen, asr_target_lengths)
                totals[source]["asr_loss_sum"] += float(loss.item())
                totals[source]["asr_loss_n"] += 1
            if "sid" in tasks or "emotion" in tasks:
                pooled = _pool_stats(z, olen)
            if "sv" in tasks and ("sid" not in tasks and "emotion" not in tasks):
                pooled = _pool_stats(z, olen)
            if "sid" in tasks:
                sid_pred = probes.sid[source](pooled).argmax(-1)
                valid_speakers = speakers >= 0
                totals[source]["sid_c"] += int((sid_pred[valid_speakers] == speakers[valid_speakers]).sum())
                totals[source]["sid_n"] += int(valid_speakers.sum())
            if "sv" in tasks:
                valid_speakers = sv_speakers >= 0
                totals[source]["sv_embs"].append(pooled[valid_speakers].detach().cpu())
                totals[source]["sv_labels"].append(sv_speakers[valid_speakers].detach().cpu())
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
        if "asr" in tasks:
            row["asr_wer"] = base_probe._word_error_rate(
                values["asr_refs"], values["asr_hyps"])
            row["asr_cer"] = base_probe._char_error_rate(
                values["asr_refs"], values["asr_hyps"])
            row["asr_loss"] = values["asr_loss_sum"] / max(values["asr_loss_n"], 1)
        if "sid" in tasks:
            row["sid_acc"] = values["sid_c"] / max(values["sid_n"], 1)
        if "sv" in tasks:
            if values["sv_embs"]:
                emb = torch.cat(values["sv_embs"], dim=0)
                lab = torch.cat(values["sv_labels"], dim=0)
                row.update(_cosine_eer(emb, lab, sv_max_pairs, sv_seed))
            else:
                row.update({"sv_eer": float("nan"), "sv_pos_pairs": 0, "sv_neg_pairs": 0})
        if "emotion" in tasks:
            conf = values["conf"]
            row["emotion_uar"] = U.uar_from_confusion(conf)
            row["emotion_acc"] = float(conf.diag().sum() / conf.sum().clamp(min=1))
            if include_emotion_details:
                support = conf.sum(dim=1)
                recall = conf.diag() / support.clamp(min=1)
                row["emotion_per_class_recall"] = {
                    EMOTION_NAMES[i]: float(recall[i]) for i in range(len(EMOTION_NAMES))
                }
                row["emotion_class_support"] = {
                    EMOTION_NAMES[i]: int(support[i]) for i in range(len(EMOTION_NAMES))
                }
                row["emotion_confusion"] = conf.to(dtype=torch.int64).tolist()
        if "prosody" in tasks:
            row["prosody_loss"] = values["prosody_loss_sum"] / max(values["prosody_n"], 1)
            row["prosody_f0_corr"] = values["f0_corr"].value()
            row["prosody_energy_corr"] = values["energy_corr"].value()
        result[source] = row
    probes.train()
    return result


def _better(task, value, best):
    return value < best if task in {"pr", "prosody", "asr"} else value > best


def _warmup_linear_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    """Warm up to the peak LR, then decay linearly to zero."""
    if warmup_steps <= 0:
        return 1.0
    if step <= warmup_steps:
        return step / max(warmup_steps, 1)
    return max(0.0, (total_steps - step) /
               max(total_steps - warmup_steps, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--audio_root", required=True)
    p.add_argument("--transcripts", required=True)
    p.add_argument("--lexicon_path", type=Path, default=None,
                   help="Pronunciation lexicon for transcript-derived PR targets. "
                        "If omitted, the checkpoint/config default is used.")
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
                   help="comma-separated probe tasks: pr,asr,sid,sv,emotion,prosody")
    p.add_argument("--pr_probe_arch", choices=("direct", "linear", "mlp"),
                   default="direct",
                   help="PR CTC probe architecture. Default preserves older MSP probes; "
                        "use 'linear' for SUPERB-style projected linear PR.")
    p.add_argument("--pr_probe_proj_dim", type=int, default=256)
    p.add_argument("--pr_probe_lr", type=float, default=None,
                   help="Peak PR learning rate. Defaults to --lr for backwards compatibility.")
    p.add_argument("--pr_probe_warmup_steps", type=int, default=0,
                   help="If positive, warm up PR then linearly decay it to zero. "
                        "The default 0 preserves the historical constant-LR protocol.")
    p.add_argument("--asr_probe_arch", choices=("lstm", "linear", "direct", "mlp"),
                   default="lstm")
    p.add_argument("--asr_probe_lr", type=float, default=5e-4)
    p.add_argument("--asr_probe_warmup_steps", type=int, default=500)
    p.add_argument("--asr_probe_proj_dim", type=int, default=1024)
    p.add_argument("--asr_lstm_hidden", type=int, default=1024)
    p.add_argument("--asr_lstm_layers", type=int, default=2)
    p.add_argument("--asr_time_mask_param", type=int, default=50)
    p.add_argument("--asr_freq_mask_param", type=int, default=64)
    p.add_argument("--asr_probe_dropout", type=float, default=0.1)
    p.add_argument("--sv_max_pairs", type=int, default=20000,
                   help="speaker-verification same/different cosine trials per eval")
    p.add_argument("--sv_seed", type=int, default=42)
    p.add_argument("--emotion_diagnostics", action="store_true",
                   help="include test emotion confusion matrices and per-class recall")
    p.add_argument("--speaker_disjoint_emotion_split", action="store_true",
                   help="probe emotion on a deterministic 80/10/10 speaker-disjoint "
                        "repartition of the closed-set MSP rows")
    p.add_argument("--output", type=Path, default=Path("msp_probe_results.json"))
    args = p.parse_args()

    pr_probe_lr = args.lr if args.pr_probe_lr is None else args.pr_probe_lr

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_sources = {"z_t", "z_L", "z_P"}
    valid_tasks = {"pr", "asr", "sid", "sv", "emotion", "prosody"}
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
    if args.speaker_disjoint_emotion_split and set(tasks) != {"emotion"}:
        raise ValueError("--speaker_disjoint_emotion_split is an emotion-only diagnostic")
    train_tasks = tuple(t for t in tasks if t != "sv")
    asr_tokenizer = CharTokenizer() if "asr" in tasks else None

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = DISConfig()
    for key, value in payload.get("analysis_config", {}).items():
        setattr(cfg, key, value)
    cfg.msp_manifest = args.manifest
    cfg.msp_audio_root = args.audio_root
    cfg.msp_transcripts = args.transcripts
    if args.lexicon_path is not None:
        cfg.lexicon_path = args.lexicon_path
    cfg.batch_size = args.batch_size
    cfg.eval_batch_size = args.eval_batch
    cfg.num_workers = args.num_workers
    cfg.invariance = False
    cfg.device = str(device)
    cfg.msp_speaker_disjoint_probe_split = args.speaker_disjoint_emotion_split
    cfg.msp_speaker_disjoint_seed = args.seed

    if not Path(cfg.lexicon_path).is_file():
        raise FileNotFoundError(
            f"Missing lexicon_path={cfg.lexicon_path}. "
            "Pass --lexicon_path to an HPC-visible librispeech-lexicon.txt."
        )

    tokenizer, train_dl, val_dl, test_dl, *extra = make_msp_dataloaders(cfg)
    model = build_dis_model(cfg).to(device)
    model.grl_head = GELUSpeakerGRLHead(cfg).to(device)
    state = checkpoint_model_state(payload)
    current = model.state_dict()
    compatible = {
        k: v for k, v in state.items()
        if k in current and torch.is_tensor(v) and tuple(v.shape) == tuple(current[k].shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    loaded_sae = sorted(k for k in compatible if str(k).startswith("sae."))
    loaded_route = sorted(k for k in compatible
                          if str(k).startswith(("routing.", "block_", "fixed_")))
    if not loaded_sae:
        raise ValueError(f"checkpoint/model mismatch: no compatible SAE tensors loaded "
                         f"from {args.checkpoint}")
    ignored_shape = sorted(
        k for k, v in state.items()
        if k in current and torch.is_tensor(v) and tuple(v.shape) != tuple(current[k].shape)
    )
    ignored_unexpected = sorted(k for k in state if k not in current)
    route_topk_buffers = {
        "sae.route_topk_enabled",
        "sae.route_topk_idx",
        "sae.route_topk_quotas",
    }
    non_encoder_missing = [k for k in missing
                           if "_spear." not in k and not k.startswith((
                               "pr_head.", "sid_head.", "grl_head.", "pr_grl_head.",
                               "prosody_head.", "prosody_grl_head.", "emotion_head.",
                               "emotion_grl_head.", "emotion_u_grl_head.",
                           )) and k not in route_topk_buffers]
    if non_encoder_missing:
        raise ValueError(f"checkpoint/model mismatch after compatible load: "
                         f"missing={non_encoder_missing[:8]}")
    print(f"[msp_probe] loaded compatible tensors: total={len(compatible)} "
          f"sae={len(loaded_sae)} route={len(loaded_route)}")
    if ignored_shape:
        print(f"[msp_probe] ignored shape-mismatched tensors: {ignored_shape[:12]}")
    if ignored_unexpected:
        print(f"[msp_probe] ignored unexpected tensors: {ignored_unexpected[:12]}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    first = next(iter(val_dl))
    reps, _ = _representations(model, first, device, sources)
    dims = {source: z.shape[-1] for source, z in reps.items()}
    probes = ProbeSet(dims, tokenizer.vocab_size,
                      asr_tokenizer.vocab_size if asr_tokenizer is not None else 0,
                      cfg.num_speakers, cfg.emotion_num_classes, train_tasks,
                      args).to(device)
    param_groups = []
    if "pr" in train_tasks:
        param_groups.append({"params": probes.pr.parameters(), "lr": pr_probe_lr,
                             "initial_lr": pr_probe_lr, "name": "pr"})
    if "asr" in train_tasks:
        param_groups.append({"params": probes.asr.parameters(), "lr": args.asr_probe_lr,
                             "initial_lr": args.asr_probe_lr, "name": "asr"})
    if "sid" in train_tasks:
        param_groups.append({"params": probes.sid.parameters(), "lr": args.lr, "name": "sid"})
    if "emotion" in train_tasks:
        param_groups.append({"params": probes.emotion.parameters(), "lr": args.lr, "name": "emotion"})
    if "prosody" in train_tasks:
        param_groups.append({"params": probes.prosody.parameters(), "lr": args.lr, "name": "prosody"})
    optimizer = (torch.optim.AdamW(param_groups, weight_decay=1e-4)
                 if param_groups else None)
    emotion_weights = (U.emotion_class_weights(train_dl.dataset.rows,
                                               cfg.emotion_num_classes, device)
                       if "emotion" in tasks else None)

    best = {(source, task): (float("inf") if task in {"pr", "prosody", "asr"} else -float("inf"))
            for source in dims for task in train_tasks}
    best_states = {}
    if train_tasks:
        iterator = iter(train_dl)
        ctc_asr = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True) \
            if "asr" in train_tasks else None
        for step in range(1, args.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_dl); batch = next(iterator)
            reps, olen = _representations(model, batch, device, sources)
            scheduled_lrs = {
                "pr": (pr_probe_lr, args.pr_probe_warmup_steps),
                "asr": (args.asr_probe_lr, args.asr_probe_warmup_steps),
            }
            for group in optimizer.param_groups:
                name = group.get("name")
                if name not in scheduled_lrs:
                    continue
                peak_lr, warmup_steps = scheduled_lrs[name]
                if warmup_steps > 0:
                    group["lr"] = group.get("initial_lr", peak_lr) * \
                        _warmup_linear_scale(step, args.steps, warmup_steps)
            if "pr" in train_tasks:
                targets = batch["targets"].to(device)
                target_lengths = batch["target_lengths"].to(device)
            if "asr" in train_tasks:
                asr_targets, asr_target_lengths = _char_targets(batch["texts"], asr_tokenizer, device)
            if "sid" in train_tasks:
                speakers = batch["speaker_ids"].to(device)
            if "emotion" in train_tasks:
                emotions = batch["emotion"].to(device)
            if "prosody" in train_tasks:
                audios = batch["audios"].to(device)
                audio_lengths = batch["audio_lengths"].to(device)
                f0, voiced, energy = U.prosody_targets_fast(audios, audio_lengths, olen)
            loss = torch.zeros((), device=device)
            for source, z in reps.items():
                if "pr" in train_tasks:
                    loss = loss + ctc_pr_loss(probes.pr[source](z), targets, olen, target_lengths)
                if "asr" in train_tasks:
                    log_probs = probes.asr[source](z)
                    loss = loss + ctc_asr(log_probs.permute(1, 0, 2), asr_targets,
                                          olen, asr_target_lengths)
                if "sid" in train_tasks or "emotion" in train_tasks:
                    pooled = _pool_stats(z, olen)
                if "sid" in train_tasks:
                    loss = loss + F.cross_entropy(probes.sid[source](pooled), speakers)
                if "emotion" in train_tasks:
                    loss = loss + F.cross_entropy(probes.emotion[source](pooled), emotions,
                                                  weight=emotion_weights)
                if "prosody" in train_tasks:
                    loss = loss + U.prosody_train_loss(probes.prosody[source](z), f0, voiced, energy, olen)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % args.val_every == 0 or step == args.steps:
                metrics = evaluate(model, probes, val_dl, device, sources, tasks,
                                   asr_tokenizer=asr_tokenizer,
                                   sv_max_pairs=args.sv_max_pairs,
                                   sv_seed=args.sv_seed)
                print(f"[probe step={step}] {json.dumps(metrics, sort_keys=True)}", flush=True)
                for source, row in metrics.items():
                    values = {}
                    if "pr" in train_tasks:
                        values["pr"] = row["pr_per"]
                    if "asr" in train_tasks:
                        values["asr"] = row["asr_cer"]
                    if "sid" in train_tasks:
                        values["sid"] = row["sid_acc"]
                    if "emotion" in train_tasks:
                        values["emotion"] = row["emotion_uar"]
                    if "prosody" in train_tasks:
                        values["prosody"] = row["prosody_loss"]
                    for task, value in values.items():
                        key = (source, task)
                        if _better(task, value, best[key]):
                            best[key] = value
                            best_states[key] = copy.deepcopy(
                                getattr(probes, task)[source].state_dict())

    for (source, task), state_dict in best_states.items():
        getattr(probes, task)[source].load_state_dict(state_dict)
    result = {"checkpoint": str(args.checkpoint), "probe_protocol": {
        "steps": args.steps,
        "val_every": args.val_every,
        "base_lr": args.lr,
        "pr_lr": pr_probe_lr,
        "pr_warmup_steps": args.pr_probe_warmup_steps,
        "asr_lr": args.asr_probe_lr,
        "asr_warmup_steps": args.asr_probe_warmup_steps,
    }, "validation_selection": {
        f"{source}.{task}": value for (source, task), value in best.items()},
        "sources": list(sources),
        "tasks": list(tasks),
        "speaker_disjoint_emotion_split": args.speaker_disjoint_emotion_split,
        "test": evaluate(model, probes, test_dl, device, sources, tasks,
                         asr_tokenizer=asr_tokenizer,
                         sv_max_pairs=args.sv_max_pairs,
                         sv_seed=args.sv_seed,
                         include_emotion_details=args.emotion_diagnostics)}
    if extra:
        result["test_unseen"] = evaluate(model, probes, extra[0], device, sources, tasks,
                                         asr_tokenizer=asr_tokenizer,
                                         sv_max_pairs=args.sv_max_pairs,
                                         sv_seed=args.sv_seed)
    atomic_json_dump(result, args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
