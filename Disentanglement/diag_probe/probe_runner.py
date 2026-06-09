#!/usr/bin/env python3
"""Post-hoc probing for disentanglement analysis (experiments A / B / C).

For each checkpoint, trains SUPERB-style probes on every source representation
and evaluates cross-task leakage.

Sources probed
--------------
  h_t   raw SPEAR features (D=1280)        — upper-bound baseline
  z_t   full SAE latent (K=5120)           — SAE baseline (no routing)
  z_L   linguistic route (K=5120)          — should encode phones, not speakers
  z_P   paralinguistic route (K=5120)      — should encode speakers, not phones

Tasks
-----
  PR    CTC phoneme recognition  → metric: PER
  SID   speaker classification   → metric: accuracy

Cross-leakage cells:
  z_L → SID  : speaker info leaking into linguistic bucket
  z_P → PR   : phoneme info leaking into speaker bucket

Usage
-----
  # Mode B — baselines only (no stage-2 checkpoint)
  python diag_probe/probe_runner.py --stage1_ckpt checkpoints/best.pt --run_name probe_B

  # Mode A — probe Run 2 checkpoint
  python diag_probe/probe_runner.py --stage1_ckpt checkpoints/best.pt \\
      --stage2_ckpt checkpoints/sid1_weakgrl/stage2_best.pt --run_name probe_A

  # Mode C — probe Run 3 checkpoint
  python diag_probe/probe_runner.py --stage1_ckpt checkpoints/best.pt \\
      --stage2_ckpt checkpoints/sid1_nogrl/stage2_best.pt --run_name probe_C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    import jiwer
except ImportError:
    jiwer = None

DIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DIS_DIR.parent
sys.path.insert(0, str(DIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "Probing" / "pr"))

from config import DISConfig
from pr_config import PRConfig

_pr_data = None


# ---------------------------------------------------------------- probe heads

class _SIDProbe(nn.Module):
    """SUPERB-style SID probe: frame projection -> masked mean -> linear."""

    def __init__(self, in_dim: int, num_speakers: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, proj_dim)
        self.linear = nn.Linear(proj_dim, num_speakers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.projector(x)
        x = _mean_pool(x, lengths)
        return self.linear(x)


class _PRProbe(nn.Module):
    """SUPERB-style PR CTC probe: frame projection -> linear -> log-softmax."""

    def __init__(self, in_dim: int, vocab_size: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, proj_dim)
        self.linear = nn.Linear(proj_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projector(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


# ---------------------------------------------------------------- helpers

def _mean_pool(z: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    B, T, _ = z.shape
    mask  = (torch.arange(T, device=z.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
    return (z * mask.unsqueeze(-1)).sum(1) / lengths.float().clamp(min=1).unsqueeze(-1)


def _make_linear_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def _extract_representations(model, audios, audio_lengths, device, use_bf16, has_routing: bool):
    audios         = audios.to(device)
    audio_lengths  = audio_lengths.to(device)

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)
    with ctx:
        if has_routing:
            out = model(audios, audio_lengths, stage=2, grl_lambda=0.0)
        else:
            out = model(audios, audio_lengths, stage=1)

    return {
        "h_t":            out["h_t"].float(),
        "z_t":            out["z_t"].float(),
        "z_L":            out.get("z_L", out["z_t"]).float(),
        "z_P":            out.get("z_P", out["z_t"]).float(),
        "out_lengths":    out["out_lengths"],
    }


def _safe_refs(refs: List[str]) -> List[str]:
    return [r if r else "SPN" for r in refs]


def _edit_distance(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _phone_error_rate(refs: List[str], hyps: List[str]) -> float:
    refs = _safe_refs(refs)
    if jiwer is not None:
        return float(jiwer.wer(refs, hyps))

    edits = total = 0
    for ref, hyp in zip(refs, hyps):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        edits += _edit_distance(hyp_tokens, ref_tokens)
        total += len(ref_tokens)
    return edits / max(total, 1)


def _phones_from_text(text: str, tokenizer) -> str:
    if _pr_data._LEXICON is None:
        raise RuntimeError("PR lexicon is not loaded. Call make_pr_dataloaders() first.")
    phones = _pr_data.text_to_phones(text, _pr_data._LEXICON)
    ids = tokenizer.encode(phones).tolist()
    return tokenizer.decode(ids)


# ---------------------------------------------------------------- probe training / eval

def _train_pr_probe(
    probe: nn.Module,
    src_key: str,
    train_dl,
    model,
    device,
    use_bf16: bool,
    has_routing: bool,
    steps: int,
    lr: float,
    warmup_steps: int,
    grad_clip: float,
) -> None:
    trainable = [p for p in probe.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = _make_linear_schedule(opt, warmup_steps, steps)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    probe.train()
    step  = 0
    model.eval()

    while step < steps:
        for audios, audio_lens, targets, target_lens, _texts in train_dl:
            feats = _extract_representations(
                model, audios, audio_lens, device, use_bf16, has_routing
            )
            z     = feats[src_key]          # (B, T, dim)
            lens  = feats["out_lengths"]
            targets = targets.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            log_probs = probe(z)
            loss = ctc_loss(log_probs.permute(1, 0, 2), targets, lens, target_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            scheduler.step()
            step += 1
            if step >= steps:
                break


def _train_sid_probe(
    probe: nn.Module,
    src_key: str,
    train_dl,
    model,
    device,
    use_bf16: bool,
    has_routing: bool,
    steps: int,
    lr: float,
    warmup_steps: int,
    grad_clip: float,
) -> None:
    trainable = [p for p in probe.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = _make_linear_schedule(opt, warmup_steps, steps)
    ce_loss = nn.CrossEntropyLoss()
    probe.train()
    step = 0
    model.eval()

    while step < steps:
        for audios, audio_lens, _targets, _target_lens, speaker_ids in train_dl:
            feats = _extract_representations(
                model, audios, audio_lens, device, use_bf16, has_routing
            )
            z = feats[src_key]
            lens = feats["out_lengths"]
            speaker_ids = speaker_ids.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            loss = ce_loss(probe(z, lens), speaker_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            scheduler.step()
            step += 1
            if step >= steps:
                break


@torch.no_grad()
def _eval_pr_probe(
    probe: nn.Module,
    src_key: str,
    val_dl,
    model,
    tokenizer,
    device,
    use_bf16: bool,
    has_routing: bool,
) -> float:
    probe.eval()
    model.eval()

    all_hyps: List[str] = []
    all_refs: List[str] = []
    for audios, audio_lens, _targets, _target_lens, texts in val_dl:
        feats = _extract_representations(
            model, audios, audio_lens, device, use_bf16, has_routing
        )
        log_probs = probe(feats[src_key])
        hyps = _greedy_pr_decode(log_probs.cpu(), feats["out_lengths"].cpu(), tokenizer)
        refs = [_phones_from_text(t, tokenizer) for t in texts]
        all_hyps.extend(hyps)
        all_refs.extend(refs)

    return _phone_error_rate(all_refs, all_hyps)


@torch.no_grad()
def _eval_sid_probe(
    probe: nn.Module,
    src_key: str,
    val_dl,
    model,
    device,
    use_bf16: bool,
    has_routing: bool,
) -> float:
    probe.eval()
    model.eval()
    correct = total = 0
    for audios, audio_lens, _targets, _target_lens, speaker_ids in val_dl:
        feats = _extract_representations(
            model, audios, audio_lens, device, use_bf16, has_routing
        )
        speaker_ids = speaker_ids.to(device, non_blocking=True)
        pred = probe(feats[src_key], feats["out_lengths"]).argmax(-1)
        correct += (pred == speaker_ids).sum().item()
        total += speaker_ids.size(0)
    return correct / max(total, 1)


def _greedy_pr_decode(log_probs: torch.Tensor, lengths: torch.Tensor, tokenizer, blank_id: int = 0) -> List[str]:
    preds = log_probs.argmax(dim=-1)
    out: List[str] = []
    for row, n in zip(preds, lengths.tolist()):
        ids = row[:n].tolist()
        collapsed, prev = [], -1
        for i in ids:
            if i != prev:
                collapsed.append(i)
                prev = i
        keep = [i for i in collapsed if i != blank_id]
        out.append(tokenizer.decode(keep))
    return out


# ---------------------------------------------------------------- main

def _parse_args():
    cfg = DISConfig()
    p   = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt",  required=True)
    p.add_argument("--stage2_ckpt",  default=None,
                   help="If given, loads routing and enables z_L / z_P probes")
    p.add_argument("--run_name",     default="probe")
    p.add_argument("--probe_steps",  type=int, default=2000)
    p.add_argument("--topk",         type=int, default=0,
                   help="Override cfg.topk (e.g. 128 for K=10240 checkpoints)")
    p.add_argument("--librispeech_cache_dir", default=str(cfg.librispeech_cache_dir))
    p.add_argument("--lexicon_path",          default=str(cfg.lexicon_path))
    p.add_argument("--max_train_examples",    type=int, default=0)   # 0 = full set (all 251 speakers)
    p.add_argument("--max_val_examples",      type=int, default=500)
    p.add_argument("--pr_max_examples",       type=int, default=0,
                   help="Cap PR train/val/test examples. 0 = full SUPERB PR splits.")
    p.add_argument("--pr_probe_lr",           type=float, default=5e-4)
    p.add_argument("--sid_probe_lr",          type=float, default=1e-4)
    p.add_argument("--probe_warmup_steps",    type=int, default=500)
    p.add_argument("--probe_grad_clip",       type=float, default=1.0)
    return p.parse_args()


def main():
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global _pr_data
    import pr_data as _pr_data_module
    from model import build_dis_model
    from train import _load_stage1_checkpoint
    from data.dataset import make_stage2_dataloaders
    _pr_data = _pr_data_module

    cfg                       = DISConfig()
    cfg.device                = str(device)
    cfg.librispeech_cache_dir = Path(args.librispeech_cache_dir)
    cfg.lexicon_path          = Path(args.lexicon_path)
    cfg.max_train_examples    = args.max_train_examples
    cfg.max_val_examples      = args.max_val_examples

    print(f"[probe] run={args.run_name}  device={device}")
    print(f"[probe] stage1_ckpt={args.stage1_ckpt}")
    print(f"[probe] stage2_ckpt={args.stage2_ckpt or '(none — baselines only)'}")
    print(
        f"[probe] probe_steps={args.probe_steps}  "
        f"sid_train_examples={args.max_train_examples}  pr_max_examples={args.pr_max_examples}"
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # PR uses the same SUPERB phone preparation as Probing/pr:
    # train-clean-100 -> train, dev-clean -> val, 74-token stress-marked phones.
    pr_cfg = PRConfig()
    pr_cfg.data_cache_dir = cfg.librispeech_cache_dir
    pr_cfg.librispeech_lexicon = cfg.lexicon_path
    pr_cfg.batch_size = cfg.batch_size
    pr_cfg.eval_batch_size = cfg.eval_batch_size
    pr_cfg.num_workers = cfg.num_workers
    pr_cfg.max_examples = args.pr_max_examples
    pr_tokenizer, pr_train_dl, pr_val_dl, _ = _pr_data.make_pr_dataloaders(pr_cfg)

    # SID keeps the current LibriSpeech speaker diagnostic split for now, but
    # uses the SUPERB SID probe head below.
    _, sid_train_dl, sid_val_dl = make_stage2_dataloaders(cfg)

    # If a stage2 checkpoint is given, override num_speakers and infer K from checkpoint
    if args.stage2_ckpt:
        _tmp = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
        _state = _tmp["model_state"]
        cfg.num_speakers = _state["sid_head.net.2.weight"].shape[0]
        ckpt_K = _state["sae.enc_weight"].shape[0]
        if ckpt_K != cfg.K:
            print(f"[probe] K overridden from checkpoint: {cfg.K} → {ckpt_K}")
            cfg.K = ckpt_K
        if "routing.logits" in _state:
            ckpt_routes = _state["routing.logits"].shape[1]
            if ckpt_routes != cfg.n_routes:
                print(f"[probe] n_routes overridden from checkpoint: {cfg.n_routes} → {ckpt_routes}")
                cfg.n_routes = ckpt_routes
        del _tmp
        print(f"[probe] num_speakers overridden from checkpoint → {cfg.num_speakers}")

    if args.topk > 0:
        print(f"[probe] topk overridden: {cfg.topk} → {args.topk}")
        cfg.topk = args.topk

    # Build and load model (frozen for extraction)
    model = build_dis_model(cfg)
    _load_stage1_checkpoint(Path(args.stage1_ckpt), model, cfg)
    has_routing = False

    if args.stage2_ckpt:
        ckpt = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
        missing, _ = model.load_state_dict(ckpt["model_state"], strict=False)
        non_spear  = [k for k in missing if not k.startswith("encoder._spear.")]
        if non_spear:
            print(f"[probe] WARNING missing keys: {non_spear[:5]}")
        print(f"[probe] loaded stage2 weights from {args.stage2_ckpt}")
        has_routing = True

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    D, K, num_spk, pr_vocab_size = cfg.D, cfg.K, cfg.num_speakers, pr_cfg.vocab_size

    # Sources and their input dimensions
    sources: List[str] = ["h_t", "z_t"]
    dims: Dict[str, int] = {"h_t": D, "z_t": K, "z_L": K, "z_P": K}
    if has_routing:
        sources += ["z_L", "z_P"]

    tasks = ["pr", "sid"]

    print(f"\n[probe] speakers={num_spk}  pr_vocab={pr_vocab_size}  D={D}  K={K}")
    print("[probe] PR data/head: Probing/pr SUPERB-style loader + CTC projector head")
    print("[probe] SID head: Probing/sid SUPERB-style projector + masked mean + linear")
    print(f"[probe] sources={sources}  tasks={tasks}\n")

    results: Dict[str, Dict[str, float]] = {}

    for src in sources:
        results[src] = {}
        in_dim = dims[src]
        for task in tasks:
            label = f"{src} → {task.upper()}"
            print(f"  training probe: {label} ...", flush=True)

            if task == "sid":
                probe = _SIDProbe(in_dim, num_spk).to(device)
                _train_sid_probe(
                    probe, src, sid_train_dl, model, device, use_bf16, has_routing,
                    steps=args.probe_steps, lr=args.sid_probe_lr,
                    warmup_steps=args.probe_warmup_steps,
                    grad_clip=args.probe_grad_clip,
                )
                score = _eval_sid_probe(
                    probe, src, sid_val_dl, model, device, use_bf16, has_routing
                )
            else:
                probe = _PRProbe(in_dim, pr_vocab_size).to(device)
                _train_pr_probe(
                    probe, src, pr_train_dl, model, device, use_bf16, has_routing,
                    steps=args.probe_steps, lr=args.pr_probe_lr,
                    warmup_steps=args.probe_warmup_steps,
                    grad_clip=args.probe_grad_clip,
                )
                score = _eval_pr_probe(
                    probe, src, pr_val_dl, model, pr_tokenizer, device, use_bf16,
                    has_routing,
                )
            results[src][task] = score
            metric = f"PER={score:.3f}" if task == "pr" else f"acc={score:.3f}"
            print(f"    {label:<22s}  {metric}", flush=True)

    # Results table
    print(f"\n{'='*60}")
    print(f"  PROBE RESULTS — {args.run_name}")
    print(f"{'='*60}")
    print(f"  {'Source':<8s}  {'PR (PER↓)':>12s}  {'SID (acc↑)':>12s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}")
    for src in sources:
        per = results[src].get("pr", float('nan'))
        acc = results[src].get("sid", float('nan'))
        flag = ""
        if src == "z_L" and results[src].get("sid", 0) > results.get("z_t", {}).get("sid", 0) * 0.8:
            flag = "  ← leakage?"
        if src == "z_P" and results[src].get("pr", 1) < results.get("h_t", {}).get("pr", 1) * 1.5:
            flag = "  ← leakage?"
        print(f"  {src:<8s}  {per:>12.3f}  {acc:>12.3f}{flag}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
