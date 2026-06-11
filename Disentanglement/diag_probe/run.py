#!/usr/bin/env python3
"""Seeded diagnostic probing for Disentanglement checkpoints.

This runner is intentionally separate from the training code and the historical
probe runner.  It is a cheap, reproducible diagnostic probe, not an official
SUPERB run:

  * PR uses the SUPERB PR phone tokenizer/head/data split, evaluated on dev-clean.
  * SID uses the LibriSpeech diagnostic speaker split, with a SUPERB-style head.
  * h_t is excluded by default because it is fixed across checkpoints.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

DIAG_DIR = Path(__file__).resolve().parent
DIS_DIR = DIAG_DIR.parent
REPO_ROOT = DIS_DIR.parent


def _prioritize_import_paths() -> None:
    """Keep Disentanglement imports ahead of Probing's top-level model.py."""
    dis_path = str(DIS_DIR)
    pr_path = str(REPO_ROOT / "Probing" / "pr")
    for path in (dis_path, pr_path):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, dis_path)
    sys.path.insert(1, pr_path)


_prioritize_import_paths()

from diag_probe import probe_runner as base_probe
from config import DISConfig
from pr_config import PRConfig


VALID_SOURCES = ("h_t", "z_t", "z_L", "z_P")


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _parse_sources(raw: str) -> List[str]:
    sources = [s.strip() for s in raw.split(",") if s.strip()]
    bad = [s for s in sources if s not in VALID_SOURCES]
    if bad:
        raise ValueError(f"Unknown source(s): {bad}. Valid: {VALID_SOURCES}")
    if not sources:
        raise ValueError("At least one source must be requested.")
    return sources


def _parse_args():
    cfg = DISConfig()
    p = argparse.ArgumentParser(description="Seeded diagnostic probe for Disentanglement.")
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--stage2_ckpt", default=None)
    p.add_argument("--run_name", default="diag_probe")
    p.add_argument("--sources", default="z_t,z_L,z_P",
                   help="Comma-separated sources. Default excludes fixed h_t.")
    p.add_argument("--tasks", default="pr,sid",
                   help="Comma-separated tasks from {pr,sid}.")
    p.add_argument("--probe_steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk", type=int, default=0)
    p.add_argument("--librispeech_cache_dir", default=str(cfg.librispeech_cache_dir))
    p.add_argument("--lexicon_path", default=str(cfg.lexicon_path))
    p.add_argument("--max_train_examples", type=int, default=0)
    p.add_argument("--max_val_examples", type=int, default=500)
    p.add_argument("--pr_max_examples", type=int, default=0)
    p.add_argument("--pr_label_set", choices=("superb", "dis"), default="superb",
                   help="PR probe labels: 'superb'=74-phone dev-clean, 'dis'=internal 41-phone stage2 val.")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pr_probe_lr", type=float, default=5e-4)
    p.add_argument("--sid_probe_lr", type=float, default=1e-3)
    p.add_argument("--probe_warmup_steps", type=int, default=0)
    p.add_argument("--probe_grad_clip", type=float, default=1.0)
    p.add_argument("--standardize_sources", action="store_true",
                   help="Per-dim z-score z_L/z_P before probing (diagnoses feature-scale artifacts).")
    p.add_argument("--spear_layernorm", action="store_true",
                   help="LayerNorm each SPEAR layer before averaging — MUST match how the checkpoint "
                        "was trained, else h_t (and z_t/z_L/z_P) won't match.")
    p.add_argument("--instance_norm_zL", action="store_true",
                   help="Instance-normalize z_L over time — MUST match training (IN has no params, "
                        "so it is not stored in the checkpoint).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    sources = _parse_sources(args.sources)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad_tasks = [t for t in tasks if t not in ("pr", "sid")]
    if bad_tasks:
        raise ValueError(f"Unknown task(s): {bad_tasks}. Valid: ('pr', 'sid')")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_pr_data = __import__("pr_data")
    _prioritize_import_paths()  # pr_data prepends Probing/; restore Disentanglement first.
    base_probe._pr_data = global_pr_data
    base_probe._STANDARDIZE_SOURCES = bool(args.standardize_sources)
    if args.standardize_sources:
        print("[diag_probe] standardize_sources=True — z-scoring z_L/z_P before probing")
    from model import build_dis_model
    from train import _load_stage1_checkpoint
    from data.dataset import make_stage2_dataloaders

    cfg = DISConfig()
    cfg.device = str(device)
    cfg.spear_layernorm = bool(args.spear_layernorm)
    cfg.instance_norm_zL = bool(args.instance_norm_zL)
    cfg.librispeech_cache_dir = Path(args.librispeech_cache_dir)
    cfg.lexicon_path = Path(args.lexicon_path)
    cfg.max_train_examples = args.max_train_examples
    cfg.max_val_examples = args.max_val_examples
    cfg.num_workers = args.num_workers

    print(f"[diag_probe] run={args.run_name}  device={device}")
    print(f"[diag_probe] seed={args.seed}")
    print(f"[diag_probe] stage1_ckpt={args.stage1_ckpt}")
    print(f"[diag_probe] stage2_ckpt={args.stage2_ckpt or '(none - baselines only)'}")
    print(f"[diag_probe] sources={sources}  tasks={tasks}")
    print(
        f"[diag_probe] probe_steps={args.probe_steps}  "
        f"pr_lr={args.pr_probe_lr}  sid_lr={args.sid_probe_lr}  "
        f"warmup={args.probe_warmup_steps}"
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if args.pr_label_set == "superb":
        pr_cfg = PRConfig()
        pr_cfg.data_cache_dir = cfg.librispeech_cache_dir
        pr_cfg.librispeech_lexicon = cfg.lexicon_path
        pr_cfg.batch_size = cfg.batch_size
        pr_cfg.eval_batch_size = cfg.eval_batch_size
        pr_cfg.num_workers = cfg.num_workers
        pr_cfg.max_examples = args.pr_max_examples
        pr_tokenizer, pr_train_dl, pr_val_dl, _ = global_pr_data.make_pr_dataloaders(pr_cfg)
        _, sid_train_dl, sid_val_dl = make_stage2_dataloaders(cfg)
        pr_vocab_size = pr_cfg.vocab_size
    else:
        _stage2_tokenizer, sid_train_dl, sid_val_dl = make_stage2_dataloaders(cfg)
        pr_tokenizer = None
        pr_train_dl = sid_train_dl
        pr_val_dl = sid_val_dl
        pr_vocab_size = cfg.vocab_size

    if args.stage2_ckpt:
        tmp = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
        state = tmp["model_state"]
        cfg.num_speakers = state["sid_head.net.2.weight"].shape[0]
        ckpt_K = state["sae.enc_weight"].shape[0]
        if ckpt_K != cfg.K:
            print(f"[diag_probe] K overridden from checkpoint: {cfg.K} -> {ckpt_K}")
            cfg.K = ckpt_K
        if "routing.logits" in state:
            ckpt_routes = state["routing.logits"].shape[1]
            if ckpt_routes != cfg.n_routes:
                print(f"[diag_probe] n_routes overridden: {cfg.n_routes} -> {ckpt_routes}")
                cfg.n_routes = ckpt_routes
        if "proj_L.proj.weight" in state:
            cfg.projection_disentanglement = True
            ckpt_dim = state["proj_L.proj.weight"].shape[0]
            if ckpt_dim != cfg.projection_dim:
                print(f"[diag_probe] projection_dim overridden: {cfg.projection_dim} -> {ckpt_dim}")
                cfg.projection_dim = ckpt_dim
            print("[diag_probe] projection_disentanglement enabled from checkpoint")
        del tmp

    if args.topk > 0:
        print(f"[diag_probe] topk overridden: {cfg.topk} -> {args.topk}")
        cfg.topk = args.topk

    model = build_dis_model(cfg)
    _load_stage1_checkpoint(Path(args.stage1_ckpt), model, cfg)
    has_routing = False
    if args.stage2_ckpt:
        ckpt = torch.load(args.stage2_ckpt, map_location=device, weights_only=False)
        # Tolerate head-architecture drift across commits.  The probe never uses
        # the adversarial heads (grl_head / pr_grl_head) for evaluation, so skip
        # any checkpoint tensor whose shape no longer matches the current model
        # (e.g. old single-Linear GRL heads vs new projector+classifier heads).
        # strict=False does NOT suppress shape mismatches — it raises — so filter
        # by shape first, otherwise re-probing any pre-merge checkpoint crashes.
        model_sd = model.state_dict()
        ckpt_state = ckpt["model_state"]
        filtered = {k: v for k, v in ckpt_state.items()
                    if k in model_sd and model_sd[k].shape == v.shape}
        skipped = [k for k in ckpt_state if k not in filtered]
        missing, _ = model.load_state_dict(filtered, strict=False)
        non_spear = [k for k in missing if not k.startswith("encoder._spear.")]
        if skipped:
            print(f"[diag_probe] skipped {len(skipped)} shape-mismatched/stale keys "
                  f"(e.g. {skipped[:3]}) — fine, probe does not use adversarial heads")
        if non_spear:
            print(f"[diag_probe] WARNING missing keys: {non_spear[:5]}")
        print(f"[diag_probe] loaded stage2 weights from {args.stage2_ckpt}")
        has_routing = True

    if not has_routing and any(s in sources for s in ("z_L", "z_P")):
        raise ValueError("z_L/z_P require --stage2_ckpt.")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    view_dim = cfg.projection_dim if cfg.projection_disentanglement else cfg.K
    dims: Dict[str, int] = {"h_t": cfg.D, "z_t": cfg.K, "z_L": view_dim, "z_P": view_dim}
    results: Dict[str, Dict[str, float]] = {}

    print(f"\n[diag_probe] speakers={cfg.num_speakers}  pr_vocab={pr_vocab_size}  D={cfg.D}  K={cfg.K}")
    if args.pr_label_set == "superb":
        print("[diag_probe] PR: SUPERB phone data/head, dev-clean diagnostic eval")
    else:
        print("[diag_probe] PR: Disentanglement 41-phone labels, stage2 val diagnostic eval")
    print("[diag_probe] SID: LibriSpeech diagnostic split, SUPERB-style SID head\n")

    for src in sources:
        results[src] = {}
        for task in tasks:
            label = f"{src} -> {task.upper()}"
            print(f"  training probe: {label} ...", flush=True)
            if task == "sid":
                probe = base_probe._SIDProbe(dims[src], cfg.num_speakers).to(device)
                base_probe._train_sid_probe(
                    probe, src, sid_train_dl, model, device, use_bf16, has_routing,
                    steps=args.probe_steps, lr=args.sid_probe_lr,
                    warmup_steps=args.probe_warmup_steps,
                    grad_clip=args.probe_grad_clip,
                )
                score = base_probe._eval_sid_probe(
                    probe, src, sid_val_dl, model, device, use_bf16, has_routing
                )
            else:
                probe = base_probe._PRProbe(dims[src], pr_vocab_size).to(device)
                base_probe._train_pr_probe(
                    probe, src, pr_train_dl, model, device, use_bf16, has_routing,
                    steps=args.probe_steps, lr=args.pr_probe_lr,
                    warmup_steps=args.probe_warmup_steps,
                    grad_clip=args.probe_grad_clip,
                )
                if args.pr_label_set == "superb":
                    score = base_probe._eval_pr_probe(
                        probe, src, pr_val_dl, model, pr_tokenizer, device, use_bf16, has_routing
                    )
                else:
                    score = base_probe._eval_pr_probe_ids(
                        probe, src, pr_val_dl, model, device, use_bf16, has_routing
                    )
            results[src][task] = score
            metric = f"PER={score:.3f}" if task == "pr" else f"acc={score:.3f}"
            print(f"    {label:<22s}  {metric}", flush=True)

    print(f"\n{'=' * 66}")
    print(f"  DIAGNOSTIC PROBE RESULTS - {args.run_name}")
    print(f"{'=' * 66}")
    print(f"  {'Source':<8s}  {'PR (PER↓)':>12s}  {'SID (acc↑)':>12s}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}")
    for src in sources:
        per = results[src].get("pr", float("nan"))
        acc = results[src].get("sid", float("nan"))
        print(f"  {src:<8s}  {per:>12.3f}  {acc:>12.3f}")
    print(f"{'=' * 66}\n")


if __name__ == "__main__":
    main()
