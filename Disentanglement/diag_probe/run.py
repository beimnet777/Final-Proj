#!/usr/bin/env python3
"""Seeded diagnostic probing for Disentanglement checkpoints.

This runner is intentionally separate from the training code and the historical
probe runner.  It is a cheap, reproducible diagnostic probe, not an official
SUPERB run:

  * PR: SUPERB 74-phone — train on train-clean-100, validate on dev-clean,
    report on test-clean.
  * SID: closed-set over train-clean-100 speakers, split by utterance into
    train / val / test; reported metric is the held-out test split.
  * h_t is excluded by default because it is fixed across checkpoints.
"""

from __future__ import annotations

import argparse
import os
import random
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# The frozen SPEAR remote implementation still calls deprecated torch.cuda.amp
# aliases on every forward. They are harmless but otherwise flood long probe
# logs with thousands of identical warnings. Keep project warnings visible.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.(autocast|custom_fwd|custom_bwd).*deprecated.*",
)

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


VALID_SOURCES = ("h_t", "z_t", "z_L", "z_P", "z_U")


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
    p.add_argument("--output_json", default="",
                   help="optional machine-readable result path")
    p.add_argument("--sources", default="z_t,z_L,z_P",
                   help="Comma-separated sources. Default excludes fixed h_t.")
    p.add_argument("--tasks", default="pr,sid",
                   help="Comma-separated tasks from {pr,sid,emotion}. Prosody is toggled by --prosody.")
    p.add_argument("--prosody", action=argparse.BooleanOptionalAction, default=cfg.prosody,
                   help="Master switch: add the per-frame log-F0 + log-energy prosody probe "
                        "(off by default — experiments run without prosody unless this is set).")
    p.add_argument("--probe_steps", type=int, default=2000,
                   help="max probe steps (early stopping may end sooner)")
    p.add_argument("--probe_val_every", type=int, default=250,
                   help="eval probe on dev every N steps for early stopping (0 = off, fixed steps)")
    p.add_argument("--probe_patience", type=int, default=5,
                   help="early-stop after this many dev evals with no improvement (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk", type=int, default=0)
    p.add_argument("--librispeech_cache_dir", default=str(cfg.librispeech_cache_dir))
    p.add_argument("--lexicon_path", default=str(cfg.lexicon_path))
    p.add_argument("--local_data", action="store_true",
                   help="read LibriSpeech from --librispeech_root instead of Hugging Face streaming")
    p.add_argument("--librispeech_root", default=str(cfg.librispeech_root))
    p.add_argument("--max_train_examples", type=int, default=0)
    p.add_argument("--max_val_examples", type=int, default=500)
    p.add_argument("--max_test_examples", type=int, default=500)
    p.add_argument("--pr_max_examples", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pr_probe_lr", type=float, default=1e-4)
    p.add_argument("--sid_probe_lr", type=float, default=1e-3)
    p.add_argument("--prosody_probe_lr", type=float, default=5e-4)
    p.add_argument("--emotion", action=argparse.BooleanOptionalAction, default=False,
                   help="add the IEMOCAP emotion probe")
    p.add_argument("--emotion_probe_lr", type=float, default=5e-4)
    p.add_argument("--iemocap_root", default=str(cfg.iemocap_root),
                   help="path to extracted IEMOCAP_full_release for emotion probing")
    p.add_argument("--iemocap_fold", type=int, default=cfg.iemocap_fold)
    p.add_argument("--iemocap_batch_size", type=int, default=cfg.iemocap_batch_size)
    p.add_argument("--iemocap_eval_batch_size", type=int, default=cfg.iemocap_eval_batch_size)
    p.add_argument("--prosody_max_train", type=int, default=2000,
                   help="Utterances in the prosody probe train pool (bounds one-time pyin F0 cost).")
    p.add_argument("--sid_probe_arch", choices=("linear", "stats", "mlp"), default="linear",
                   help="SID probe: 'linear'=projector->mean-pool->linear (SUPERB-style; blind to "
                        "instance-normed features); 'stats'=projector->ReLU->mean+std pool->linear; "
                        "'mlp'=projector->ReLU->mean-pool->linear (SUPERB + one ReLU).")
    p.add_argument("--pr_probe_arch", choices=("linear", "mlp"), default="linear",
                   help="PR probe: 'linear'=projector->linear (SUPERB-style); "
                        "'mlp'=projector->ReLU->linear (SUPERB + one ReLU).")
    p.add_argument("--sid_dataset", choices=("libri", "arctic"), default="libri",
                   help="SID probe data source. 'libri'=LibriSpeech 251 speakers (default, leakage); "
                        "'arctic'=CMU ARCTIC 18 speakers (matched-distribution check for invariance runs).")
    p.add_argument("--arctic_root", type=str,
                   default="../Probing/data/CMU_ARCTIC",
                   help="ARCTIC root containing cmu_us_<spk>_arctic/wav/arctic_*.wav (used when --sid_dataset arctic).")
    p.add_argument("--arctic_sid_seed", type=int, default=42,
                   help="Seed for the random per-speaker utterance split when --sid_dataset arctic.")
    # Prequential MDL probe (Voita & Titov 2020, EMNLP).  Runs in addition to
    # the standard accuracy/PER probe and reports codelength in kbits and
    # compression-over-uniform.  No effect on the existing leakage table.
    p.add_argument("--mdl_probe", action=argparse.BooleanOptionalAction, default=False,
                   help="Also run a prequential MDL probe per (source, task).")
    p.add_argument("--mdl_only", action=argparse.BooleanOptionalAction, default=False,
                   help="Skip the standard accuracy/PER probe and report only MDL "
                        "codelength. Implies --mdl_probe. Use when standard probe "
                        "numbers already exist and only the matched-protocol "
                        "prequential metric is needed.")
    p.add_argument("--mdl_steps_per_block", type=int, default=1250,
                   help="Probe optimisation steps per MDL block (8 phases x 1250 ~= 10k updates).")
    p.add_argument("--mdl_max_train_examples", type=int, default=4000,
                   help="Cap on cached train examples for the MDL probe (keeps memory bounded).")
    p.add_argument("--probe_warmup_steps", type=int, default=0)
    p.add_argument("--probe_grad_clip", type=float, default=1.0)
    p.add_argument("--spear_layernorm", action="store_true",
                   help="LayerNorm each SPEAR layer before averaging — MUST match how the checkpoint "
                        "was trained, else h_t (and z_t/z_L/z_P) won't match.")
    p.add_argument("--instance_norm_zL", action="store_true",
                   help="Instance-normalize z_L over time — MUST match training (IN has no params, "
                        "so it is not stored in the checkpoint).")
    p.add_argument("--vib_zL_weight", type=float, default=0.0,
                   help="Match training: builds the vib_logvar param (eval uses the mean, no noise).")
    p.add_argument("--vib_zL_layernorm", action="store_true",
                   help="Match training: param-free LayerNorm on z_L before VIB.")
    p.add_argument("--hard_gumbel_routing", action=argparse.BooleanOptionalAction, default=True,
                   help="Routing eval mode — MUST match training (hard argmax vs soft fractional masks).")
    p.add_argument("--fixed_blocks", action="store_true",
                   help="Option A: fixed L/P/U blocks — MUST match training.")
    # Projection mode (z_L/z_P are learned views of z_t) — MUST match training so
    # the probe rebuilds proj_L/proj_P and reads the correct z_L/z_P.
    p.add_argument("--projection_disentanglement", action="store_true")
    p.add_argument("--projection_reconstruct", action="store_true")
    p.add_argument("--projection_nonlinear", action="store_true")
    p.add_argument("--projection_dim", type=int, default=cfg.projection_dim)
    p.add_argument("--projection_hidden", type=int, default=cfg.projection_hidden)
    p.add_argument("--projection_u_dim", type=int, default=cfg.projection_u_dim)
    p.add_argument("--per_block_topk", action=argparse.BooleanOptionalAction, default=cfg.per_block_topk,
                   help="MUST match training: per-block TopK vs global TopK.")
    p.add_argument("--K_L",    type=int, default=cfg.K_L)
    p.add_argument("--K_P",    type=int, default=cfg.K_P)
    p.add_argument("--K_U",    type=int, default=cfg.K_U)
    p.add_argument("--topk_L", type=int, default=cfg.topk_L)
    p.add_argument("--topk_P", type=int, default=cfg.topk_P)
    p.add_argument("--topk_U", type=int, default=cfg.topk_U)
    p.add_argument("--gumbel_tau_start", type=float, default=cfg.gumbel_tau_start,
                   help="Accepted for parity with training scripts; eval uses tau_end for soft routing.")
    p.add_argument("--gumbel_tau_end", type=float, default=0.1,
                   help="Soft-routing eval temperature — match training's final tau.")
    p.add_argument("--n_routes", type=int, default=cfg.n_routes,
                   help="Routing factors (2=z_L,z_P; 3=+z_U). Auto-overridden from the checkpoint if they differ.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    sources = _parse_sources(args.sources)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad_tasks = [t for t in tasks if t not in ("pr", "sid", "prosody", "emotion")]
    if bad_tasks:
        raise ValueError(f"Unknown task(s): {bad_tasks}. Valid: ('pr', 'sid', 'prosody', 'emotion')")
    # The --prosody switch is authoritative: --prosody adds it, --no-prosody removes
    # it (even if listed in --tasks).  Default off → experiments run without prosody.
    if args.prosody and "prosody" not in tasks:
        tasks.append("prosody")
    elif not args.prosody and "prosody" in tasks:
        tasks.remove("prosody")
    if args.emotion and "emotion" not in tasks:
        tasks.append("emotion")
    # --mdl_only implies --mdl_probe (the MDL run is the whole point).
    if args.mdl_only:
        args.mdl_probe = True
        if "prosody" in tasks:
            print("[diag_probe] --mdl_only: prosody not supported by MDL probe, dropping.")
            tasks.remove("prosody")
        if "emotion" in tasks:
            print("[diag_probe] --mdl_only: emotion not supported by MDL probe, dropping.")
            tasks.remove("emotion")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from model import build_dis_model
    from train import _load_stage1_checkpoint
    from data.dataset import make_stage2_dataloaders

    cfg = DISConfig()
    cfg.device = str(device)
    cfg.spear_layernorm = bool(args.spear_layernorm)
    cfg.local_data = bool(args.local_data)
    cfg.librispeech_root = Path(args.librispeech_root)
    cfg.instance_norm_zL = bool(args.instance_norm_zL)
    cfg.hard_gumbel_routing = bool(args.hard_gumbel_routing)
    cfg.gumbel_tau_start = args.gumbel_tau_start
    cfg.gumbel_tau_end = args.gumbel_tau_end
    cfg.n_routes = args.n_routes
    cfg.fixed_blocks = bool(args.fixed_blocks)
    cfg.per_block_topk = bool(args.per_block_topk)
    cfg.vib_zL_weight = args.vib_zL_weight
    cfg.vib_zL_layernorm = bool(args.vib_zL_layernorm)
    cfg.projection_disentanglement = bool(args.projection_disentanglement)
    cfg.projection_reconstruct     = bool(args.projection_reconstruct)
    cfg.projection_nonlinear       = bool(args.projection_nonlinear)
    cfg.projection_dim             = args.projection_dim
    cfg.projection_hidden          = args.projection_hidden
    cfg.projection_u_dim           = args.projection_u_dim
    cfg.prosody = bool(args.prosody)
    cfg.emotion = bool(args.emotion or "emotion" in tasks)
    cfg.iemocap_root = Path(args.iemocap_root)
    cfg.iemocap_fold = args.iemocap_fold
    cfg.iemocap_batch_size = args.iemocap_batch_size
    cfg.iemocap_eval_batch_size = args.iemocap_eval_batch_size
    cfg.K_L, cfg.K_P, cfg.K_U = args.K_L, args.K_P, args.K_U
    cfg.topk_L, cfg.topk_P, cfg.topk_U = args.topk_L, args.topk_P, args.topk_U
    cfg.librispeech_cache_dir = Path(args.librispeech_cache_dir)
    cfg.lexicon_path = Path(args.lexicon_path)
    cfg.max_train_examples = args.max_train_examples
    cfg.max_val_examples = args.max_val_examples
    cfg.max_test_examples = args.max_test_examples
    cfg.num_workers = args.num_workers

    print(f"[diag_probe] run={args.run_name}  device={device}")
    print(f"[diag_probe] seed={args.seed}")
    print(f"[diag_probe] stage1_ckpt={args.stage1_ckpt}")
    print(f"[diag_probe] stage2_ckpt={args.stage2_ckpt or '(none - baselines only)'}")
    print(f"[diag_probe] sources={sources}  tasks={tasks}")
    print(
        f"[diag_probe] probe_steps={args.probe_steps}  "
        f"pr_lr={args.pr_probe_lr}  sid_lr={args.sid_probe_lr}  "
        f"warmup={args.probe_warmup_steps}  "
        f"val_every={args.probe_val_every}  patience={args.probe_patience}"
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # PR: SUPERB 74-phone — train-clean-100 → train, dev-clean → val, test-clean → test.
    # Build lazily so SID-only recovery probes do not touch PR data.
    pr_tokenizer = pr_train_dl = pr_val_dl = pr_test_dl = None
    pr_vocab_size = cfg.vocab_size
    if "pr" in tasks:
        global_pr_data = __import__("pr_data")
        _prioritize_import_paths()  # pr_data prepends Probing/; restore Disentanglement first.
        base_probe._pr_data = global_pr_data
        pr_cfg = PRConfig()
        pr_cfg.data_cache_dir = cfg.librispeech_cache_dir
        pr_cfg.librispeech_lexicon = cfg.lexicon_path
        pr_cfg.local_data = cfg.local_data
        pr_cfg.librispeech_root = cfg.librispeech_root
        pr_cfg.batch_size = cfg.batch_size
        pr_cfg.eval_batch_size = cfg.eval_batch_size
        pr_cfg.num_workers = cfg.num_workers
        pr_cfg.max_examples = args.pr_max_examples
        pr_tokenizer, pr_train_dl, pr_val_dl, pr_test_dl = global_pr_data.make_pr_dataloaders(pr_cfg)
        pr_vocab_size = pr_cfg.vocab_size
    else:
        print("[diag_probe] PR task not requested — skipping PR dataloaders.")
    # SID: closed-set — same speakers, split by utterance into train / val / test.
    _, sid_train_dl, sid_val_dl, sid_test_dl = make_stage2_dataloaders(cfg)
    emo_train_dl = emo_val_dl = emo_test_dl = None
    if "emotion" in tasks:
        from data.iemocap_emotion import make_iemocap_emotion_dataloaders
        emo_train_dl, emo_val_dl, emo_test_dl = make_iemocap_emotion_dataloaders(cfg)
    # Optional override: matched-distribution probe on ARCTIC's 18 speakers.
    sid_num_classes_override = None
    if args.sid_dataset == "arctic":
        from data.arctic_sid import make_arctic_sid_dataloaders
        sid_num_classes_override, sid_train_dl, sid_val_dl, sid_test_dl = (
            make_arctic_sid_dataloaders(
                arctic_root=args.arctic_root,
                sample_rate=cfg.sample_rate,
                batch_size=cfg.batch_size,
                eval_batch_size=cfg.eval_batch_size,
                num_workers=cfg.num_workers,
                seed=args.arctic_sid_seed,
            )
        )
        print(f"[diag_probe] SID dataset = ARCTIC ({sid_num_classes_override} speakers, seed {args.arctic_sid_seed})")

    if args.stage2_ckpt:
        tmp = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
        state = tmp["model_state"]
        cfg.num_speakers = state["sid_head.fc.weight"].shape[0]
        if "pr_head.fc.weight" in state:
            cfg.vocab_size = state["pr_head.fc.weight"].shape[0]
            if "pr" not in tasks:
                pr_vocab_size = cfg.vocab_size
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
        # Prosody is only meaningful if the model was TRAINED with it (the
        # checkpoint then carries prosody_head weights).  Gate the prosody probe
        # on that, so we never report prosody numbers for a model that never
        # learned to put prosody anywhere.
        ckpt_has_prosody = any(k.startswith("prosody_head") for k in state)
        if "prosody" in tasks and not ckpt_has_prosody:
            print("[diag_probe] checkpoint has no prosody head (prosody not trained) "
                  "— skipping the prosody probe.")
            tasks.remove("prosody")
        del tmp
    elif "prosody" in tasks:
        # No stage2 checkpoint → baseline only → prosody was never trained.
        print("[diag_probe] no stage2 checkpoint — skipping the prosody probe.")
        tasks.remove("prosody")

    if args.topk > 0:
        if args.topk != cfg.topk:
            print(f"[diag_probe] topk overridden: {cfg.topk} -> {args.topk}")
        else:
            print(f"[diag_probe] topk confirmed: {args.topk}")
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
    dims: Dict[str, int] = {"h_t": cfg.D, "z_t": cfg.K, "z_L": view_dim,
                            "z_P": view_dim, "z_U": view_dim}
    results: Dict[str, Dict[str, float]] = {}

    print(f"\n[diag_probe] speakers={cfg.num_speakers}  pr_vocab={pr_vocab_size}  D={cfg.D}  K={cfg.K}")
    if "pr" in tasks:
        print("[diag_probe] PR: SUPERB 74-phone — train-clean-100 → val=dev-clean, test=test-clean")
    else:
        print("[diag_probe] PR: skipped")
    print(f"[diag_probe] SID: closed-set utterance split (val/test held out); probe arch={args.sid_probe_arch}")
    if "emotion" in tasks:
        print(f"[diag_probe] Emotion: IEMOCAP fold={cfg.iemocap_fold}  root={cfg.iemocap_root}")
    print("[diag_probe] reported metric = TEST (val shown in parentheses)\n")

    # Prosody: build the audio+target train pool ONCE (pyin F0 is slow; this caches
    # it).  Targets are source-independent, so the same pool feeds every bucket.
    prosody_pool = None
    if "prosody" in tasks:
        print(f"[diag_probe] building prosody train pool (<= {args.prosody_max_train} utts, one-time pyin) ...",
              flush=True)
        prosody_pool = base_probe._build_prosody_pool(
            sid_train_dl, model, device, use_bf16, has_routing, args.prosody_max_train)
        print(f"[diag_probe] prosody pool: {len(prosody_pool)} batches", flush=True)

    for src in sources:
        results[src] = {}
        for task in tasks:
            label = f"{src} -> {task.upper()}"
            print(f"  training probe: {label} ...", flush=True)
            if task == "emotion":
                if emo_train_dl is None or emo_val_dl is None or emo_test_dl is None:
                    raise RuntimeError("Emotion task requested but IEMOCAP dataloaders were not built.")
                probe = base_probe._EmotionProbeStats(dims[src], cfg.emotion_num_classes).to(device)
                val_cache = base_probe._cache_emotion_features(
                    src, emo_val_dl, model, device, use_bf16, has_routing)
                test_cache = base_probe._cache_emotion_features(
                    src, emo_test_dl, model, device, use_bf16, has_routing)
                best_val = base_probe._train_emotion_probe(
                    probe, src, emo_train_dl, model, device, use_bf16, has_routing,
                    steps=args.probe_steps, lr=args.emotion_probe_lr,
                    warmup_steps=args.probe_warmup_steps,
                    grad_clip=args.probe_grad_clip,
                    val_cache=val_cache, val_every=args.probe_val_every,
                    patience=args.probe_patience,
                )
                val_score = (best_val if best_val is not None
                             else base_probe._eval_emotion_probe_cached(probe, val_cache, device))
                test_score = base_probe._eval_emotion_probe_cached(probe, test_cache, device)
                results[src]["emotion"] = test_score
                results[src]["emotion_val"] = val_score
                print(f"    {label:<22s}  acc test={test_score:.3f}  (val {val_score:.3f})", flush=True)
                continue
            if task == "prosody":
                probe = base_probe._ProsodyProbe(dims[src]).to(device)
                val_cache  = base_probe._cache_features(src, sid_val_dl,  model, device, use_bf16, has_routing, "prosody")
                test_cache = base_probe._cache_features(src, sid_test_dl, model, device, use_bf16, has_routing, "prosody")
                best = base_probe._train_prosody_probe(
                    probe, src, prosody_pool, model, device, use_bf16, has_routing,
                    steps=args.probe_steps, lr=args.prosody_probe_lr,
                    warmup_steps=args.probe_warmup_steps,
                    grad_clip=args.probe_grad_clip,
                    val_cache=val_cache, val_every=args.probe_val_every,
                    patience=args.probe_patience,
                )
                f0c, ec = base_probe._eval_prosody_probe_cached(probe, test_cache, device)
                vf0, ve = best if best is not None else base_probe._eval_prosody_probe_cached(probe, val_cache, device)
                results[src]["prosody_f0"]  = f0c
                results[src]["prosody_e"]   = ec
                results[src]["prosody_f0_val"] = vf0
                results[src]["prosody_e_val"]  = ve
                print(f"    {label:<22s}  F0 r={f0c:.3f}  E r={ec:.3f}  (val F0 {vf0:.3f}, E {ve:.3f})", flush=True)
                continue
            if not args.mdl_only:
                if task == "sid":
                    sid_map = {"stats": base_probe._SIDProbeStats,
                               "mlp":   base_probe._SIDProbeMLP,
                               "linear": base_probe._SIDProbe}
                    sid_cls = sid_map.get(args.sid_probe_arch, base_probe._SIDProbe)
                    n_sid = sid_num_classes_override if sid_num_classes_override is not None else cfg.num_speakers
                    probe = sid_cls(dims[src], n_sid).to(device)
                    # Cache dev/test features once (frozen model) → cheap repeated evals.
                    val_cache  = base_probe._cache_features(src, sid_val_dl,  model, device, use_bf16, has_routing, "sid")
                    test_cache = base_probe._cache_features(src, sid_test_dl, model, device, use_bf16, has_routing, "sid")
                    best_val = base_probe._train_sid_probe(
                        probe, src, sid_train_dl, model, device, use_bf16, has_routing,
                        steps=args.probe_steps, lr=args.sid_probe_lr,
                        warmup_steps=args.probe_warmup_steps,
                        grad_clip=args.probe_grad_clip,
                        val_cache=val_cache, val_every=args.probe_val_every,
                        patience=args.probe_patience,
                    )
                    val_score  = (best_val if best_val is not None
                                  else base_probe._eval_sid_probe_cached(probe, val_cache, device))
                    test_score = base_probe._eval_sid_probe_cached(probe, test_cache, device)
                else:
                    if pr_train_dl is None or pr_val_dl is None or pr_test_dl is None or pr_tokenizer is None:
                        raise RuntimeError("PR task requested but PR dataloaders were not built.")
                    pr_cls = (base_probe._PRProbeMLP if args.pr_probe_arch == "mlp"
                              else base_probe._PRProbe)
                    probe = pr_cls(dims[src], pr_vocab_size).to(device)
                    val_cache  = base_probe._cache_features(src, pr_val_dl,  model, device, use_bf16, has_routing, "pr")
                    test_cache = base_probe._cache_features(src, pr_test_dl, model, device, use_bf16, has_routing, "pr")
                    best_val = base_probe._train_pr_probe(
                        probe, src, pr_train_dl, model, device, use_bf16, has_routing,
                        steps=args.probe_steps, lr=args.pr_probe_lr,
                        warmup_steps=args.probe_warmup_steps,
                        grad_clip=args.probe_grad_clip,
                        val_cache=val_cache, tokenizer=pr_tokenizer,
                        val_every=args.probe_val_every, patience=args.probe_patience,
                    )
                    val_score  = (best_val if best_val is not None
                                  else base_probe._eval_pr_probe_cached(probe, val_cache, pr_tokenizer, device))
                    test_score = base_probe._eval_pr_probe_cached(probe, test_cache, pr_tokenizer, device)
                results[src][task] = test_score          # reported metric = TEST
                results[src][task + "_val"] = val_score
                unit = "PER" if task == "pr" else "acc"
                print(f"    {label:<22s}  {unit} test={test_score:.3f}  (val {val_score:.3f})", flush=True)

            if args.mdl_probe:
                # Prequential MDL (Voita & Titov 2020): codelength under a
                # sequence of probes trained on growing prefixes, evaluated on
                # the next slice.  Reported in kbits and compression-over-
                # uniform; orthogonal to the accuracy number above.
                num_cls   = ((sid_num_classes_override if sid_num_classes_override is not None else cfg.num_speakers)
                             if task == "sid" else pr_vocab_size)
                mdl_lr    = args.sid_probe_lr if task == "sid" else args.pr_probe_lr
                if task == "pr" and pr_train_dl is None:
                    raise RuntimeError("PR MDL requested but PR dataloader was not built.")
                mdl_train = sid_train_dl     if task == "sid" else pr_train_dl
                mdl = base_probe.run_mdl_probe(
                    src_key=src, task=task, in_dim=dims[src], num_classes=num_cls,
                    train_dl=mdl_train, model=model, device=device,
                    use_bf16=use_bf16, has_routing=has_routing,
                    lr=mdl_lr, steps_per_block=args.mdl_steps_per_block,
                    max_train_examples=args.mdl_max_train_examples,
                    sid_probe_arch=args.sid_probe_arch,
                    pr_probe_arch=args.pr_probe_arch,
                )
                results[src][task + "_mdl_kbits"]      = mdl["codelength_kbits"]
                results[src][task + "_mdl_uniform_kbits"] = mdl["uniform_kbits"]
                results[src][task + "_mdl_compression"] = mdl["compression"]
                results[src][task + "_mdl_n"]          = mdl["n_examples"]
                print(f"    {label:<22s}  MDL kbits={mdl['codelength_kbits']:.2f} / "
                      f"uniform={mdl['uniform_kbits']:.2f}  "
                      f"compression={mdl['compression']*100:.1f}%  "
                      f"(n={mdl['n_examples']}, {mdl['n_blocks']} blocks)", flush=True)

    has_prosody = "prosody" in tasks
    has_emotion = "emotion" in tasks
    width = 66 + (28 if has_prosody else 0) + (15 if has_emotion else 0)
    if args.mdl_only:
        # Skip the standard-probe summary table; the MDL table below is the
        # only output relevant in this mode.
        print(f"\n[diag_probe] --mdl_only: standard accuracy/PER probe skipped; "
              f"see existing logs for those numbers.")
    else:
        print(f"\n{'=' * width}")
        print(f"  DIAGNOSTIC PROBE RESULTS - {args.run_name}")
        print(f"{'=' * width}")
        header = f"  {'Source':<8s}  {'PR (PER↓)':>12s}  {'SID (acc↑)':>12s}"
        if has_prosody:
            header += f"  {'F0 (r↑)':>12s}  {'E (r↑)':>12s}"
        if has_emotion:
            header += f"  {'EMO (acc↑)':>12s}"
        print(header)
        dashes = f"  {'-' * 8}  {'-' * 12}  {'-' * 12}"
        if has_prosody:
            dashes += f"  {'-' * 12}  {'-' * 12}"
        if has_emotion:
            dashes += f"  {'-' * 12}"
        print(dashes)
        for src in sources:
            per = results[src].get("pr", float("nan"))
            acc = results[src].get("sid", float("nan"))
            row = f"  {src:<8s}  {per:>12.3f}  {acc:>12.3f}"
            if has_prosody:
                f0c = results[src].get("prosody_f0", float("nan"))
                ec  = results[src].get("prosody_e", float("nan"))
                row += f"  {f0c:>12.3f}  {ec:>12.3f}"
            if has_emotion:
                emo = results[src].get("emotion", float("nan"))
                row += f"  {emo:>12.3f}"
            print(row)
        print(f"{'=' * width}\n")

    if args.mdl_probe:
        mwidth = 80
        print(f"{'=' * mwidth}")
        print(f"  MDL CODELENGTH (prequential, fixed optimisation protocol) - {args.run_name}")
        print(f"{'=' * mwidth}")
        print(f"  {'Source':<8s}  {'Task':<4s}  {'kbits ↓':>10s}  "
              f"{'uniform':>10s}  {'compr% ↑':>10s}  {'n':>6s}")
        print(f"  {'-'*8}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")
        for src in sources:
            for task in ("pr", "sid"):
                k = task + "_mdl_kbits"
                if k not in results[src]:
                    continue
                kb = results[src][k]
                ub = results[src].get(task + "_mdl_uniform_kbits", float("nan"))
                cp = 100.0 * results[src].get(task + "_mdl_compression", 0.0)
                n  = results[src].get(task + "_mdl_n", 0)
                print(f"  {src:<8s}  {task.upper():<4s}  {kb:>10.2f}  "
                      f"{ub:>10.2f}  {cp:>10.1f}  {n:>6d}")
        print(f"{'=' * mwidth}\n")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps({
            "run_name": args.run_name, "sources": sources, "tasks": tasks,
            "seed": args.seed, "sid_probe_arch": args.sid_probe_arch,
            "pr_probe_arch": args.pr_probe_arch, "results": results,
        }, indent=2, sort_keys=True) + "\n")
        tmp.replace(output)


if __name__ == "__main__":
    main()
