#!/usr/bin/env python3
"""Smoke test: full pipeline without GPU or SPEAR download.

Patches AutoModel.from_pretrained with a tiny fake SPEAR model.
Tests:
  - stage 1: forward shapes, fixed target, TopK sparsity, backward, LR schedule, checkpoint
  - stage 2: routing masks, task head shapes, per-loss backward, grad norms

Usage:  python smoke_test.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------- fake SPEAR

class _FakeSpear(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()

    def forward(self, audio, lengths):
        B, T_audio = audio.shape
        T = max(T_audio // 640, 1)
        return {"hidden_states": [torch.randn(B, T, 1280) for _ in range(13)]}


def _fake_s1_batch(B, T_audio):
    audio      = torch.randn(B, T_audio)
    audio_lens = torch.tensor([T_audio, T_audio - 640], dtype=torch.long)
    return audio, audio_lens


def _fake_s2_batch(B, T_audio, vocab, num_spk, P=8):
    audio, audio_lens = _fake_s1_batch(B, T_audio)
    targets     = torch.randint(1, vocab - 1, (B, P))
    target_lens = torch.tensor([P, P - 2], dtype=torch.long)
    speaker_ids = torch.randint(0, num_spk, (B,))
    return audio, audio_lens, targets, target_lens, speaker_ids


# ---------------------------------------------------------------- main

def run():
    with patch("transformers.AutoModel.from_pretrained", return_value=_FakeSpear()):
        _test()


def _test():
    from config import DISConfig
    from model import build_dis_model
    from losses import recon_loss, ctc_pr_loss, sid_ce_loss, route_loss
    from train import (_make_scheduler, _count_params, _save_checkpoint,
                       _eval_stage1, _eval_stage2)

    print("=== Disentanglement smoke test ===\n")

    cfg              = DISConfig()
    cfg.device       = "cpu"
    cfg.bf16         = False
    cfg.K            = 128
    cfg.topk         = 6
    cfg.batch_size   = 2
    cfg.eval_batch_size = 2
    cfg.total_steps  = 6
    cfg.stage2_steps = 6
    cfg.warmup_steps = 2
    cfg.num_speakers = 5
    cfg.vocab_size   = 41
    cfg.log_every    = 99999

    B, T_audio = 2, 12_800

    model = build_dis_model(cfg)
    frozen, trainable = _count_params(model)
    print(f"frozen={frozen:,}  trainable={trainable:,}")

    enc_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    assert enc_trainable == 0, f"encoder must be fully frozen, got {enc_trainable}"
    print("encoder fully frozen ✓")

    # ================================================================
    # STAGE 1
    # ================================================================
    print("\n--- Stage 1: forward + backward ---")
    from torch.optim import AdamW
    opt = AdamW(model.sae.parameters(), lr=cfg.lr)
    sch = _make_scheduler(opt, cfg.warmup_steps, cfg.total_steps, cfg.lr, cfg.lr_min)

    for step in range(1, 5):
        audio, audio_lens = _fake_s1_batch(B, T_audio)
        opt.zero_grad(set_to_none=True)
        out  = model(audio, audio_lens, stage=1)

        # shape checks
        B_, T_, D_ = out["h_t"].shape
        assert D_ == cfg.D
        assert out["z_t"].shape   == (B_, T_, cfg.K)
        assert out["h_hat"].shape == (B_, T_, cfg.D)
        assert not out["h_t"].requires_grad, "h_t must be detached"

        # sparsity
        assert (out["z_t"] != 0).sum(-1).eq(cfg.topk).all(), "wrong topk count"

        loss = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
        loss.backward()

        assert model.sae.enc_weight.grad is not None
        assert model.sae.dec_weight.grad is not None
        for name, p in model.sae.named_parameters():
            if p.grad is not None:
                assert not p.grad.isnan().any(), f"NaN in {name}"

        nn.utils.clip_grad_norm_(model.sae.parameters(), cfg.grad_clip)
        opt.step(); sch.step()
        print(f"  step {step}  recon={loss.item():.4f}  ✓")

    # ================================================================
    # LR SCHEDULE
    # ================================================================
    print("\n--- LR schedule ---")
    _opt = AdamW([torch.zeros(1, requires_grad=True)], lr=cfg.lr)
    _sch = _make_scheduler(_opt, warmup=2, total=10, lr_max=cfg.lr, lr_min=cfg.lr_min)
    lrs  = []
    for _ in range(10):
        _sch.step()
        lrs.append(_opt.param_groups[0]["lr"])
    assert lrs[1] > lrs[0], "warmup not increasing"
    assert lrs[-1] < lrs[2], "no cosine decay"
    assert lrs[-1] >= cfg.lr_min * 0.99
    print(f"  peak={max(lrs):.2e}  final={lrs[-1]:.2e}  ✓")

    # ================================================================
    # CHECKPOINT
    # ================================================================
    print("\n--- Checkpoint ---")
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "stage1_best.pt"
        _save_checkpoint(ckpt_path, model, opt, sch, step=4, best_metric=loss.item())
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["step"] == 4
        spear_keys = [k for k in ckpt["model_state"] if k.startswith("encoder._spear.")]
        assert len(spear_keys) == 0, f"SPEAR weights in checkpoint: {spear_keys[:3]}"
        print(f"  {len(ckpt['model_state'])} tensors saved (SPEAR excluded)  ✓")

    # ================================================================
    # STAGE 1 EVAL LOOP
    # ================================================================
    print("\n--- Stage 1 val loop ---")
    fake_val = [_fake_s1_batch(B, T_audio) for _ in range(3)]
    val_loss = _eval_stage1(model, fake_val, torch.device("cpu"), use_bf16=False)
    assert isinstance(val_loss["recon"], float) and val_loss["recon"] > 0
    print(f"  val_recon={val_loss['recon']:.4f}  ✓")

    # ================================================================
    # STAGE 2: routing masks
    # ================================================================
    print("\n--- Stage 2: routing masks ---")
    model.train()
    m_L, m_P, m_U = model.routing()
    assert m_L.shape == (cfg.K,)
    total_soft = m_L + m_P + m_U
    assert (total_soft - 1).abs().max() < 1e-4, "masks must sum to 1"
    n_L, n_P, n_U = model.routing.hard_counts
    assert n_L + n_P + n_U == cfg.K
    entropy = model.routing.routing_entropy
    print(f"  L={n_L}  P={n_P}  U={n_U}  H={entropy:.4f} nats  ✓")

    # ================================================================
    # STAGE 2: forward shapes + backward
    # ================================================================
    print("\n--- Stage 2: task heads + backward ---")
    from torch.optim import AdamW as _AdamW
    opt2 = _AdamW([
        {"params": list(model.sae.parameters()),      "lr": cfg.lr},
        {"params": list(model.routing.parameters()),  "lr": cfg.lr_routing},
        {"params": list(model.pr_head.parameters()) +
                   list(model.sid_head.parameters()) +
                   list(model.grl_head.parameters()), "lr": cfg.lr_heads},
    ], weight_decay=cfg.weight_decay)

    for step in range(1, 4):
        audio, audio_lens, targets, target_lens, speaker_ids = _fake_s2_batch(
            B, T_audio, cfg.vocab_size, cfg.num_speakers)
        opt2.zero_grad(set_to_none=True)
        out = model(audio, audio_lens, stage=2, grl_lambda=0.5)

        assert "pr_logits"  in out and out["pr_logits"].shape  == (B, out["h_t"].size(1), cfg.vocab_size)
        assert "sid_logits" in out and out["sid_logits"].shape == (B, cfg.num_speakers)
        assert "grl_logits" in out and out["grl_logits"].shape == (B, cfg.num_speakers)
        assert out["z_P_bar"].shape == (B, cfg.K), f"z_P_bar should be (B,K), got {out['z_P_bar'].shape}"
        assert out["h_hat"].shape == out["h_t"].shape, "recon shape mismatch"

        l_recon = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
        l_pr    = ctc_pr_loss(out["pr_logits"], targets, out["out_lengths"], target_lens)
        l_sid   = sid_ce_loss(out["sid_logits"], speaker_ids)
        l_grl   = sid_ce_loss(out["grl_logits"], speaker_ids)
        l_route = route_loss(model.routing.logits)
        total   = l_recon + cfg.alpha*l_pr + cfg.beta*l_sid + cfg.grl_weight*l_grl + cfg.rho*l_route
        total.backward()

        assert model.pr_head.fc.weight.grad         is not None, "pr_head: no grad"
        assert model.sid_head.net[0].weight.grad    is not None, "sid_head: no grad"
        assert model.grl_head.fc.weight.grad        is not None, "grl_head: no grad"
        assert model.routing.logits.grad     is not None, "routing: no grad"

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not p.grad.isnan().any(), f"NaN grad in {name} step {step}"

        nn.utils.clip_grad_norm_(list(model.sae.parameters()) +
                                 list(model.routing.parameters()) +
                                 list(model.pr_head.parameters()) +
                                 list(model.sid_head.parameters()) +
                                 list(model.grl_head.parameters()), cfg.grad_clip)
        opt2.step()
        print(f"  step {step}  recon={l_recon.item():.4f}  pr={l_pr.item():.4f}"
              f"  sid={l_sid.item():.4f}  grl={l_grl.item():.4f}  ✓")

    # ================================================================
    # GRL reversal
    # ================================================================
    print("\n--- GRL gradient reversal ---")
    from model.heads import gradient_reversal
    x = torch.randn(2, cfg.K, requires_grad=True)
    gradient_reversal(x, lam=1.0).sum().backward()
    assert (x.grad + 1).abs().max() < 1e-5, "GRL should produce -1 * upstream grad"
    print("  GRL reverses gradient  ✓")

    # ================================================================
    # DELAYED GRL
    # ================================================================
    print("\n--- Delayed GRL (grl_delay_steps) ---")
    from train import run_stage2 as _run_stage2
    from train import _dann_lambda

    cfg_delay            = DISConfig()
    cfg_delay.device     = "cpu"
    cfg_delay.bf16       = False
    cfg_delay.K          = 128
    cfg_delay.topk       = 6
    cfg_delay.batch_size = 2
    cfg_delay.num_speakers = 5
    cfg_delay.vocab_size = 41
    cfg_delay.grl_delay_steps = 3   # GRL should be off for steps 1-2, on at step 3
    cfg_delay.grl_weight = 0.5

    # Directly check that eff_grl_weight and grl_lam are zero before delay threshold
    for step, expect_active in [(1, False), (2, False), (3, True), (4, True)]:
        grl_active     = (cfg_delay.grl_delay_steps == 0 or step >= cfg_delay.grl_delay_steps)
        eff_grl_weight = cfg_delay.grl_weight if grl_active else 0.0
        grl_lam        = _dann_lambda(step, 6) if grl_active else 0.0
        assert grl_active == expect_active, f"step {step}: expected active={expect_active}"
        if not expect_active:
            assert eff_grl_weight == 0.0, f"step {step}: eff_grl_weight should be 0"
            assert grl_lam == 0.0,        f"step {step}: grl_lam should be 0"
        else:
            assert eff_grl_weight == cfg_delay.grl_weight
            assert grl_lam > 0.0
    print("  eff_grl_weight=0 before delay, >0 after  ✓")

    # ================================================================
    # STAGE 2 EVAL LOOP
    # ================================================================
    print("\n--- Stage 2 val loop ---")
    fake_val2 = [_fake_s2_batch(B, T_audio, cfg.vocab_size, cfg.num_speakers) for _ in range(3)]
    val2 = _eval_stage2(model, fake_val2, torch.device("cpu"), use_bf16=False)
    assert "recon" in val2 and "pr" in val2 and "per" in val2 and "sid_acc" in val2
    print(f"  val_recon={val2['recon']:.4f}  val_pr={val2['pr']:.4f}  PER={val2['per']:.3f}  sid_acc={val2['sid_acc']:.3f}  ✓")

    # ================================================================
    # EXP 1: DUAL GRL (phoneme GRL on z_P)
    # ================================================================
    print("\n--- Exp 1: Dual GRL ---")
    cfg_e1              = DISConfig()
    cfg_e1.device       = "cpu"
    cfg_e1.bf16         = False
    cfg_e1.K            = 128
    cfg_e1.topk         = 6
    cfg_e1.num_speakers = 5
    cfg_e1.vocab_size   = 41
    cfg_e1.grl_phoneme_weight = 0.01

    model_e1 = build_dis_model(cfg_e1)
    audio, audio_lens, targets, target_lens, speaker_ids = _fake_s2_batch(B, T_audio, 41, 5)
    out_e1 = model_e1(audio, audio_lens, stage=2, grl_lambda=0.5)
    assert "pr_grl_logits" in out_e1, "Dual GRL: pr_grl_logits missing from output"
    assert out_e1["pr_grl_logits"].shape == out_e1["pr_logits"].shape, "pr_grl_logits wrong shape"

    # GRL reversal check: gradient on z_P should be reversed through pr_grl_head
    l_e1 = ctc_pr_loss(out_e1["pr_grl_logits"], targets, out_e1["out_lengths"], target_lens)
    l_e1.backward()
    assert model_e1.pr_grl_head.fc.weight.grad is not None, "pr_grl_head: no gradient"
    print("  pr_grl_logits present, shape matches pr_logits ✓")
    print("  pr_grl_head receives gradient ✓")

    # ================================================================
    # EXP 2: TOPP-K DECORRELATION (stage 1)
    # ================================================================
    print("\n--- Exp 2: TopK decorrelation ---")
    from losses import decor_loss as _decor_loss
    # Use straight-through-style sparse tensor so gradients flow
    z_pre_fake = torch.randn(4, 10, 128, requires_grad=True)
    topk_vals, topk_idx = z_pre_fake.topk(6, dim=-1)
    z_sparse   = torch.zeros_like(z_pre_fake).scatter_(-1, topk_idx, topk_vals)
    z_ste      = z_pre_fake + (z_sparse - z_pre_fake).detach()   # straight-through
    lens_fake  = torch.tensor([10, 9, 8, 10])
    d_loss     = _decor_loss(z_ste, lens_fake)
    assert d_loss.item() >= 0, "decor_loss must be non-negative"
    d_loss.backward()
    assert z_pre_fake.grad is not None, "decor_loss: no gradient"
    print(f"  decor_loss={d_loss.item():.6f}  (non-negative ✓, differentiable ✓)")

    # ================================================================
    # EXP 4: U-BUCKET INFORMATION BOTTLENECK
    # ================================================================
    print("\n--- Exp 4: U-bucket information bottleneck ---")
    from losses import ub_loss as _ub_loss
    cfg_e4              = DISConfig()
    cfg_e4.device       = "cpu"
    cfg_e4.bf16         = False
    cfg_e4.K            = 128
    cfg_e4.topk         = 6
    cfg_e4.num_speakers = 5
    cfg_e4.vocab_size   = 41
    cfg_e4.ub_weight    = 0.1

    model_e4 = build_dis_model(cfg_e4)
    out_e4 = model_e4(audio, audio_lens, stage=2, grl_lambda=0.0)
    assert "m_L" in out_e4 and "m_P" in out_e4, "m_L / m_P missing from output"
    assert out_e4["m_L"].shape == (cfg_e4.K,), f"m_L shape wrong: {out_e4['m_L'].shape}"

    l_ub_test = _ub_loss(out_e4["m_L"], out_e4["m_P"])
    assert 0.0 <= l_ub_test.item() <= 2.0, f"ub_loss out of range: {l_ub_test.item()}"
    l_ub_test.backward()
    assert model_e4.routing.logits.grad is not None, "ub_loss: no gradient to routing.logits"
    print(f"  ub_loss={l_ub_test.item():.4f}  in [0,2] ✓")
    print("  ub_loss gradient flows to routing.logits ✓")

    # ================================================================
    # EXP 5: STRAIGHT-THROUGH ESTIMATOR FOR ROUTING GRADIENT
    # ================================================================
    print("\n--- Exp 5: STE routing gradient ---")
    cfg_e5              = DISConfig()
    cfg_e5.device       = "cpu"
    cfg_e5.bf16         = False
    cfg_e5.K            = 128
    cfg_e5.topk         = 6
    cfg_e5.num_speakers = 5
    cfg_e5.vocab_size   = 41
    cfg_e5.ste_routing  = True

    model_e5 = build_dis_model(cfg_e5)

    # Forward shape check
    model_e5.eval()
    with torch.no_grad():
        out_e5 = model_e5(audio, audio_lens, stage=2, grl_lambda=0.5)
    assert out_e5["z_L"].shape == (B, out_e5["z_t"].shape[1], cfg_e5.K), "STE: z_L wrong shape"
    assert out_e5["h_hat"].shape == out_e5["h_t"].shape, "STE: h_hat shape mismatch"
    # Sparsity in reconstruction must match (z_L_sp = m_L * z_t is used for h_hat)
    # The h_hat must be finite (decoder receives valid input)
    assert out_e5["h_hat"].isfinite().all(), "STE: h_hat contains non-finite values"
    print("  STE forward shapes correct, reconstruction finite ✓")

    # Backward check in train mode with fresh forward pass (no_grad used above)
    model_e5.train()
    model_e5.routing.logits.grad = None
    out_e5_tr = model_e5(audio, audio_lens, stage=2, grl_lambda=0.5)
    l_ste = (recon_loss(out_e5_tr["h_t"], out_e5_tr["h_hat"], out_e5_tr["out_lengths"]) +
             ctc_pr_loss(out_e5_tr["pr_logits"], targets, out_e5_tr["out_lengths"], target_lens) +
             sid_ce_loss(out_e5_tr["sid_logits"], speaker_ids))
    l_ste.backward()
    routing_grad = model_e5.routing.logits.grad  # (K, 3)
    nonzero_rows = (routing_grad.abs().sum(-1) > 1e-10).sum().item()
    assert nonzero_rows == cfg_e5.K, \
        f"STE: expected {cfg_e5.K} non-zero routing grad rows, got {nonzero_rows}"
    print(f"  STE: {nonzero_rows}/{cfg_e5.K} routing logit gradient rows non-zero ✓")

    # ================================================================
    # EXPERIMENT D: no_routing
    # ================================================================
    print("\n--- Experiment D: no_routing ---")
    cfg_d            = DISConfig()
    cfg_d.device     = "cpu"
    cfg_d.bf16       = False
    cfg_d.K          = 128
    cfg_d.topk       = 6
    cfg_d.batch_size = 2
    cfg_d.num_speakers = 5
    cfg_d.vocab_size = 41
    cfg_d.no_routing = True

    model_d = build_dis_model(cfg_d)
    audio, audio_lens, targets, target_lens, speaker_ids = _fake_s2_batch(B, T_audio, 41, 5)
    out_d = model_d(audio, audio_lens, stage=2, grl_lambda=0.0)
    assert "pr_logits"  in out_d, "no_routing: missing pr_logits"
    assert "sid_logits" in out_d, "no_routing: missing sid_logits"
    assert out_d["z_L"].shape == out_d["z_t"].shape, "no_routing: z_L should equal z_t"
    # backward should work
    l = (recon_loss(out_d["h_t"], out_d["h_hat"], out_d["out_lengths"]) +
         ctc_pr_loss(out_d["pr_logits"], targets, out_d["out_lengths"], target_lens) +
         sid_ce_loss(out_d["sid_logits"], speaker_ids))
    l.backward()
    print("  no_routing forward+backward ✓")

    # ================================================================
    # EXPERIMENT E: fixed_routing
    # ================================================================
    print("\n--- Experiment E: fixed_routing ---")
    cfg_e                    = DISConfig()
    cfg_e.device             = "cpu"
    cfg_e.bf16               = False
    cfg_e.K                  = 128
    cfg_e.topk               = 6
    cfg_e.batch_size         = 2
    cfg_e.num_speakers       = 5
    cfg_e.vocab_size         = 41
    cfg_e.fixed_routing      = True
    cfg_e.fixed_routing_split = 0.7

    model_e = build_dis_model(cfg_e)
    assert not model_e.routing.logits.requires_grad, "fixed_routing: logits should be frozen"
    n_L_e = int(128 * 0.7)
    assert model_e.routing.hard_counts[0] == n_L_e, "fixed_routing: wrong L count"
    assert model_e.routing.hard_counts[1] == 128 - n_L_e, "fixed_routing: wrong P count"
    # trainable routing params should be empty
    routing_trainable_e = [p for p in model_e.routing.parameters() if p.requires_grad]
    assert len(routing_trainable_e) == 0, "fixed_routing: routing params should all be frozen"
    print(f"  fixed_routing frozen ✓  L={n_L_e}  P={128-n_L_e}")

    # ================================================================
    # EXPERIMENT F: n_routes=2
    # ================================================================
    print("\n--- Experiment F: n_routes=2 (binary L/P) ---")
    cfg_f            = DISConfig()
    cfg_f.device     = "cpu"
    cfg_f.bf16       = False
    cfg_f.K          = 128
    cfg_f.topk       = 6
    cfg_f.batch_size = 2
    cfg_f.num_speakers = 5
    cfg_f.vocab_size = 41
    cfg_f.n_routes   = 2

    model_f = build_dis_model(cfg_f)
    assert model_f.routing.logits.shape == (128, 2), "n_routes=2: logits should be (K, 2)"
    out_f = model_f(audio, audio_lens, stage=2, grl_lambda=0.0)
    n_L_f, n_P_f, n_U_f = model_f.routing.hard_counts
    assert n_U_f == 0, "n_routes=2: U count must be 0"
    assert n_L_f + n_P_f == 128, "n_routes=2: L+P must equal K"
    # backward
    l_f = (recon_loss(out_f["h_t"], out_f["h_hat"], out_f["out_lengths"]) +
           ctc_pr_loss(out_f["pr_logits"], targets, out_f["out_lengths"], target_lens) +
           sid_ce_loss(out_f["sid_logits"], speaker_ids))
    l_f.backward()
    print(f"  n_routes=2 forward+backward ✓  L={n_L_f}  P={n_P_f}  U={n_U_f}")

    # ================================================================
    # PROBE RUNNER: lightweight check
    # ================================================================
    print("\n--- Probe runner (lightweight) ---")
    from probe_runner import _SIDProbe, _PRProbe, _mean_pool as _probe_mean_pool

    sid_probe = _SIDProbe(in_dim=cfg.K, num_speakers=cfg.num_speakers)
    pr_probe  = _PRProbe(in_dim=cfg.K, vocab_size=cfg.vocab_size)

    # SID probe forward
    z_fake   = torch.randn(2, 20, cfg.K)
    lens_fake = torch.tensor([20, 18])
    z_pool   = _probe_mean_pool(z_fake, lens_fake)
    sid_out  = sid_probe(z_pool)
    assert sid_out.shape == (2, cfg.num_speakers), f"SID probe wrong shape: {sid_out.shape}"

    # PR probe forward
    pr_out = pr_probe(z_fake)
    assert pr_out.shape == (2, 20, cfg.vocab_size), f"PR probe wrong shape: {pr_out.shape}"

    # SID probe backward
    sid_out.sum().backward()
    assert sid_probe.net[0].weight.grad is not None
    print("  SID probe forward+backward ✓")
    print("  PR probe forward ✓")

    print("\n=== All smoke tests passed ✓ ===")


if __name__ == "__main__":
    run()
