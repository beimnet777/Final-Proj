#!/usr/bin/env python3
"""Smoke test for old+Gao single-stage architecture."""

from __future__ import annotations

import sys, tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))


class _FakeSpear(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = 0
        self.config.num_hidden_layers = 0

    def forward(self, audio, lengths):
        B, T_audio = audio.shape
        T = max(int(T_audio) // 640, 1)
        D, L = 1280, 13
        return {"hidden_states": [torch.randn(B, T, D) for _ in range(L)]}


def _fake_batch(B, T_audio, vocab, num_spk, P=8):
    audio       = torch.randn(B, T_audio)
    audio_lens  = torch.tensor([T_audio, T_audio - 640], dtype=torch.long)
    targets     = torch.randint(1, vocab - 1, (B, P))
    target_lens = torch.tensor([P, P - 2], dtype=torch.long)
    speaker_ids = torch.randint(0, num_spk, (B,))
    return audio, audio_lens, targets, target_lens, speaker_ids


def run():
    with patch("transformers.AutoModel.from_pretrained", return_value=_FakeSpear()):
        _test()


def _test():
    from config import DISConfig
    from model import build_dis_model
    from losses import recon_loss, ctc_pr_loss, sid_ce_loss, decorr_loss, route_loss
    from train import (_build_optimizer, _make_scheduler, _trainable_params,
                       _save_checkpoint, _load_checkpoint)

    print("=== Old+Gao smoke test ===\n")

    # Verify data paths resolve before wasting GPU time
    from config import DISConfig as _C
    _cfg_check = _C()
    assert _cfg_check.lexicon_path.exists(), \
        f"lexicon not found: {_cfg_check.lexicon_path}"
    assert _cfg_check.librispeech_cache_dir.exists(), \
        f"cache dir not found: {_cfg_check.librispeech_cache_dir}"
    print(f"Data paths OK ✓")

    cfg              = DISConfig()
    cfg.device       = "cpu"
    cfg.bf16         = False
    cfg.num_speakers = 5
    cfg.vocab_size   = 41
    cfg.K            = 128
    cfg.topk         = 6
    cfg.decorr_max_frames = 50
    cfg.total_steps  = 10

    B, T_audio = 2, 12_800

    model = build_dis_model(cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built   trainable params = {n:,}")

    opt   = _build_optimizer(cfg, model)
    sched = _make_scheduler(opt, warmup=2, total=5)
    model.train()

    print("\n--- Forward / backward ---")
    for step in range(1, 4):
        audio, audio_lens, targets, target_lens, speaker_ids = _fake_batch(
            B, T_audio, cfg.vocab_size, cfg.num_speakers)
        opt.zero_grad(set_to_none=True)

        out = model(audio, audio_lens, grl_lambda=0.5)

        # shape checks — all outputs always present
        B_, T_, D_ = out["h_t"].shape
        assert D_ == cfg.D,                                    "h_t D wrong"
        assert out["z_t"].shape    == (B_, T_, cfg.K),         "z_t shape wrong"
        assert out["h_hat"].shape  == (B_, T_, cfg.D),         "h_hat shape wrong"
        assert out["z_P_bar"].shape == (B_, 2 * cfg.K),        "z_P_bar shape wrong"
        assert out["pr_logits"].shape  == (B_, T_, cfg.vocab_size), "pr_logits shape wrong"
        assert out["sid_logits"].shape == (B_, cfg.num_speakers),   "sid_logits shape wrong"
        assert out["grl_logits"].shape == (B_, cfg.num_speakers),   "grl_logits shape wrong"

        l_r   = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
        l_d   = decorr_loss(out["z_t"], out["out_lengths"], max_frames=50)
        l_ro  = route_loss(model.routing.logits)
        l_pr  = ctc_pr_loss(out["pr_logits"], targets, out["out_lengths"], target_lens)
        l_sid = sid_ce_loss(out["sid_logits"], speaker_ids)
        l_grl = sid_ce_loss(out["grl_logits"], speaker_ids)
        total = (l_r + cfg.delta * l_d + cfg.rho * l_ro
                 + cfg.alpha * l_pr + cfg.beta * l_sid + l_grl)
        total.backward()

        # gradient checks — all params get gradient
        assert model.sae.enc_weight.grad    is not None, "enc_weight: no grad"
        assert model.sae.dec_weight.grad    is not None, "dec_weight: no grad"
        assert model.routing.logits.grad    is not None, "routing logits: no grad"
        assert model.encoder.mix_logits.grad is not None,"mix_logits: no grad"
        assert model.pr_head.fc2.weight.grad is not None,"pr_head: no grad"
        assert model.sid_head.fc.weight.grad is not None,"sid_head: no grad"
        assert model.grl_head.fc.weight.grad is not None,"grl_head: no grad"

        all_p = _trainable_params(model)
        for p in all_p:
            if p.grad is not None:
                assert not p.grad.isnan().any(), f"NaN grad at step {step}"
        nn.utils.clip_grad_norm_(all_p, cfg.grad_clip)
        opt.step(); sched.step()
        print(f"  step {step}  recon={l_r:.4f}  pr={l_pr:.4f}  sid={l_sid:.4f}  ✓")

    # checkpoint
    print("\n--- Checkpoint ---")
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "best.pt"
        _save_checkpoint(ckpt_path, model, opt, sched, step=3,
                         best_metric=l_r.item(), cfg=cfg)
        assert ckpt_path.exists()
        ckpt = _load_checkpoint(ckpt_path, model, cfg)
        assert ckpt["step"] == 3
        print("  save/load ✓")

    # GRL reversal
    print("\n--- GRL gradient reversal ---")
    x = torch.randn(2, cfg.K, requires_grad=True)
    from model.heads import gradient_reversal
    y = gradient_reversal(x, lam=1.0)
    y.sum().backward()
    assert (x.grad + torch.ones_like(x.grad)).abs().max() < 1e-6
    print("  GRL reverses gradient ✓")

    # routing entropy metric
    print("\n--- Routing ---")
    n_L, n_P, n_U = model.routing.hard_counts
    H = model.routing.routing_entropy
    assert n_L + n_P + n_U == cfg.K
    print(f"  L={n_L}  P={n_P}  U={n_U}  entropy={H:.4f} ✓")

    print("\n=== All smoke tests passed ✓ ===")


if __name__ == "__main__":
    run()
