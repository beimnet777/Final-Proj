#!/usr/bin/env python3
"""Smoke test: full pipeline without GPU or SPEAR download.

Patches AutoModel.from_pretrained with a tiny fake SPEAR model.
Runs a few training steps on CPU with synthetic data and verifies:
  - forward shapes
  - reconstruction loss
  - backward / gradients on SAE params only
  - h_t is fully detached (no grad flows into SPEAR)
  - cosine LR schedule
  - checkpoint save / load
  - _evaluate (val loop)

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
    """Tiny stub.  Returns L random hidden states of shape (B, T, D)."""

    def __init__(self) -> None:
        super().__init__()
        self.config = MagicMock()

    def forward(self, audio: torch.Tensor, lengths: torch.Tensor):
        B, T_audio = audio.shape
        T = max(T_audio // 640, 1)
        D, L = 1280, 13
        return {"hidden_states": [torch.randn(B, T, D) for _ in range(L)]}


# ---------------------------------------------------------------- helpers

def _fake_batch(B: int, T_audio: int):
    audio       = torch.randn(B, T_audio)
    audio_lens  = torch.tensor([T_audio, T_audio - 640], dtype=torch.long)
    return audio, audio_lens


# ---------------------------------------------------------------- main

def run() -> None:
    with patch("transformers.AutoModel.from_pretrained", return_value=_FakeSpear()):
        _test()


def _test() -> None:
    from config import DISConfig
    from model import build_dis_model
    from losses import recon_loss
    from train import _make_scheduler, _count_params, _save_checkpoint, _evaluate

    print("=== SAE smoke test ===\n")

    # ---- tiny config (avoids large matrices on CPU)
    cfg          = DISConfig()
    cfg.device   = "cpu"
    cfg.bf16     = False
    cfg.K        = 128
    cfg.topk     = 6
    cfg.batch_size      = 2
    cfg.eval_batch_size = 2
    cfg.total_steps     = 6
    cfg.warmup_steps    = 2
    cfg.log_every       = 99999

    B, T_audio = 2, 12_800   # ~0.8 s → T_frames ≈ 20

    # ================================================================
    # MODEL BUILD
    # ================================================================
    model = build_dis_model(cfg)
    frozen, trainable = _count_params(model)
    print(f"frozen={frozen:,}  trainable={trainable:,}")
    assert trainable > 0, "no trainable params"

    # Encoder should have zero trainable params
    enc_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    assert enc_trainable == 0, f"encoder should be fully frozen, got {enc_trainable} trainable"
    print("encoder fully frozen ✓")

    # ================================================================
    # FORWARD — shapes
    # ================================================================
    print("\n--- Forward shapes ---")
    model.eval()
    audio, audio_lens = _fake_batch(B, T_audio)
    with torch.no_grad():
        out = model(audio, audio_lens)

    B_, T_, D_ = out["h_t"].shape
    assert D_ == cfg.D,                           f"h_t D={D_} expected {cfg.D}"
    assert out["z_t"].shape   == (B_, T_, cfg.K), f"z_t shape {out['z_t'].shape}"
    assert out["h_hat"].shape == (B_, T_, cfg.D), f"h_hat shape {out['h_hat'].shape}"
    assert out["z_pre"].shape == (B_, T_, cfg.K), f"z_pre shape {out['z_pre'].shape}"
    assert out["out_lengths"].shape == (B_,),     f"out_lengths shape {out['out_lengths'].shape}"
    print(f"  h_t  : {tuple(out['h_t'].shape)}  ✓")
    print(f"  z_t  : {tuple(out['z_t'].shape)}  ✓")
    print(f"  h_hat: {tuple(out['h_hat'].shape)}  ✓")

    # ================================================================
    # FIXED TARGET — h_t must not carry gradients
    # ================================================================
    print("\n--- h_t is detached ---")
    model.train()
    audio, audio_lens = _fake_batch(B, T_audio)
    out = model(audio, audio_lens)
    assert not out["h_t"].requires_grad, "h_t should be detached (fixed target)"
    print("  h_t.requires_grad=False  ✓")

    # ================================================================
    # SPARSITY — exactly topk non-zeros per frame
    # ================================================================
    print("\n--- TopK sparsity ---")
    z = out["z_t"]
    nonzero_per_frame = (z != 0).sum(dim=-1)   # (B, T)
    assert (nonzero_per_frame == cfg.topk).all(), \
        f"expected {cfg.topk} non-zeros per frame, got {nonzero_per_frame.unique()}"
    print(f"  exactly topk={cfg.topk} active per frame  ✓")

    # ================================================================
    # LOSS + BACKWARD
    # ================================================================
    print("\n--- Loss + backward ---")
    from torch.optim import AdamW
    optimizer = AdamW(model.sae.parameters(), lr=cfg.lr)
    scheduler = _make_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps, cfg.lr, cfg.lr_min)

    for step in range(1, 5):
        audio, audio_lens = _fake_batch(B, T_audio)
        optimizer.zero_grad(set_to_none=True)
        out  = model(audio, audio_lens)
        loss = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
        loss.backward()

        # SAE params must have gradients
        assert model.sae.enc_weight.grad is not None, "enc_weight: no grad"
        assert model.sae.dec_weight.grad is not None, "dec_weight: no grad"
        assert model.sae.b_pre.grad      is not None, "b_pre: no grad"

        # No NaNs
        for name, p in model.sae.named_parameters():
            if p.grad is not None:
                assert not p.grad.isnan().any(), f"NaN grad in {name} at step {step}"

        nn.utils.clip_grad_norm_(model.sae.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        print(f"  step {step}  recon={loss.item():.4f}  ✓")

    # ================================================================
    # LR SCHEDULE — warmup then cosine decay
    # ================================================================
    print("\n--- LR schedule ---")
    from torch.optim import AdamW as _AdamW
    _opt = _AdamW([torch.zeros(1, requires_grad=True)], lr=cfg.lr)
    _sch = _make_scheduler(_opt, warmup=2, total=10, lr_max=cfg.lr, lr_min=cfg.lr_min)

    lrs = []
    for _ in range(10):
        _sch.step()
        lrs.append(_opt.param_groups[0]["lr"])

    assert lrs[1] > lrs[0], "LR should increase during warmup"
    assert lrs[-1] < lrs[2], "LR should decrease after warmup (cosine decay)"
    assert lrs[-1] >= cfg.lr_min * 0.99, f"LR below lr_min: {lrs[-1]}"
    print(f"  warmup peak={max(lrs):.2e}  final={lrs[-1]:.2e}  ✓")

    # ================================================================
    # CHECKPOINT save / load
    # ================================================================
    print("\n--- Checkpoint ---")
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "best.pt"
        _save_checkpoint(ckpt_path, model, optimizer, scheduler, step=4, best_metric=loss.item())
        assert ckpt_path.exists(), "checkpoint not created"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt["step"] == 4,                            "step mismatch"
        assert abs(ckpt["best_metric"] - loss.item()) < 1e-6, "metric mismatch"
        # SPEAR weights must NOT be in the checkpoint
        spear_keys = [k for k in ckpt["model_state"] if k.startswith("encoder._spear.")]
        assert len(spear_keys) == 0, f"SPEAR weights leaked into checkpoint: {spear_keys[:3]}"
        print(f"  saved {len(ckpt['model_state'])} weight tensors (SPEAR excluded)  ✓")

    # ================================================================
    # EVALUATE (val loop)
    # ================================================================
    print("\n--- Validation loop ---")
    from torch.utils.data import DataLoader, TensorDataset

    # Tiny fake val DataLoader that yields (audio, audio_lengths) tuples
    class _FakeValDataset:
        def __init__(self):
            self.data = [_fake_batch(2, T_audio) for _ in range(3)]
        def __iter__(self):
            return iter(self.data)

    val_loss = _evaluate(model, _FakeValDataset(), torch.device("cpu"), use_bf16=False)
    assert isinstance(val_loss, float) and val_loss > 0, f"val_loss={val_loss}"
    print(f"  val_recon={val_loss:.4f}  ✓")

    print("\n=== All smoke tests passed ✓ ===")


if __name__ == "__main__":
    run()
