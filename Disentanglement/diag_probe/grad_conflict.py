#!/usr/bin/env python3
"""Test the 'reconstruction defends speaker in z_L' hypothesis directly.

For each batch we compute, on z_L (and on the SAE encoder weight):
  u_recon  = -dL_recon/dz_L   (the direction reconstruction wants z_L to move)
  u_remove = +dL_disc/dz_L    (the direction the speaker adversary wants z_L to move,
                               i.e. ASCEND the discriminator loss -> make speaker unreadable)
and report cos(u_recon, u_remove).

  cos < 0  -> the two updates CONFLICT  -> reconstruction DEFENDS speaker in z_L  (hypothesis TRUE)
  cos ~ 0  -> independent               -> reconstruction does not defend speaker
  cos > 0  -> aligned                   -> removing speaker would HELP reconstruction

We recover the raw +dL_disc/dz_L by calling the GRL head with lam=-1 (its reversal
backward multiplies by -lam = +1, undoing the reversal).
"""
import argparse
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

DIS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DIS_DIR.parent
for p in (str(DIS_DIR), str(REPO_ROOT / "Probing" / "pr")):
    while p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(DIS_DIR))

from config import DISConfig
from model import build_dis_model
from train import _load_stage1_checkpoint
from data.dataset import make_stage2_dataloaders
from losses import recon_loss, ctc_pr_loss, sid_ce_loss, sid_ce_loss_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fixed_blocks", action="store_true")
    p.add_argument("--per_block_topk", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--K_L", type=int, default=3072)
    p.add_argument("--K_P", type=int, default=1024)
    p.add_argument("--K_U", type=int, default=1024)
    p.add_argument("--topk_L", type=int, default=160)
    p.add_argument("--topk_P", type=int, default=64)
    p.add_argument("--topk_U", type=int, default=32)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--spear_layernorm", action="store_true")
    p.add_argument("--grl_attention_pool", action="store_true")
    p.add_argument("--local_data", action="store_true")
    p.add_argument("--train_split_dir", default="train-clean-100")
    p.add_argument("--n_batches", type=int, default=20)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DISConfig()
    cfg.device = str(dev)
    cfg.fixed_blocks = args.fixed_blocks
    cfg.per_block_topk = args.per_block_topk
    cfg.K_L, cfg.K_P, cfg.K_U = args.K_L, args.K_P, args.K_U
    cfg.topk_L, cfg.topk_P, cfg.topk_U = args.topk_L, args.topk_P, args.topk_U
    cfg.topk = args.topk
    cfg.spear_layernorm = args.spear_layernorm
    cfg.grl_attention_pool = args.grl_attention_pool
    cfg.local_data = args.local_data
    cfg.train_split_dir = args.train_split_dir

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model_state"]
    cfg.num_speakers = state["sid_head.fc.weight"].shape[0]
    cfg.K = state["sae.enc_weight"].shape[0]
    del state

    _, train_dl, _, _ = make_stage2_dataloaders(cfg)

    model = build_dis_model(cfg)
    _load_stage1_checkpoint(Path(args.ckpt), model, cfg)
    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    msd = model.state_dict()
    filt = {k: v for k, v in ckpt["model_state"].items() if k in msd and msd[k].shape == v.shape}
    miss, _ = model.load_state_dict(filt, strict=False)
    non_spear = [k for k in miss if not k.startswith("encoder._spear.")]
    if non_spear:
        print(f"[grad_conflict] missing (non-SPEAR): {non_spear[:6]}")
    model.to(dev).eval()
    for n, prm in model.named_parameters():
        prm.requires_grad_(n.startswith("sae."))   # only SAE params; SPEAR frozen
    enc_w = model.sae.enc_weight

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cos_rec_z, cos_rec_w, cos_pr_z, cos_pr_w = [], [], [], []
    it = iter(train_dl)
    for i in range(args.n_batches):
        try:
            audios, alen, targets, tlen, spk = next(it)
        except StopIteration:
            break
        audios, alen, spk = audios.to(dev), alen.to(dev), spk.to(dev)
        targets, tlen = targets.to(dev), tlen.to(dev)
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16
               else torch.autocast("cuda", enabled=False))
        with ctx:
            out = model(audios, alen, stage=2, grl_lambda=0.0)
            zL = out["z_L"]
            L_recon = recon_loss(out["h_t"], out["h_hat"], out["out_lengths"])
            L_pr    = ctc_pr_loss(out["pr_logits"], targets, out["out_lengths"], tlen)
            logits = model.grl_head(zL, out["out_lengths"], -1.0)   # lam=-1 -> raw +dL/dz
            L_disc = (sid_ce_loss_frames(logits, spk, out["out_lengths"])
                      if logits.dim() == 3 else sid_ce_loss(logits, spk))

        g_rec_z  = torch.autograd.grad(L_recon, zL,    retain_graph=True)[0].float()
        g_pr_z   = torch.autograd.grad(L_pr,    zL,    retain_graph=True)[0].float()
        g_disc_z = torch.autograd.grad(L_disc,  zL,    retain_graph=True)[0].float()
        g_rec_w  = torch.autograd.grad(L_recon, enc_w, retain_graph=True)[0].float()
        g_pr_w   = torch.autograd.grad(L_pr,    enc_w, retain_graph=True)[0].float()
        g_disc_w = torch.autograd.grad(L_disc,  enc_w, retain_graph=False)[0].float()

        # update directions: task wants -g_task, removal wants +g_disc
        crz = F.cosine_similarity((-g_rec_z).flatten()[None], g_disc_z.flatten()[None]).item()
        cpz = F.cosine_similarity((-g_pr_z).flatten()[None],  g_disc_z.flatten()[None]).item()
        crw = F.cosine_similarity((-g_rec_w).flatten()[None], g_disc_w.flatten()[None]).item()
        cpw = F.cosine_similarity((-g_pr_w).flatten()[None],  g_disc_w.flatten()[None]).item()
        cos_rec_z.append(crz); cos_pr_z.append(cpz); cos_rec_w.append(crw); cos_pr_w.append(cpw)
        print(f"batch {i:2d}: recon={L_recon.item():.3f} pr={L_pr.item():.3f} disc={L_disc.item():.3f} | "
              f"z_L: recon={crz:+.4f} PR={cpz:+.4f} | encW: recon={crw:+.4f} PR={cpw:+.4f}", flush=True)

    def rep(name, a):
        print(f"  {name:<34s}: {st.mean(a):+.4f}  (sd {st.pstdev(a):.4f}, n={len(a)})")
    print(f"\n===== MEAN cos(task-update, speaker-removal-update) =====")
    rep("RECON vs removal  (z_L)",  cos_rec_z)
    rep("PR    vs removal  (z_L)",  cos_pr_z)
    rep("RECON vs removal  (enc_W)", cos_rec_w)
    rep("PR    vs removal  (enc_W)", cos_pr_w)
    print("  cos < 0 => that task CONFLICTS with speaker-removal => it DEFENDS speaker in z_L")
    print("  cos ~ 0 => independent;  cos > 0 => removing speaker would help that task")


if __name__ == "__main__":
    main()
