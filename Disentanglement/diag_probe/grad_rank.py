#!/usr/bin/env python3
"""Why does grl_p remove phonemes but grl barely removes speaker?
Hypothesis: the pooled speaker adversary gives a LOW-RANK per-frame removal
signal (every frame shares the pooled bottleneck), while the per-frame phoneme
adversary gives a HIGH-RANK one.  We measure the effective rank of the per-frame
removal gradients for each, per utterance.

  effective rank = exp(entropy of normalized singular values)   (a soft rank)
  stable rank    = ||G||_F^2 / ||G||_2^2
"""
import argparse, statistics as st, sys
from pathlib import Path
import torch

DIS_DIR = Path(__file__).resolve().parent.parent
for p in (str(DIS_DIR), str(DIS_DIR.parent / "Probing" / "pr")):
    while p in sys.path: sys.path.remove(p)
sys.path.insert(0, str(DIS_DIR))
from config import DISConfig
from model import build_dis_model
from train import _load_stage1_checkpoint
from data.dataset import make_stage2_dataloaders
from losses import ctc_pr_loss, sid_ce_loss, sid_ce_loss_frames


def eff_ranks(G):  # G: (T, K) per-frame gradient matrix for one utterance
    s = torch.linalg.svdvals(G.float())
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0, 0.0
    p = s / s.sum()
    eff = float(torch.exp(-(p * torch.log(p)).sum()))     # entropy effective rank
    stable = float((s.pow(2).sum() / s[0].pow(2)))         # stable rank
    return eff, stable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--K_L", type=int, default=3072); ap.add_argument("--K_P", type=int, default=1024)
    ap.add_argument("--K_U", type=int, default=1024)
    ap.add_argument("--topk_L", type=int, default=160); ap.add_argument("--topk_P", type=int, default=64)
    ap.add_argument("--topk_U", type=int, default=32); ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--grl_attention_pool", action="store_true")
    ap.add_argument("--grl_dense_context", action="store_true")
    ap.add_argument("--grl_context_kernel", type=int, default=31)
    ap.add_argument("--n_batches", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DISConfig(); cfg.device = str(dev)
    cfg.fixed_blocks = True; cfg.per_block_topk = True
    cfg.K_L, cfg.K_P, cfg.K_U = a.K_L, a.K_P, a.K_U
    cfg.topk_L, cfg.topk_P, cfg.topk_U = a.topk_L, a.topk_P, a.topk_U
    cfg.topk = a.topk; cfg.spear_layernorm = True
    cfg.grl_attention_pool = a.grl_attention_pool
    cfg.grl_dense_context = a.grl_dense_context
    cfg.grl_context_kernel = a.grl_context_kernel
    head = "dense_context" if a.grl_dense_context else ("attention_pool" if a.grl_attention_pool else "mean_pool")
    print(f"[grad_rank] speaker head = {head}   ckpt = {a.ckpt}")
    cfg.local_data = True; cfg.train_split_dir = "train-clean-100"
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state"]
    cfg.num_speakers = state["sid_head.fc.weight"].shape[0]; cfg.K = state["sae.enc_weight"].shape[0]; del state
    _, dl, _, _ = make_stage2_dataloaders(cfg)
    model = build_dis_model(cfg); _load_stage1_checkpoint(Path(a.ckpt), model, cfg)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False); msd = model.state_dict()
    model.load_state_dict({k: v for k, v in ck["model_state"].items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model.to(dev).eval()
    for n, prm in model.named_parameters(): prm.requires_grad_(n.startswith("sae."))
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    spk_eff, spk_st, ph_eff, ph_st = [], [], [], []
    spk_rows, ph_rows = [], []          # pooled across utterances -> global subspace rank
    it = iter(dl)
    for _ in range(a.n_batches):
        try: au, al, tg, tl, spk = next(it)
        except StopIteration: break
        au, al, spk, tg, tl = au.to(dev), al.to(dev), spk.to(dev), tg.to(dev), tl.to(dev)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
        with ctx:
            out = model(au, al, stage=2, grl_lambda=0.0)
            zL, zP = out["z_L"], out["z_P"]
            sp_logits = model.grl_head(zL, out["out_lengths"], -1.0)
            L_sp = sid_ce_loss_frames(sp_logits, spk, out["out_lengths"]) if sp_logits.dim() == 3 else sid_ce_loss(sp_logits, spk)
            ph_logits = model.pr_grl_head(zP, -1.0)
            L_ph = ctc_pr_loss(ph_logits, tg, out["out_lengths"], tl)
        gL = torch.autograd.grad(L_sp, zL, retain_graph=True)[0].float()   # (B,T,K_L) speaker removal
        gP = torch.autograd.grad(L_ph, zP, retain_graph=False)[0].float()  # (B,T,K_P) phoneme removal
        lens = out["out_lengths"]
        for b in range(zL.shape[0]):
            T = int(lens[b]);
            if T < 4: continue
            e, s = eff_ranks(gL[b, :T]); spk_eff.append(e); spk_st.append(s)
            e, s = eff_ranks(gP[b, :T]); ph_eff.append(e); ph_st.append(s)
            # keep a few mean-removed rows per utterance for the global subspace estimate
            spk_rows.append(gL[b, :T].cpu()); ph_rows.append(gP[b, :T].cpu())

    def global_eff(rows, cap=4000):
        G = torch.cat(rows, 0)
        if G.shape[0] > cap:
            G = G[torch.randperm(G.shape[0])[:cap]]
        return eff_ranks(G)            # (eff_rank, stable_rank) of the whole stacked matrix

    g_spk_e, g_spk_s = global_eff(spk_rows)
    g_ph_e,  g_ph_s  = global_eff(ph_rows)
    print(f"\n===== PER-UTTERANCE effective rank of removal gradients (T x K) =====")
    print(f"  SPEAKER (z_L):  eff_rank={st.mean(spk_eff):6.2f}   stable_rank={st.mean(spk_st):6.2f}   (K_L={a.K_L})")
    print(f"  PHONEME (z_P):  eff_rank={st.mean(ph_eff):6.2f}   stable_rank={st.mean(ph_st):6.2f}   (K_P={a.K_P})")
    print(f"  ratio phoneme/speaker eff_rank = {st.mean(ph_eff)/max(st.mean(spk_eff),1e-9):.1f}x")
    print(f"\n===== GLOBAL subspace rank (all frames stacked, N x K) =====")
    print(f"  SPEAKER (z_L):  eff_rank={g_spk_e:6.2f}   stable_rank={g_spk_s:6.2f}")
    print(f"  PHONEME (z_P):  eff_rank={g_ph_e:6.2f}   stable_rank={g_ph_s:6.2f}")
    print("  => low speaker rank vs high phoneme rank means the speaker adversary erases too few")
    print("     directions: speaker survives in the orthogonal complement that no gradient touches.")


if __name__ == "__main__":
    main()
