#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=MLMI-bbg25-SL2-GPU
#SBATCH --job-name=var_ht
#SBATCH --output=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/var_ht_%j.out
#SBATCH --error=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement/logs/diag/var_ht_%j.err

# Measures per-dim Var(h_t) with and without SPEAR LayerNorm, to confirm the
# recon-loss jump (0.007 -> 0.08) is target-scale, not worse reconstruction.

set -euo pipefail
. /etc/profile.d/modules.sh
module purge 2>/dev/null
module load rhel8/default-amp 2>/dev/null

PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
DIS_DIR=/rds/user/bbg25/hpc-work/Thesis/Final-Proj/Disentanglement
export PYTHONUNBUFFERED=1
export HF_HOME="${DIS_DIR}/../Probing/data/hf_home"
export HF_DATASETS_CACHE="${DIS_DIR}/../Probing/data/datasets_cache"
export HF_HUB_CACHE="${DIS_DIR}/../Probing/data/hub_cache"
mkdir -p "${DIS_DIR}/logs/diag"
cd "${DIS_DIR}"

${PYTHON} -u - <<'PY'
import sys; sys.path.insert(0, ".")
import torch
from config import DISConfig
from model.spear_encoder import SpearEncoder
from data.dataset import make_stage1_dataloaders

cfg = DISConfig()
cfg.max_train_examples = 256
cfg.max_val_examples = 8
cfg.batch_size = 16
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", cfg.device)

enc = SpearEncoder(cfg).to(cfg.device).eval()
train_dl, _ = make_stage1_dataloaders(cfg)

def masked_var(h, out_lens):
    B, T, D = h.shape
    m = (torch.arange(T, device=h.device)[None, :] < out_lens[:, None]).float().unsqueeze(-1)
    n = m.sum()
    mean = (h * m).sum((0, 1)) / n
    var  = (((h - mean) ** 2) * m).sum((0, 1)) / n
    return var.mean().item(), int(n.item())

# average over several batches for stability
import itertools
res = {False: [], True: []}
for audios, lens in itertools.islice(train_dl, 8):
    audios, lens = audios.to(cfg.device), lens.to(cfg.device)
    for ln in (False, True):
        enc._layernorm = ln
        with torch.no_grad():
            h, ol = enc(audios, lens)
        v, n = masked_var(h, ol)
        res[ln].append(v)

import statistics as st
v0 = st.mean(res[False]); v1 = st.mean(res[True])
print(f"spear_layernorm=False  Var(h_t)={v0:.6f}  RMS={v0**0.5:.4f}")
print(f"spear_layernorm=True   Var(h_t)={v1:.6f}  RMS={v1**0.5:.4f}")
print(f"variance ratio (LN / raw) = {v1/v0:.2f}x")
print(f"-> recon-MSE should scale ~{v1/v0:.1f}x; pre-LN ~0.007 implies post-LN ~{0.007*v1/v0:.3f}")
PY
echo "Finished : $(date)"
