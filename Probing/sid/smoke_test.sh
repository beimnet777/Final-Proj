#!/usr/bin/env bash
# smoke_test.sh — fast end-to-end sanity check for the SID pipeline.
#
# 1. Unit-tests the data pipeline (split counts, label encoding, crop behaviour).
# 2. Runs sid_run.py for 1 epoch with 32 examples/split on CPU (wav2vec2-base).
# 3. Verifies the summary JSON is written and test_acc is sensible.
#
# Usage (from inside sid/ or anywhere in the repo):
#   bash Final-Proj/Probing/sid/smoke_test.sh
#
set -euo pipefail

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
SID_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOXCELEB1_ROOT="/rds/user/bbg25/hpc-work/data/VoxCeleb1"
TMP="$(mktemp -d /tmp/sid_smoke_XXXXXX)"

cleanup() { rm -rf "${TMP}"; }
trap cleanup EXIT

echo "=== SID smoke test ==="
echo "    SID_DIR        : ${SID_DIR}"
echo "    VOXCELEB1_ROOT : ${VOXCELEB1_ROOT}"
echo "    TMP            : ${TMP}"
echo ""

# ── 1. Prerequisite checks ───────────────────────────────────────────────────
META="${SID_DIR}/veri_test_class.txt"
if [[ ! -f "${META}" ]]; then
    echo "ERROR: veri_test_class.txt not found at ${META}"
    echo "It should already be bundled. Check the sid/ directory."
    exit 1
fi

if [[ ! -d "${VOXCELEB1_ROOT}/dev/wav" ]]; then
    echo "ERROR: VoxCeleb1 dev split not found under ${VOXCELEB1_ROOT}"
    echo "Run: python sid/download_data.py --out_dir ${VOXCELEB1_ROOT}"
    exit 1
fi

# ── 2. Unit tests (no encoder, no network) ───────────────────────────────────
echo "--- Unit tests ---"
cd "${SID_DIR}"

"${PYTHON}" - <<'PYEOF'
import sys
sys.path.insert(0, ".")
import random, torch

# Config defaults
from sid_config import SIDConfig
cfg = SIDConfig()
assert hasattr(cfg, "meta_data"),            "meta_data missing from SIDConfig"
assert hasattr(cfg, "train_max_duration_s"), "train_max_duration_s missing from SIDConfig"
assert not hasattr(cfg, "val_split"),        "val_split should be removed from SIDConfig"
assert not hasattr(cfg, "max_duration_s"),   "max_duration_s should be removed from SIDConfig"
assert cfg.train_max_duration_s == 8.0,      f"Expected 8.0s, got {cfg.train_max_duration_s}"
print("[PASS] SIDConfig fields correct")

# _parse_meta split counts
from sid_data import _parse_meta
import pathlib
vox_root = pathlib.Path("/rds/user/bbg25/hpc-work/data/VoxCeleb1")
train_r, val_r, test_r = _parse_meta(cfg.meta_data, vox_root)
assert len(train_r) == 138361, f"train: expected 138361, got {len(train_r)}"
assert len(val_r)   ==   6904, f"val:   expected 6904,   got {len(val_r)}"
assert len(test_r)  ==   8251, f"test:  expected 8251,   got {len(test_r)}"
print(f"[PASS] _parse_meta: train={len(train_r)}, val={len(val_r)}, test={len(test_r)}")

# Label encoding: id10001 → 0, id11251 → 1250
labels = set(r[1] for r in train_r)
assert 0    in labels, "label 0 missing"
assert 1250 in labels, "label 1250 missing"
assert max(labels) == 1250, f"max label {max(labels)}"
print(f"[PASS] Label encoding: {len(labels)} unique labels, range 0–{max(labels)}")

# max_examples cap
tr2, va2, te2 = _parse_meta(cfg.meta_data, vox_root, max_examples=10)
assert len(tr2) == 10 and len(va2) == 10 and len(te2) == 10
print("[PASS] max_examples cap works")

# All splits from dev/wav/ (not test/wav/)
for rec in test_r[:20]:
    p = str(rec[0])
    assert "/dev/wav/" in p, f"test record not in dev/wav: {p}"
print("[PASS] All test records come from dev/wav/")

# VoxCeleb1Dataset: random_crop=True crops; random_crop=False does not
from sid_data import VoxCeleb1Dataset
sample_rec = train_r[0]
ds_crop    = VoxCeleb1Dataset([sample_rec], 16000, max_samples=128000, random_crop=True)
ds_full    = VoxCeleb1Dataset([sample_rec], 16000, max_samples=0,      random_crop=False)
wav_c, n_c, lbl_c = ds_crop[0]
wav_f, n_f, lbl_f = ds_full[0]
assert n_c <= 128000,   f"cropped length {n_c} > 128000"
assert n_f == wav_f.shape[0], "full length mismatch"
assert lbl_c == lbl_f, "label changed between crop/full"
print(f"[PASS] Dataset: cropped n={n_c} (≤128000), full n={n_f}")

# Collate
from sid_data import _collate  # internal, import directly
batch = [ds_full[0], ds_full[0]]
audios, lens, lbls = _collate(batch)
assert audios.shape[0] == 2
assert lens.shape[0] == 2
print(f"[PASS] _collate: batch shape {list(audios.shape)}")

print("\n=== ALL UNIT TESTS PASSED ===")
PYEOF

echo ""

# ── 3. End-to-end with tiny subset ───────────────────────────────────────────
echo "--- End-to-end (1 epoch, 32 examples/split, CPU, wav2vec2-base) ---"
echo ""

"${PYTHON}" sid_run.py \
    --probe final \
    --model_family hf \
    --model_id facebook/wav2vec2-base \
    --voxceleb1_root "${VOXCELEB1_ROOT}" \
    --epochs 1 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --warmup_steps 2 \
    --max_examples 32 \
    --train_max_duration_s 8.0 \
    --checkpoint_dir "${TMP}/ckpt" \
    --runs_dir       "${TMP}/runs" \
    --log_dir        "${TMP}/logs"

# ── 4. Verify summary JSON ────────────────────────────────────────────────────
SUMMARY=$(ls "${TMP}/runs"/*_sid_final_summary.json 2>/dev/null | head -1)
if [[ -z "${SUMMARY}" ]]; then
    echo "FAIL: no summary JSON found in ${TMP}/runs"
    exit 1
fi

echo ""
echo "=== Summary JSON ==="
cat "${SUMMARY}"

# Sanity-check test_acc is a real number between 0 and 1
"${PYTHON}" - <<PYEOF
import json, sys
data = json.load(open("${SUMMARY}"))
acc  = data["test_acc"]
assert isinstance(acc, float) and 0.0 <= acc <= 1.0, f"test_acc out of range: {acc}"
print(f"\n[PASS] test_acc={acc:.4f}  num_speakers={data['num_speakers']}")
# Confirm meta-based split was used (speakers should be 1251 not 40)
assert data["num_speakers"] == 1251, f"Expected 1251 speakers, got {data['num_speakers']}"
print("[PASS] num_speakers=1251 (SUPERB-compliant split confirmed)")
PYEOF

echo ""
echo "=== SMOKE TEST PASSED ==="
