#!/usr/bin/env bash
# smoke_test.sh — fast end-to-end sanity check for the ASR pipeline.
#
# 1. Unit tests: split names, tokenizer, collation.
# 2. End-to-end: 1 training step + eval with wav2vec2-base on CPU.
#
# Usage (from inside asr/ or anywhere in the repo):
#   bash Final-Proj/Probing/asr/smoke_test.sh
#
set -euo pipefail

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
ASR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_CACHE="$(dirname "${ASR_DIR}")/data"
TMP="$(mktemp -d /tmp/asr_smoke_XXXXXX)"

cleanup() { rm -rf "${TMP}"; }
trap cleanup EXIT

echo "=== ASR smoke test ==="
echo "    ASR_DIR    : ${ASR_DIR}"
echo "    DATA_CACHE : ${DATA_CACHE}"
echo "    TMP        : ${TMP}"
echo ""

cd "${ASR_DIR}"

# ── 1. Unit tests ────────────────────────────────────────────────────────────
echo "--- Unit tests ---"

"${PYTHON}" - "${DATA_CACHE}" <<'PYEOF'
import sys, torch
sys.path.insert(0, ".")
from config import Config
from data import _stream_examples, CharTokenizer, LibriSpeechCharDataset, collate_fn

cfg = Config()
cfg.data_cache_dir = sys.argv[1]

N = 8  # small number to avoid downloading full shards

# Config sanity
assert not hasattr(cfg, "val_split"), "val_split should have been removed from Config"
assert cfg.train_hours == 100.0, f"train_hours should be 100.0, got {cfg.train_hours}"
print("[PASS] Config: val_split removed, train_hours=100.0")

# Split names are valid (HF librispeech_asr "clean" config)
print("Streaming a few examples from each split (requires network / HF cache)...")
train_ex = _stream_examples("train.100",  cfg, n=N)
val_ex   = _stream_examples("validation", cfg, n=N)   # dev-clean
test_ex  = _stream_examples("test",       cfg, n=N)   # test-clean
assert len(train_ex) == N, f"train: expected {N}, got {len(train_ex)}"
assert len(val_ex)   == N, f"val:   expected {N}, got {len(val_ex)}"
assert len(test_ex)  == N, f"test:  expected {N}, got {len(test_ex)}"
print(f"[PASS] Split names valid: train={len(train_ex)}, val={len(val_ex)}, test={len(test_ex)}")

# Val (dev-clean) and train (train-clean-100) must be disjoint
train_ids = {ex["id"] for ex in train_ex}
val_ids   = {ex["id"] for ex in val_ex}
assert train_ids.isdisjoint(val_ids), f"Overlap between train and val IDs: {train_ids & val_ids}"
print("[PASS] Train and val IDs are disjoint (no leakage)")

# Val and test must be disjoint
test_ids = {ex["id"] for ex in test_ex}
assert val_ids.isdisjoint(test_ids), f"Overlap between val and test IDs: {val_ids & test_ids}"
print("[PASS] Val and test IDs are disjoint")

# CharTokenizer round-trip
tok = CharTokenizer(cfg.vocab, blank_id=cfg.blank_id)
ids = tok.encode("hello world")
assert ids.dtype == torch.long
decoded = tok.decode(ids.tolist())
assert decoded == "hello world", f"Round-trip failed: {decoded!r}"
assert cfg.blank_id not in ids.tolist(), "blank_id should not appear in encoded output"
print("[PASS] CharTokenizer encode/decode round-trip")

# Dataset + collation
ds = LibriSpeechCharDataset(val_ex, tok, cfg.sample_rate)
audio, target, text = ds[0]
assert audio.dim() == 1, f"Expected 1-D audio, got shape {audio.shape}"
assert target.dim() == 1
assert isinstance(text, str) and len(text) > 0
print(f"[PASS] LibriSpeechCharDataset: audio={tuple(audio.shape)}, target={tuple(target.shape)}")

batch = [ds[i] for i in range(min(4, len(ds)))]
audios, audio_lens, targets, target_lens, texts = collate_fn(batch)
assert audios.shape[0] == len(batch), "batch size mismatch"
assert (audio_lens <= audios.shape[1]).all(), "audio_lens exceeds padded length"
assert (target_lens <= targets.shape[1]).all(), "target_lens exceeds padded length"
print(f"[PASS] collate_fn: audios={tuple(audios.shape)}, audio_lens={audio_lens.tolist()}")

print("\n=== ALL UNIT TESTS PASSED ===")
import sys, os; sys.stdout.flush(); os._exit(0)  # flush then skip PyArrow cleanup crash
PYEOF

echo ""

# ── 2. End-to-end: 1 training step + eval ────────────────────────────────────
echo "--- End-to-end (1 step, 8 examples/split, CPU, wav2vec2-base) ---"
echo ""

"${PYTHON}" - "${DATA_CACHE}" "${TMP}" <<'PYEOF'
import sys, random, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
from config import Config
from data import _stream_examples, CharTokenizer, LibriSpeechCharDataset, collate_fn
from model import build_model
from torch.utils.data import DataLoader

cfg = Config()
cfg.data_cache_dir   = sys.argv[1]
cfg.checkpoint_dir   = sys.argv[2] + "/ckpt"
cfg.model_id         = "facebook/wav2vec2-base"
cfg.model_family     = "hf"
cfg.probe_type       = "final"
cfg.batch_size       = 4
cfg.eval_batch_size  = 4

N = 8
print("Loading splits...")
train_ex = _stream_examples("train.100",  cfg, n=N)
val_ex   = _stream_examples("validation", cfg, n=N)
test_ex  = _stream_examples("test",       cfg, n=N)

tok = CharTokenizer(cfg.vocab, blank_id=cfg.blank_id)
make_dl = lambda exs, shuffle: DataLoader(
    LibriSpeechCharDataset(exs, tok, cfg.sample_rate),
    batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate_fn,
)
train_dl = make_dl(train_ex, True)
val_dl   = make_dl(val_ex,   False)
test_dl  = make_dl(test_ex,  False)

print("Building model (wav2vec2-base, CPU)...")
encoder, probe = build_model(cfg)
device = torch.device("cpu")
encoder.to(device); probe.to(device)

# One training step
probe.train(); encoder.eval()
audios, audio_lens, targets, target_lens, _ = next(iter(train_dl))
audios = audios.to(device); audio_lens = audio_lens.to(device)
targets = targets.to(device); target_lens = target_lens.to(device)

with torch.no_grad():
    layers = encoder(audios, audio_lens)
    frame_lens = encoder.output_lengths(audio_lens)

logits = probe(layers)
log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
loss = nn.CTCLoss(blank=cfg.blank_id, zero_infinity=True)(
    log_probs, targets, frame_lens, target_lens
)
assert torch.isfinite(loss), f"CTC loss is not finite: {loss.item()}"
print(f"[PASS] Training step: CTC loss={loss.item():.4f}")

# One eval batch
probe.eval()
audios, audio_lens, _, _, texts = next(iter(val_dl))
with torch.no_grad():
    layers = encoder(audios.to(device), audio_lens.to(device))
    frame_lens = encoder.output_lengths(audio_lens.to(device))
    logits = probe(layers)
assert logits.shape[2] == cfg.vocab_size, \
    f"logits vocab dim {logits.shape[2]} != vocab_size {cfg.vocab_size}"
print(f"[PASS] Eval batch: logits shape {tuple(logits.shape)}")

# One test batch
audios, audio_lens, _, _, _ = next(iter(test_dl))
with torch.no_grad():
    layers = encoder(audios.to(device), audio_lens.to(device))
    logits = probe(layers)
print(f"[PASS] Test batch:  logits shape {tuple(logits.shape)}")

print("\n=== END-TO-END TEST PASSED ===")
import sys, os; sys.stdout.flush(); os._exit(0)  # flush then skip PyArrow cleanup crash
PYEOF

echo ""
echo "=== SMOKE TEST PASSED ==="
