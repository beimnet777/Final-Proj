#!/usr/bin/env bash
# smoke_test.sh — fast end-to-end sanity check for the PR pipeline.
#
# Streams 32 examples from each LibriSpeech split (train/val/test),
# runs pr_run.py for 1 epoch on CPU with facebook/wav2vec2-base,
# and checks that the summary JSON is written.
#
# Usage (from inside pr/ or anywhere in the repo):
#   bash Final-Proj/Probing/pr/smoke_test.sh
#
set -euo pipefail

PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
PR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_CACHE="$(dirname "${PR_DIR}")/data"
TMP="$(mktemp -d /tmp/pr_smoke_XXXXXX)"

cleanup() { rm -rf "${TMP}"; }
trap cleanup EXIT

echo "=== PR smoke test ==="
echo "    PR_DIR     : ${PR_DIR}"
echo "    DATA_CACHE : ${DATA_CACHE}"
echo "    TMP        : ${TMP}"
echo ""

# ── 1. Lexicon (download once to shared cache if missing) ────────────────────
LEXICON_PATH="${DATA_CACHE}/librispeech-lexicon.txt"
if [[ ! -f "${LEXICON_PATH}" ]]; then
    echo "Downloading LibriSpeech lexicon..."
    mkdir -p "${DATA_CACHE}"
    wget -q https://www.openslr.org/resources/11/librispeech-lexicon.txt \
         -O "${LEXICON_PATH}"
fi

# ── 2. Run pr_run.py with tiny settings ──────────────────────────────────────
# - facebook/wav2vec2-base (~95 MB, HF family) instead of SPEAR-XLarge (~1.7 GB)
#   so the test finishes in a few minutes on CPU.
# - max_examples=32 streams only 32 utterances per split.
# - 1 epoch, batch=4, warmup=2 steps.
cd "${PR_DIR}"

echo "Running pr_run.py (1 epoch, 32 examples/split, CPU, wav2vec2-base)..."
echo ""

"${PYTHON}" pr_run.py \
    --probe final \
    --model_family hf \
    --model_id facebook/wav2vec2-base \
    --epochs 1 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --warmup_steps 2 \
    --max_examples 32 \
    --data_cache_dir "${DATA_CACHE}" \
    --lexicon_path "${LEXICON_PATH}" \
    --checkpoint_dir "${TMP}/ckpt" \
    --runs_dir "${TMP}/runs" \
    --log_dir "${TMP}/logs"

# ── 3. Verify summary JSON was written ───────────────────────────────────────
SUMMARY=$(ls "${TMP}/runs"/*_pr_final_summary.json 2>/dev/null | head -1)
if [[ -z "${SUMMARY}" ]]; then
    echo ""
    echo "FAIL: no summary JSON found in ${TMP}/runs"
    exit 1
fi

echo ""
echo "=== Summary JSON ==="
cat "${SUMMARY}"
echo ""
echo "=== SMOKE TEST PASSED ==="
