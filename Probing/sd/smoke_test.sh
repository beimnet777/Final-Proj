#!/usr/bin/env bash
# smoke_test.sh — fast end-to-end sanity check for the SD pipeline.
#
# Creates a tiny temp ASVspoof2019 LA layout (symlinks to real files),
# runs sd_run.py for 1 epoch on CPU, and checks the summary JSON is written.
#
# Usage (from inside sd/ or the repo root):
#   bash Final-Proj/Probing/sd/smoke_test.sh
#
set -euo pipefail

PYTHON="/home/bbg25/.conda/envs/mlmi4/bin/python"
REAL_LA="/rds/user/bbg25/hpc-work/data/ASVspoof2019/LA"
SD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d /tmp/sd_smoke_XXXXXX)"

cleanup() { rm -rf "${TMP}"; }
trap cleanup EXIT

echo "=== SD smoke test ==="
echo "    SD_DIR  : ${SD_DIR}"
echo "    REAL_LA : ${REAL_LA}"
echo "    TMP     : ${TMP}"
echo ""

# ── 1. Build tiny ASVspoof2019 LA layout ─────────────────────────────────────
# We use symlinks to real audio files so no copying is needed.
PROTO_DIR="${TMP}/LA/ASVspoof2019_LA_cm_protocols"
TRAIN_DIR="${TMP}/LA/ASVspoof2019_LA_train/flac"
DEV_DIR="${TMP}/LA/ASVspoof2019_LA_dev/flac"
EVAL_DIR="${TMP}/LA/ASVspoof2019_LA_eval/flac"

mkdir -p "${PROTO_DIR}" "${TRAIN_DIR}" "${DEV_DIR}" "${EVAL_DIR}"

# Pick 8 bonafide + 8 spoof from train; same for dev and eval
for SPLIT in train dev eval; do
    case "${SPLIT}" in
        train) REAL_DIR="${REAL_LA}/ASVspoof2019_LA_train/flac"
               PROTO_SRC="${REAL_LA}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
               LINK_DIR="${TRAIN_DIR}" ;;
        dev)   REAL_DIR="${REAL_LA}/ASVspoof2019_LA_dev/flac"
               PROTO_SRC="${REAL_LA}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
               LINK_DIR="${DEV_DIR}" ;;
        eval)  REAL_DIR="${REAL_LA}/ASVspoof2019_LA_eval/flac"
               PROTO_SRC="${REAL_LA}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
               LINK_DIR="${EVAL_DIR}" ;;
    esac

    # Grab 8 bonafide and 8 spoof lines from the real protocol
    # (|| true prevents pipefail from triggering on grep+head SIGPIPE)
    BON=$(grep " bonafide$" "${PROTO_SRC}" | head -8 || true)
    SPO=$(grep " spoof$"    "${PROTO_SRC}" | head -8 || true)
    LINES=$(printf '%s\n%s\n' "${BON}" "${SPO}")

    # Write tiny protocol file
    case "${SPLIT}" in
        train) echo "${LINES}" > "${PROTO_DIR}/ASVspoof2019.LA.cm.train.trn.txt" ;;
        dev)   echo "${LINES}" > "${PROTO_DIR}/ASVspoof2019.LA.cm.dev.trl.txt"   ;;
        eval)  echo "${LINES}" > "${PROTO_DIR}/ASVspoof2019.LA.cm.eval.trl.txt"  ;;
    esac

    # Symlink the actual flac files
    while IFS= read -r line; do
        utt_id=$(echo "${line}" | awk '{print $2}')
        src="${REAL_DIR}/${utt_id}.flac"
        [[ -f "${src}" ]] && ln -sf "${src}" "${LINK_DIR}/${utt_id}.flac"
    done <<< "${LINES}"
done

echo "Tiny dataset:"
echo "  train flac: $(ls ${TRAIN_DIR} | wc -l)"
echo "  dev   flac: $(ls ${DEV_DIR}   | wc -l)"
echo "  eval  flac: $(ls ${EVAL_DIR}  | wc -l)"
echo ""

# ── 2. Run sd_run.py ──────────────────────────────────────────────────────────
RUNS_DIR="${TMP}/runs"
mkdir -p "${RUNS_DIR}"

cd "${SD_DIR}"

echo "Running sd_run.py (1 epoch, batch=2, CPU, wav2vec2-base for speed)..."
echo ""

# Use facebook/wav2vec2-base (~95 MB) instead of SPEAR-XLarge (~1.7 GB)
# so the smoke test finishes in <5 min on CPU.
# This validates: data loading, model build, training loop, EER eval, JSON output.
"${PYTHON}" sd_run.py \
    --probe weighted \
    --asv19_la_root "${TMP}/LA" \
    --model_family hf \
    --model_id facebook/wav2vec2-base \
    --epochs 1 \
    --batch_size 2 \
    --eval_batch_size 2 \
    --warmup_steps 2 \
    --runs_dir "${RUNS_DIR}" \
    --checkpoint_dir "${TMP}/ckpt" \
    --log_dir "${TMP}/logs"

# ── 3. Verify summary JSON was written ───────────────────────────────────────
SUMMARY=$(ls "${RUNS_DIR}"/*_sd_weighted_summary.json 2>/dev/null | head -1)
if [[ -z "${SUMMARY}" ]]; then
    echo ""
    echo "FAIL: no summary JSON found in ${RUNS_DIR}"
    exit 1
fi

echo ""
echo "=== Summary JSON ==="
cat "${SUMMARY}"
echo ""
echo "=== SMOKE TEST PASSED ==="
