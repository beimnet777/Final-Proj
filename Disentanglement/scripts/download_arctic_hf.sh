#!/bin/bash
# Download CMU ARCTIC from the HuggingFace mirror MikhailT/cmu-arctic into
# Probing/data/CMU_ARCTIC_hf as a raw snapshot (parquet shards + README).
#
# We use a separate directory from the festvox layout (Probing/data/CMU_ARCTIC)
# so a later "materialize to wav + txt.done.data" step can be re-run without
# clobbering anything.  Total size on disk: ~1.6 GB.
#
# Idempotent: snapshot_download skips files that are already present and have
# matching hashes.
#
# Usage:
#   bash Disentanglement/scripts/download_arctic_hf.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/Probing/data"
TARGET_DIR="${DATA_ROOT}/CMU_ARCTIC_hf"
mkdir -p "${TARGET_DIR}"

PYTHON="${PYTHON:-python}"

echo "REPO_ROOT  = ${REPO_ROOT}"
echo "TARGET_DIR = ${TARGET_DIR}"
echo ""

# Route HF cache into the project so we don't pollute $HOME on the cluster.
export HF_HOME="${DATA_ROOT}/hf_home"
export HF_HUB_CACHE="${DATA_ROOT}/hub_cache"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

${PYTHON} - <<PY
from huggingface_hub import snapshot_download
import os, sys

target = os.environ.get("TARGET_DIR_PY") or r"""${TARGET_DIR}"""
path = snapshot_download(
    repo_id="MikhailT/cmu-arctic",
    repo_type="dataset",
    local_dir=target,
    local_dir_use_symlinks=False,
    max_workers=4,
)
print(f"snapshot at: {path}", flush=True)
PY

echo ""
echo "=== contents ==="
ls -lh "${TARGET_DIR}" | head -40
echo ""
sz=$(du -sh "${TARGET_DIR}" 2>/dev/null | cut -f1)
echo "total size: ${sz}"
echo ""
echo "Done.  Next step: decide whether to (A) materialize wavs into"
echo "${DATA_ROOT}/CMU_ARCTIC/cmu_us_<spk>_arctic/{wav,etc/txt.done.data}"
echo "or (B) point ARCTICIndex at the parquet directly."
