#!/usr/bin/env bash
# Prepare a Git clone for Disentanglement work on CBL blackwell.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: setup.sh [--download-librispeech]

Creates an isolated environment and scratch directories. The optional flag
downloads train-clean-100, dev-clean and test-clean from OpenSLR.

Environment overrides:
  BLACKWELL_PYTHON          Python executable (default: python3)
  BLACKWELL_SCRATCH_ROOT    Storage root (default: /scratch/$USER)
  BLACKWELL_VENV            Virtual environment path
  BLACKWELL_DATA_ROOT       Dataset path
  PYTORCH_INDEX_URL         Official wheel index (default: cu130)
EOF
}

download_librispeech=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --download-librispeech) download_librispeech=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

BLACKWELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BLACKWELL_DIR}/../.." && pwd)"
PYTHON_BIN="${BLACKWELL_PYTHON:-python3}"
SCRATCH_ROOT="${BLACKWELL_SCRATCH_ROOT:-/scratch/${USER:?USER is not set}}"
VENV="${BLACKWELL_VENV:-${SCRATCH_ROOT}/venvs/final-proj}"
DATA_ROOT="${BLACKWELL_DATA_ROOT:-${SCRATCH_ROOT}/data}"
HF_ROOT="${SCRATCH_ROOT}/hf"
NLTK_ROOT="${SCRATCH_ROOT}/nltk_data"

command -v "$PYTHON_BIN" >/dev/null || {
    echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
    exit 1
}
"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 15)):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is unsupported by this setup; "
        "select Python 3.10 through 3.14 with BLACKWELL_PYTHON.")
PY

mkdir -p "$SCRATCH_ROOT" "$DATA_ROOT" "$HF_ROOT" "$NLTK_ROOT" "$(dirname "$VENV")"
if [[ ! -x "$VENV/bin/python" ]]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Match the Colab strategy by keeping a compatible system build, but require the
# exact Torch/TorchAudio release pair used by this repository. TorchAudio links
# against Torch C++ extensions and mixed release lines are unsupported.
if ! python - <<'PY' >/dev/null 2>&1
import torch, torchaudio
assert torch.__version__.split("+")[0] == "2.11.0"
assert torchaudio.__version__.split("+")[0] == "2.11.0"
assert tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 8)
PY
then
    PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
    python -m pip install --upgrade --force-reinstall \
        torch==2.11.0 torchaudio==2.11.0 --index-url "$PYTORCH_INDEX_URL"
fi
python -m pip install -r "$REPO_ROOT/Disentanglement/requirements-colab.txt"

NLTK_DATA="$NLTK_ROOT" python - <<'PY'
import nltk

resources = {
    "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "cmudict": "corpora/cmudict",
}
for package, resource in resources.items():
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, quiet=True, raise_on_error=True)
        nltk.data.find(resource)
PY

download() {
    local url="$1" output="$2"
    if command -v wget >/dev/null; then
        wget -c -O "$output" "$url"
    elif command -v curl >/dev/null; then
        curl -fL --retry 5 -C - -o "$output" "$url"
    else
        echo "ERROR: install wget or curl to download LibriSpeech" >&2
        return 1
    fi
}

if (( download_librispeech )); then
    mkdir -p "$DATA_ROOT/archives"
    for split in train-clean-100 dev-clean test-clean; do
        archive="$DATA_ROOT/archives/${split}.tar.gz"
        download "https://www.openslr.org/resources/12/${split}.tar.gz" "$archive"
        tar -xzf "$archive" -C "$DATA_ROOT"
    done
fi

if [[ -d "$DATA_ROOT/LibriSpeech/train-clean-100" ]]; then
    cp "$REPO_ROOT/Probing/data/librispeech-lexicon.txt" \
       "$DATA_ROOT/librispeech-lexicon.txt"
fi

python - <<'PY'
import sys, torch, torchaudio
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("torchaudio:", torchaudio.__version__)
if torch.version.cuda is None:
    raise SystemExit("ERROR: installed PyTorch is CPU-only")
if torch.__version__.split("+")[0] != "2.11.0" or torchaudio.__version__.split("+")[0] != "2.11.0":
    raise SystemExit("ERROR: expected matching torch/torchaudio 2.11.0 releases")
print("GPU initialization deferred until an allocated GPU is selected.")
PY

{
    printf 'setup_time=%s\n' "$(date --iso-8601=seconds)"
    printf 'git_commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'python=%s\n' "$(python --version 2>&1)"
    printf 'venv=%s\n' "$VENV"
    printf 'data_root=%s\n' "$DATA_ROOT"
} > "$SCRATCH_ROOT/setup_metadata.txt"
python -m pip freeze > "$SCRATCH_ROOT/environment.freeze.txt"

echo
echo "Blackwell environment setup complete."
echo "Environment: $VENV"
echo "Data root:   $DATA_ROOT"
echo "Package snapshot: $SCRATCH_ROOT/environment.freeze.txt"
