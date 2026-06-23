#!/bin/bash
# Download CMU ARCTIC, VCTK, and (best-effort) ESD into Probing/data/, next to
# the existing LibriSpeech tree.  Idempotent: safe to re-run, skips anything
# already present.
#
# Cluster usage:
#   bash Disentanglement/scripts/download_invariance_corpora.sh
# Local usage:
#   bash Disentanglement/scripts/download_invariance_corpora.sh
#
# ESD requires a Google Drive download — if `gdown` is missing the script
# prints the manual link and exits non-zero on that corpus only.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/Probing/data"
mkdir -p "${DATA_ROOT}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "(LibriSpeech expected at ${DATA_ROOT}/LibriSpeech)"
echo ""

# Pick downloader
if command -v wget >/dev/null 2>&1; then
    DL="wget --quiet --show-progress -c -O"
else
    DL="curl -fL --retry 3 --retry-delay 5 -C - -o"
fi

# ---------------------------------------------------------------- CMU ARCTIC
ARCTIC_DIR="${DATA_ROOT}/CMU_ARCTIC"
mkdir -p "${ARCTIC_DIR}"
ARCTIC_SPEAKERS=(awb bdl clb jmk ksp rms slt)
ARCTIC_URL_BASE="http://festvox.org/cmu_arctic/cmu_arctic/packed"

echo "=== CMU ARCTIC (7 speakers, ~7 GB total) ==="
for spk in "${ARCTIC_SPEAKERS[@]}"; do
    spk_dir="${ARCTIC_DIR}/cmu_us_${spk}_arctic"
    archive="${ARCTIC_DIR}/cmu_us_${spk}_arctic-0.95-release.tar.bz2"
    if [[ -d "${spk_dir}/wav" ]]; then
        echo "  [skip] ${spk}: already extracted"
        continue
    fi
    if [[ ! -f "${archive}" ]]; then
        echo "  [get ] ${spk}"
        ${DL} "${archive}" "${ARCTIC_URL_BASE}/cmu_us_${spk}_arctic-0.95-release.tar.bz2" || {
            echo "  [fail] ${spk} download"; continue; }
    fi
    echo "  [tar ] ${spk}"
    tar -xjf "${archive}" -C "${ARCTIC_DIR}" && rm -f "${archive}"
done
echo "ARCTIC done.  Layout: ${ARCTIC_DIR}/cmu_us_<speaker>_arctic/{wav,etc/txt.done.data}"
echo ""

# ---------------------------------------------------------------- VCTK
VCTK_DIR="${DATA_ROOT}/VCTK"
VCTK_ARCHIVE="${DATA_ROOT}/VCTK-Corpus-0.92.zip"
VCTK_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

echo "=== VCTK (110 speakers, ~10 GB) ==="
if [[ -d "${VCTK_DIR}/wav48_silence_trimmed" ]]; then
    echo "  [skip] VCTK: already extracted"
else
    if [[ ! -f "${VCTK_ARCHIVE}" ]]; then
        echo "  [get ] VCTK archive"
        ${DL} "${VCTK_ARCHIVE}" "${VCTK_URL}" || echo "  [warn] VCTK download incomplete"
    fi
    if [[ -f "${VCTK_ARCHIVE}" ]]; then
        echo "  [unzip] VCTK"
        mkdir -p "${VCTK_DIR}"
        unzip -q -n "${VCTK_ARCHIVE}" -d "${VCTK_DIR}" && rm -f "${VCTK_ARCHIVE}"
    fi
fi
echo "VCTK done.  Layout: ${VCTK_DIR}/wav48_silence_trimmed/<speaker>/*.flac + ${VCTK_DIR}/txt/<speaker>/*.txt"
echo ""

# ---------------------------------------------------------------- ESD
ESD_DIR="${DATA_ROOT}/ESD"
ESD_ARCHIVE="${DATA_ROOT}/ESD.zip"
# Google Drive file id from the HLTSingapore/Emotional-Speech-Data README
ESD_GDRIVE_ID="1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-D"

echo "=== ESD (10 EN + 10 ZH speakers x 350 sentences x 5 emotions, ~7 GB) ==="
if [[ -d "${ESD_DIR}" ]] && ls "${ESD_DIR}" 2>/dev/null | grep -qE '^00[12][0-9]$'; then
    echo "  [skip] ESD: already extracted"
else
    mkdir -p "${ESD_DIR}"
    if [[ ! -f "${ESD_ARCHIVE}" ]]; then
        if command -v gdown >/dev/null 2>&1; then
            echo "  [get ] ESD via gdown"
            gdown --id "${ESD_GDRIVE_ID}" -O "${ESD_ARCHIVE}" || echo "  [fail] gdown — see manual instructions below"
        else
            echo "  [warn] gdown not installed.  ESD must be downloaded manually:"
            echo "         https://drive.google.com/file/d/${ESD_GDRIVE_ID}/view"
            echo "         then place at ${ESD_ARCHIVE} and re-run this script."
        fi
    fi
    if [[ -f "${ESD_ARCHIVE}" ]]; then
        echo "  [unzip] ESD"
        unzip -q -n "${ESD_ARCHIVE}" -d "${ESD_DIR}" && rm -f "${ESD_ARCHIVE}"
    fi
fi
echo "ESD done.  Layout: ${ESD_DIR}/00<NN>/{Angry,Happy,Neutral,Sad,Surprise}/*.wav  (speakers 0011-0020 are English)"
echo ""

# ---------------------------------------------------------------- summary
echo "=== Summary ==="
for d in "${ARCTIC_DIR}" "${VCTK_DIR}" "${ESD_DIR}" "${DATA_ROOT}/LibriSpeech"; do
    if [[ -d "${d}" ]]; then
        sz=$(du -sh "${d}" 2>/dev/null | cut -f1)
        printf "  %-40s  %s\n" "$(basename "${d}")" "${sz}"
    else
        printf "  %-40s  MISSING\n" "$(basename "${d}")"
    fi
done
echo ""
echo "Next: run pyworld extraction:"
echo "  python Disentanglement/scripts/extract_prosody.py --data_root ${DATA_ROOT}"
