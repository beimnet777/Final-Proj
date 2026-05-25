#!/usr/bin/env bash
# =============================================================================
# Download spoof-detection datasets for the SD probing pipeline.
#
# Usage:
#   bash sd/download_data.sh [TARGET_DIR]
#
# TARGET_DIR defaults to /rds/user/${USER}/hpc-work/data
#
# What is auto-downloaded (public Zenodo records, no login required):
#   • ASVspoof 2019 LA    — Zenodo 3816685
#   • ASVspoof 2021 LA eval — Zenodo 4837263  (audio only; keys below)
#   • ASVspoof 2021 DF eval — Zenodo 4835108  (audio only; keys below)
#   • ASVspoof 2021 eval keys — Zenodo 4835070
#
# What requires manual download (registration / access request):
#   • In-The-Wild (ITW)
#   • DFEval 2024
#   • Famous Figures
#   • ASVSpoofLD
#
# Expected final layout (passed to sd_config.py / sd_run.py):
#   TARGET/
#     ASVspoof2019/LA/
#       ASVspoof2019_LA_cm_protocols/   ← protocol .txt files
#       ASVspoof2019_LA_train/flac/
#       ASVspoof2019_LA_dev/flac/
#       ASVspoof2019_LA_eval/flac/
#     ASVspoof2021_LA/
#       flac/                            ← eval audio
#     ASVspoof2021_DF/
#       flac/                            ← eval audio
#     ASVspoof2021_keys/
#       keys_2021_LA_eval.txt
#       keys_2021_DF_eval.txt
#     ITW/                               ← manual
#     DFEval2024/                        ← manual
#     FamousFigures/                     ← manual
#     ASVSpoofLD/                        ← manual
# =============================================================================

set -euo pipefail

TARGET="${1:-/rds/user/${USER}/hpc-work/data}"
mkdir -p "${TARGET}"

echo "Download target: ${TARGET}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Helper: download all files from a Zenodo record using the public REST API.
# Skips files that already exist.
# ─────────────────────────────────────────────────────────────────────────────
zenodo_download() {
    local record_id="$1"
    local dest_dir="$2"
    mkdir -p "${dest_dir}"
    echo "  [zenodo ${record_id}] fetching file list …"
    local api_json
    api_json=$(curl -fsSL "https://zenodo.org/api/records/${record_id}")
    echo "${api_json}" | python3 -c "
import json, sys, os, subprocess
dest = '${dest_dir}'
data = json.load(sys.stdin)
files = data.get('files', [])
if not files:
    # Newer Zenodo API uses 'entries' key
    files = data.get('metadata', {}).get('files', [])
for f in files:
    fname = f.get('key') or f.get('filename')
    url   = (f.get('links') or {}).get('self') or f.get('url') or ''
    if not fname or not url:
        continue
    out = os.path.join(dest, fname)
    if os.path.exists(out):
        print(f'  [skip]  {fname}')
        continue
    print(f'  [wget]  {fname}')
    subprocess.run(['wget', '-q', '--show-progress', '-O', out, url], check=True)
"
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract all archives in a directory into a target directory.
# ─────────────────────────────────────────────────────────────────────────────
extract_all() {
    local src_dir="$1"
    local dst_dir="$2"
    mkdir -p "${dst_dir}"
    for f in "${src_dir}"/*.zip "${src_dir}"/*.tar.gz "${src_dir}"/*.tgz; do
        [[ -f "$f" ]] || continue
        echo "  [extract] $(basename $f) → ${dst_dir}"
        case "$f" in
            *.zip)           unzip -q "$f" -d "${dst_dir}" ;;
            *.tar.gz|*.tgz)  tar -xzf "$f" -C "${dst_dir}" ;;
        esac
    done
}

# =============================================================================
# 1. ASVspoof 2019 LA  (Edinburgh DataShare — direct public download)
# =============================================================================
echo "=== 1/4  ASVspoof 2019 LA (Edinburgh DataShare) ==="
ASV19_OUT="${TARGET}/ASVspoof2019"
ASV19_ZIP="${ASV19_OUT}/LA.zip"
mkdir -p "${ASV19_OUT}"
if [[ ! -f "${ASV19_ZIP}" ]]; then
    echo "  [wget]  LA.zip (~7.1 GB) …"
    wget -q --show-progress \
        -O "${ASV19_ZIP}" \
        "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
else
    echo "  [skip]  LA.zip already downloaded"
fi
if [[ ! -d "${ASV19_OUT}/LA" ]]; then
    echo "  [extract] LA.zip → ${ASV19_OUT}"
    unzip -q "${ASV19_ZIP}" -d "${ASV19_OUT}"
else
    echo "  [skip]  LA/ already extracted"
fi
echo "  → ${ASV19_OUT}/LA/"
echo ""

# =============================================================================
# 2. ASVspoof 2021 LA eval audio
# =============================================================================
echo "=== 2/4  ASVspoof 2021 LA eval audio (Zenodo 4837263) ==="
ASV21_LA_DL="${TARGET}/ASVspoof2021_LA/_download"
ASV21_LA_OUT="${TARGET}/ASVspoof2021_LA"
zenodo_download 4837263 "${ASV21_LA_DL}"
extract_all     "${ASV21_LA_DL}" "${ASV21_LA_OUT}"
echo "  → ${ASV21_LA_OUT}/"
echo ""

# =============================================================================
# 3. ASVspoof 2021 DF eval audio
# =============================================================================
echo "=== 3/4  ASVspoof 2021 DF eval audio (Zenodo 4835108) ==="
ASV21_DF_DL="${TARGET}/ASVspoof2021_DF/_download"
ASV21_DF_OUT="${TARGET}/ASVspoof2021_DF"
zenodo_download 4835108 "${ASV21_DF_DL}"
extract_all     "${ASV21_DF_DL}" "${ASV21_DF_OUT}"
echo "  → ${ASV21_DF_OUT}/"
echo ""

# =============================================================================
# 4. ASVspoof 2021 eval keys  (hosted directly on asvspoof.org)
# =============================================================================
echo "=== 4/4  ASVspoof 2021 eval keys (asvspoof.org) ==="
ASV21_KEYS_OUT="${TARGET}/ASVspoof2021_keys"
mkdir -p "${ASV21_KEYS_OUT}"
for subset in LA DF; do
    tarball="${ASV21_KEYS_OUT}/${subset}-keys-full.tar.gz"
    if [[ ! -f "${tarball}" ]]; then
        echo "  [wget]  ${subset}-keys-full.tar.gz"
        wget -q --show-progress \
            -O "${tarball}" \
            "https://www.asvspoof.org/asvspoof2021/${subset}-keys-full.tar.gz"
    else
        echo "  [skip]  ${subset}-keys-full.tar.gz already downloaded"
    fi
    echo "  [extract] ${subset}-keys-full.tar.gz → ${ASV21_KEYS_OUT}"
    tar -xzf "${tarball}" -C "${ASV21_KEYS_OUT}"
done
echo "  → ${ASV21_KEYS_OUT}/"
echo ""

# =============================================================================
# Manual-download datasets
# =============================================================================
cat << MANUAL
=============================================================================
The following datasets require manual download (registration / access form):
=============================================================================

┌─ In-The-Wild (ITW) ─────────────────────────────────────────────────────────
│  Paper  : Müller et al., "Does Audio Deepfake Detection Generalize?", IS2022
│  Form   : https://deepfake-total.com/in_the_wild
│  Place  : ${TARGET}/ITW/
│  Layout : ITW/{bonafide,spoof}/*.flac   OR   ITW/meta.csv + audio/
└──────────────────────────────────────────────────────────────────────────────

┌─ DFEval 2024 ───────────────────────────────────────────────────────────────
│  Site   : https://deepfake-2024.github.io/
│  Place  : ${TARGET}/DFEval2024/
└──────────────────────────────────────────────────────────────────────────────

┌─ Famous Figures ────────────────────────────────────────────────────────────
│  Paper  : https://arxiv.org/abs/2406.06052
│  Place  : ${TARGET}/FamousFigures/
└──────────────────────────────────────────────────────────────────────────────

┌─ ASVSpoofLD ────────────────────────────────────────────────────────────────
│  Site   : https://www.asvspoof.org/
│  Place  : ${TARGET}/ASVSpoofLD/
└──────────────────────────────────────────────────────────────────────────────

Once placed, pass the paths via CLI args or sd_config.py defaults.
MANUAL

echo ""
echo "=== Auto-download complete ==="
echo "Paths to pass to sd_run.py:"
echo "  --asv19_la_root    ${ASV19_OUT}/LA"
echo "  --asv21_la_root    ${ASV21_LA_OUT}"
echo "  --asv21_la_keys    ${ASV21_KEYS_OUT}/keys_2021_LA_eval.txt"
echo "  --asv21_df_root    ${ASV21_DF_OUT}"
echo "  --asv21_df_keys    ${ASV21_KEYS_OUT}/keys_2021_DF_eval.txt"
