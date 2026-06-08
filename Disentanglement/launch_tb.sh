#!/bin/bash
# launch_tb.sh — one-stop TensorBoard for the whole disentanglement project.
#
# Auto-discovers every run under runs/<name>/tb/<stage>_<timestamp>/ and serves
# them all in a single TensorBoard with clean, comparable names like:
#     beta_003_s2 , ste_s2 , decor_only_s1 , decor_only_s2 , ...
#
# Usage:
#   bash launch_tb.sh                 # default port 6006
#   bash launch_tb.sh 6010            # custom port
#   bash launch_tb.sh 6006 --include-legacy   # also show pre-rename runs/tb/*
#
# Remote (HPC) — open a tunnel from your laptop, then browse localhost:PORT:
#   ssh -N -L 6006:<node>:6006 bbg25@login.hpc.cam.ac.uk

set -euo pipefail

DIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${DIS_DIR}/runs"
PYTHON=/home/bbg25/.conda/envs/mlmi4/bin/python
TB=/home/bbg25/.conda/envs/mlmi4/bin/tensorboard

PORT="${1:-6006}"
INCLUDE_LEGACY=0
[[ "${2:-}" == "--include-legacy" ]] && INCLUDE_LEGACY=1

# ---- build a clean name->path mapping for every event directory ----------
declare -A seen
spec=""

stage_tag() {           # leaf dir name -> short stage tag
    case "$1" in
        stage1*) echo "s1" ;;
        stage2*) echo "s2" ;;
        sae*)    echo "sae" ;;
        *)       echo "$(echo "$1" | cut -d_ -f1)" ;;
    esac
}

add_entry() {           # $1 = name, $2 = path
    local name="$1" path="$2"
    if [[ -n "${seen[$name]:-}" ]]; then
        seen[$name]=$(( seen[$name] + 1 ))
        name="${name}_${seen[$name]}"
    else
        seen[$name]=1
    fi
    [[ -n "$spec" ]] && spec="${spec},"
    spec="${spec}${name}:${path}"
}

# Discover leaf event directories (those directly containing *.tfevents.*)
while IFS= read -r ev; do
    leafdir="$(dirname "$ev")"
    rel="${leafdir#${RUNS_DIR}/}"             # e.g. beta_003/tb/stage2_2026...
    run="${rel%%/*}"                          # beta_003   (or 'tb' for legacy)
    leaf="$(basename "$leafdir")"             # stage2_2026...

    if [[ "$run" == "tb" ]]; then
        [[ "$INCLUDE_LEGACY" -eq 0 ]] && continue
        add_entry "legacy_${leaf}" "$leafdir"
    else
        add_entry "${run}_$(stage_tag "$leaf")" "$leafdir"
    fi
done < <(find "$RUNS_DIR" -name "*.tfevents.*" | sort)

if [[ -z "$spec" ]]; then
    echo "No tfevents found under ${RUNS_DIR}"; exit 1
fi

echo "=== Disentanglement TensorBoard ==="
echo "Runs discovered:"
echo "$spec" | tr ',' '\n' | sed 's/:.*//' | sed 's/^/   /'
echo ""
echo "Node : $(hostname)"
echo "Port : ${PORT}"
echo "Tunnel from laptop:"
echo "   ssh -N -L ${PORT}:$(hostname):${PORT} ${USER}@login.hpc.cam.ac.uk"
echo "Then open: http://localhost:${PORT}"
echo ""

exec "$TB" --logdir_spec "$spec" --port "$PORT" --bind_all --reload_multifile true
