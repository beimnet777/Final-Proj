#!/usr/bin/env bash
# Launch a single TensorBoard instance with four named, colour-coded groups.
#
# Sidebar run names will appear as:
#   ASR/lstm            ASR/weighted_lstm
#   ER/final/fold1 … fold5 / summary
#   ER/weighted/fold1 … fold5 / summary
#   PR/final            PR/weighted
#   SID/final           SID/weighted
#
# Each task gets its own colour in the sidebar and in all charts.
#
# Usage:
#   bash tools/launch_tensorboard.sh
#   then open  http://localhost:6006
#
# On CSD3 (HPC) forward the port to your laptop first:
#   ssh -L 6006:localhost:6006 bbg25@login.hpc.cam.ac.uk
#
# Stop with:  kill $(cat /tmp/tb_pid.txt)
#
# TIP — The "Custom Scalars" tab (top nav) shows every task's metrics in
#        named panels (Training / Error Rates / Accuracy / Layer Weights)
#        without needing to hunt through the sidebar.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TB_EXPORTS="${SCRIPT_DIR}/../tb_exports"
PYTHON="${PYTHON:-/home/bbg25/.conda/envs/mlmi4/bin/python}"
TB_CMD="${PYTHON} -m tensorboard.main"
PORT="${TB_PORT:-6006}"

if [[ ! -d "${TB_EXPORTS}" ]]; then
  echo "ERROR: tb_exports/ not found.  Run first:"
  echo "  python tools/export_to_tensorboard.py --clean"
  exit 1
fi

# Build the logdir_spec string: NAME:path pairs, comma-separated.
# Each NAME becomes a colour-coded prefix in the sidebar.
LOGDIR_SPEC="\
ASR:${TB_EXPORTS}/asr,\
ER:${TB_EXPORTS}/er,\
PR:${TB_EXPORTS}/pr,\
SID:${TB_EXPORTS}/sid"

PID_FILE="/tmp/tb_pid.txt"

echo "Starting TensorBoard on port ${PORT} ..."
${TB_CMD} \
  --logdir_spec="${LOGDIR_SPEC}" \
  --port="${PORT}" \
  --bind_all \
  > "/tmp/tb.log" 2>&1 &

echo $! > "${PID_FILE}"

echo ""
echo "TensorBoard running at  http://localhost:${PORT}"
echo ""
echo "Sidebar groups:"
echo "  ASR  — lstm, weighted_lstm"
echo "  ER   — final/fold1..5/summary, weighted/fold1..5/summary"
echo "  PR   — final, weighted"
echo "  SID  — final, weighted"
echo ""
echo ">>> Open the 'Custom Scalars' tab for organised per-task panels <<<"
echo ""
echo "Stop with:  kill \$(cat ${PID_FILE})"
