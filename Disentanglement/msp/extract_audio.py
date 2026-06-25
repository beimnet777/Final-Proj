#!/usr/bin/env python3
"""Stream-extract only the subset WAVs from MSP-Podcast Audios.tar.gz.

GNU tar 1.30 mishandles this macOS/PAX archive (bails early), so we stream with
Python's tarfile (one sequential pass) and extract only members in members.txt.

Resume-friendly: members already present on disk are skipped (the gzip stream is
still scanned sequentially — it is not seekable — but nothing is re-written).
Robust: dest dirs are pre-created and per-member errors don't abort the run.
"""
import argparse
import sys
import tarfile
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tar", default="/rds/project/rds-xyBFuSj0hm0/dataset/"
                                  "MSP-Podcast-2.0/Audios.tar.gz")
ap.add_argument("--members", required=True)
ap.add_argument("--dest", required=True)
args = ap.parse_args()

wanted = {ln.strip() for ln in open(args.members) if ln.strip()}
dest = Path(args.dest)
dest.mkdir(parents=True, exist_ok=True)
# Pre-create the sub-dirs the archive uses, so per-file extraction never races
# on makedirs (the failure mode that aborted the first run).
for sub in {Path(m).parent for m in wanted}:
    (dest / sub).mkdir(parents=True, exist_ok=True)
print(f"[extract] want {len(wanted)} members -> {dest}", flush=True)

already = sum(1 for m in wanted if (dest / m).exists() and (dest / m).stat().st_size > 0)
print(f"[extract] {already} already present, resuming", flush=True)

n_seen = n_ok = n_err = 0
t0 = time.time()
with tarfile.open(args.tar, "r|gz") as tar:           # streaming, non-seekable
    for m in tar:
        n_seen += 1
        if n_seen % 20000 == 0:
            print(f"[extract] scanned {n_seen:,}, have {n_ok + already:,}/{len(wanted)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if not m.isfile() or m.name not in wanted:
            continue
        out = dest / m.name
        if out.exists() and out.stat().st_size > 0:    # resume: skip existing
            continue
        if m.name.startswith("/") or ".." in Path(m.name).parts:
            continue
        try:
            with tar.extractfile(m) as src:            # write bytes ourselves —
                out.write_bytes(src.read())            # no makedirs race
            n_ok += 1
        except Exception as e:                         # survive transient FS hiccups
            n_err += 1
            print(f"[extract] ERROR on {m.name}: {e}", file=sys.stderr, flush=True)
        if n_ok + already >= len(wanted):
            print("[extract] all members present, stopping scan", flush=True)
            break

have = sum(1 for m in wanted if (dest / m).exists() and (dest / m).stat().st_size > 0)
print(f"[extract] DONE: {have}/{len(wanted)} present "
      f"(+{n_ok} this run, {n_err} errors, scanned {n_seen:,}, {time.time()-t0:.0f}s)",
      flush=True)
sys.exit(0 if have == len(wanted) else 1)
