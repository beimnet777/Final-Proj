#!/usr/bin/env python3
"""Cheap local checks before submitting Slurm jobs on Cambridge HPC.

This catches the path mistakes that waste queue/GPU time:
  * Blackwell/`/scratch` paths in Slurm scripts intended for HPC.
  * MSP train/probe calls that do not pass an explicit lexicon path.
  * Slurm shell syntax errors.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_SCRIPT_DIRS = (
    Path("Disentanglement/msp/slurm"),
    Path("Disentanglement/blackwell/slurm"),
)


def shell_syntax_ok(script: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        ["bash", "-n", str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode == 0, proc.stderr.strip()


def check_script(script: Path) -> list[str]:
    text = script.read_text()
    errors: list[str] = []

    ok, err = shell_syntax_ok(script)
    if not ok:
        errors.append(f"bash -n failed: {err}")

    if "/scratch/" in text:
        errors.append("contains /scratch/ path; HPC scripts must use /rds/user/... or repo-relative paths")

    if re.search(r"blackwell:/scratch|BLACKWELL_DATA_ROOT", text):
        errors.append("contains Blackwell-specific remote/data-root reference")

    if re.search(r"msp\\.(?:run|probe)", text):
        if "--lexicon_path" not in text:
            errors.append("MSP run/probe call without explicit --lexicon_path")
        if "Missing lexicon" not in text:
            errors.append("MSP script lacks lexicon preflight check")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Specific scripts or directories to check. Defaults to MSP and Blackwell Slurm dirs.",
    )
    args = parser.parse_args()

    roots = args.paths or list(DEFAULT_SCRIPT_DIRS)
    scripts: list[Path] = []
    for root in roots:
        if root.is_dir():
            scripts.extend(sorted(root.glob("*.sh")))
        elif root.is_file():
            scripts.append(root)
        else:
            print(f"[warn] missing path: {root}", file=sys.stderr)

    failures: dict[Path, list[str]] = {}
    for script in scripts:
        errors = check_script(script)
        if errors:
            failures[script] = errors

    if failures:
        print("[hpc_preflight] FAILED", file=sys.stderr)
        for script, errors in failures.items():
            print(f"\n{script}", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"[hpc_preflight] OK ({len(scripts)} scripts checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
