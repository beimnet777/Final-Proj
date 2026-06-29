"""Unified launcher for MSP and LibriSpeech disentanglement experiments.

This module deliberately delegates to the existing trainers.  It owns only the
reproducible preset, Colab runtime and artifact contract.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .experiment_presets import LIBRI_EXPERIMENTS, MSP_EXPERIMENTS, PRESETS, resolve_preset
from .training_runtime import atomic_json_dump, dataset_fingerprint, mirror_file, resolve_microbatch

ROOT = Path(__file__).resolve().parents[1]
DIS = Path(__file__).resolve().parent


def _find_first(candidates: list[Path], what: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"could not find {what}; checked: {', '.join(map(str, candidates))}")


def _flag(name: str) -> str:
    return "--" + name


def _libri_args(config: dict) -> list[str]:
    bool_optional = {"hard_gumbel_routing", "club_enabled", "vicreg_full"}
    args: list[str] = []
    for key, value in config.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(_flag(key))
            elif key in bool_optional:
                args.append("--no-" + key)
        else:
            args.extend([_flag(key), str(value)])
    return args


def _msp_args(config: dict) -> list[str]:
    args: list[str] = []
    for key, value in config.items():
        if key == "pcgrad":
            if not value: args.append("--no_pcgrad")
        elif key == "hard_routing":
            if not value: args.append("--soft_routing")
        elif key == "invariance":
            if not value: args.append("--no_invariance")
        elif isinstance(value, bool):
            if value: args.append(_flag(key))
        elif value is not None:
            args.extend([_flag(key), str(value)])
    return args


def _resolve_data(experiment: str, root: Path) -> tuple[dict, list[Path]]:
    if experiment in MSP_EXPERIMENTS:
        manifest = _find_first([
            root / "manifest.csv", root / "msp_subset" / "manifest.csv",
            root / "data" / "msp_subset" / "manifest.csv",
        ], "MSP manifest.csv")
        transcripts = _find_first([
            root / "Transcripts.zip", root / "transcripts.zip",
            root / "Transcripts", root / "transcripts",
        ], "MSP transcripts")
        fp = [manifest, transcripts]
        if (root / "bundle_manifest.json").exists(): fp.append(root / "bundle_manifest.json")
        return {
            "manifest": str(manifest), "audio_root": str(root),
            "transcripts": str(transcripts),
        }, fp
    libri = root / "LibriSpeech" if (root / "LibriSpeech").exists() else root
    train = libri / "train-clean-100"
    if not train.exists():
        raise FileNotFoundError(f"LibriSpeech train-clean-100 not found under {libri}")
    lexicon = _find_first([
        root / "librispeech-lexicon.txt", ROOT / "Probing" / "data" / "librispeech-lexicon.txt",
    ], "LibriSpeech lexicon")
    fp = [train, lexicon]
    if (root / "bundle_manifest.json").exists(): fp.append(root / "bundle_manifest.json")
    return {"librispeech_root": str(libri), "lexicon_path": str(lexicon)}, fp


def _probe_commands(a, checkpoint: Path) -> list[list[str]]:
    commands = []
    probe_dir = a.output_dir.resolve() / "probes"; probe_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in a.probe_seeds.split(",") if x.strip()]
    preset = resolve_preset(a.experiment, a.profile)
    libri = a.data_root.resolve() / "LibriSpeech"
    if not libri.exists(): libri = a.data_root.resolve()
    common = ["--n_routes", str(preset.get("n_routes", 2)), "--spear_layernorm",
              "--local_data", "--librispeech_root", str(libri),
              "--gumbel_tau_end", str(preset.get("gumbel_tau_end", 0.1))]
    common.append("--hard_gumbel_routing" if preset.get("hard_gumbel_routing", True)
                  else "--no-hard_gumbel_routing")
    for seed in seeds:
        for sid_arch in ("linear", "stats", "mlp"):
            run_name = f"probe_{a.experiment}_{a.probe_dataset}_sid_{sid_arch}_s{seed}"
            commands.append([
                sys.executable, "diag_probe/run.py", "--stage2_ckpt", str(checkpoint),
                "--stage1_ckpt", str(checkpoint), "--run_name", run_name,
                "--output_json", str(probe_dir / f"{run_name}.json"),
                "--sources", "z_L,z_P,z_t", "--tasks", "sid",
                "--sid_probe_arch", sid_arch, "--sid_dataset", a.probe_dataset,
                "--probe_steps", str(a.probe_steps), "--probe_patience", "0",
                "--seed", str(seed), *common,
            ])
        for pr_arch in ("linear", "mlp"):
            run_name = f"probe_{a.experiment}_pr_{pr_arch}_s{seed}"
            commands.append([
                sys.executable, "diag_probe/run.py", "--stage2_ckpt", str(checkpoint),
                "--stage1_ckpt", str(checkpoint), "--run_name", run_name,
                "--output_json", str(probe_dir / f"{run_name}.json"),
                "--sources", "z_L,z_P,z_t", "--tasks", "pr",
                "--pr_probe_arch", pr_arch, "--probe_steps", str(a.probe_steps),
                "--probe_patience", "0", "--seed", str(seed), *common,
            ])
    return commands


def _run_probes(a, output: Path) -> int:
    checkpoint = Path(a.probe_checkpoint) if a.probe_checkpoint else _find_first([
        output / "final.pt", output / "stage2_final.pt", output / "stage2_best.pt",
        output / "best.pt",
    ], "completed training checkpoint")
    status_path = output / "probe_status.json"
    if a.drive_mirror and (a.drive_mirror / "probe_status.json").exists() and not status_path.exists():
        status_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.drive_mirror / "probe_status.json", status_path)
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for command in _probe_commands(a, checkpoint):
        key = command[command.index("--run_name") + 1]
        if status.get(key) == "complete":
            print(f"[runner] probe already complete: {key}")
            continue
        print("[runner]", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=DIS)
        if result.returncode:
            status[key] = f"failed:{result.returncode}"
            atomic_json_dump(status, status_path)
            if a.drive_mirror: mirror_file(status_path, a.drive_mirror)
            return result.returncode
        status[key] = "complete"
        atomic_json_dump(status, status_path)
        if a.drive_mirror:
            mirror_file(status_path, a.drive_mirror)
            result_path = output / "probes" / f"{key}.json"
            if result_path.exists(): mirror_file(result_path, a.drive_mirror / "probes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=sorted(PRESETS), required=True)
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    p.add_argument("--phase", choices=("train", "probe"), default="train")
    p.add_argument("--effective_batch_size", type=int, default=16)
    p.add_argument("--microbatch_size", default="auto")
    p.add_argument("--resume", default="auto")
    p.add_argument("--segment_steps", type=int, default=250)
    p.add_argument("--max_runtime_minutes", type=float, default=600)
    p.add_argument("--resume_every", type=int, default=50)
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--drive_mirror", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--probe_checkpoint")
    p.add_argument("--probe_dataset", choices=("libri", "arctic"), default="libri")
    p.add_argument("--probe_seeds", default="42,43,44")
    p.add_argument("--probe_steps", type=int, default=10000)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    output = a.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    if a.phase == "probe":
        if a.experiment not in LIBRI_EXPERIMENTS:
            raise ValueError("the unified diagnostic probe phase currently supports LibriSpeech presets")
        return _run_probes(a, output)

    data_args, fp_paths = _resolve_data(a.experiment, a.data_root.resolve())
    dataset_hash = dataset_fingerprint(fp_paths)
    micro, accumulation = resolve_microbatch(a.microbatch_size, a.effective_batch_size)
    config = resolve_preset(a.experiment, a.profile)
    if a.experiment in LIBRI_EXPERIMENTS and config.get("dual_invariance") and accumulation > 1:
        for key in ("pairs_alpha_per_step", "pairs_beta_per_step"):
            total_pairs = int(config[key])
            if total_pairs % accumulation:
                raise ValueError(
                    f"{key}={total_pairs} must divide gradient accumulation={accumulation} "
                    "to preserve the preset's effective pair budget")
            config[key] = total_pairs // accumulation
    config.update(data_args)
    if a.experiment in LIBRI_EXPERIMENTS:
        config["speaker_stratified_holdout"] = True
    config.update(seed=a.seed, batch_size=micro, num_workers=0,
                  checkpoint_dir=str(output))
    runtime = {
        "experiment": a.experiment, "profile": a.profile,
        "dataset_fingerprint": dataset_hash, "effective_batch_size": a.effective_batch_size,
        "microbatch_size": micro, "gradient_accumulation_steps": accumulation,
        "resume": a.resume, "segment_steps": a.segment_steps,
        "max_runtime_minutes": a.max_runtime_minutes, "resume_every": a.resume_every,
        "precision": a.precision, "drive_mirror": str(a.drive_mirror) if a.drive_mirror else "",
    }
    atomic_json_dump({**config, **runtime}, output / "resolved_config.yaml")
    atomic_json_dump(runtime, output / "manifest.json")
    if a.drive_mirror:
        mirror_file(output / "resolved_config.yaml", a.drive_mirror)
        mirror_file(output / "manifest.json", a.drive_mirror)

    common = {
        "resume": a.resume, "segment_steps": a.segment_steps,
        "max_runtime_minutes": a.max_runtime_minutes, "resume_every": a.resume_every,
        "gradient_accumulation_steps": accumulation, "precision": a.precision,
        "dataset_fingerprint": dataset_hash, "experiment_preset": a.experiment,
        "drive_mirror": str(a.drive_mirror) if a.drive_mirror else "",
    }
    if a.experiment in MSP_EXPERIMENTS:
        config.update(common)
        command = [sys.executable, "-m", "msp.run", *_msp_args(config)]
    else:
        config.update(common, runs_dir=str(output / "runs"), log_dir=str(output / "logs"))
        command = [sys.executable, "run.py", *_libri_args(config)]
    print(json.dumps({"command": command, "runtime": runtime}, indent=2), flush=True)
    if a.dry_run:
        return 0
    env = os.environ.copy(); env.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.run(command, cwd=DIS, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
