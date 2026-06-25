#!/usr/bin/env python3
"""Build a small, speaker/emotion-decorrelated subset of MSP-Podcast 2.0.

Why a subset:  the full corpus is 264,705 turns / ~2,500 train speakers / 409 h.
We don't need that to break IEMOCAP's 10-actor speaker<->emotion confound — a few
hundred speakers, each carrying several emotions, decorrelates speaker and emotion
while keeping runs fast and the SID head small.

What it does:
  * reads Labels/labels_consensus.csv  (FileName,EmoClass,EmoAct,EmoVal,EmoDom,
    SpkrID,Gender,Split_Set) — this single file already ties emotion+speaker+split.
  * keeps Train rows with a KNOWN speaker and an EmoClass in --emotions (default
    big-4: N/H/S/A, mapped to the IEMOCAP index order neutral/happy/sad/angry).
  * keeps only speakers with enough utts AND enough distinct emotions (so the
    speaker's own emotion varies — that is the decorrelation we want).
  * ranks those speakers by emotion balance, takes --n_speakers of them, and caps
    each speaker's utts (balanced across its emotions) so no speaker dominates.
  * holds out --val_frac of each train speaker's utts as a closed-set val set, and
    optionally draws --test_speakers UNSEEN speakers from the Dev split for a
    speaker-independent emotion test (the claim IEMOCAP cannot support).

Outputs (metadata only — no audio is touched):
  * <out>/manifest.csv   one row per utterance with the final split + indices
  * <out>/members.txt     audio members for:  tar -xzf Audios.tar.gz -T members.txt
  * <out>/speakers.csv    speaker_idx -> original SpkrID + per-speaker counts

PR targets are NOT built here.  Build them at load time from the human transcript
via the existing data.dataset.text_to_phones() (lexicon + g2p -> SUPERB ARPABET),
NOT from the ForceAligned phones tier — that tier mixes ARPABET and IPA across
files and would need a lossy remap.  CTC is alignment-free, so the transcript is
all you need.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import zipfile
from collections import defaultdict
from pathlib import Path

# MSP primary-emotion code -> (name, index).  Index order matches IEMOCAP's
# EMOTION_NAMES = (neutral, happy_excited, sad, angry) so the emotion head and any
# IEMOCAP-trained baseline stay comparable.
EMO_CODE = {"N": ("neutral", 0), "H": ("happy", 1), "S": ("sad", 2), "A": ("angry", 3)}


def _read_consensus(labels_arg: Path):
    """Yield dict rows from labels_consensus.csv, whether given the CSV directly,
    its parent dir, or the Labels.zip."""
    if labels_arg.suffix == ".zip":
        with zipfile.ZipFile(labels_arg) as z:
            name = next(n for n in z.namelist() if n.endswith("labels_consensus.csv"))
            text = io.TextIOWrapper(z.open(name), encoding="utf-8")
            yield from csv.DictReader(text)
        return
    csv_path = labels_arg
    if labels_arg.is_dir():
        csv_path = next(labels_arg.rglob("labels_consensus.csv"))
    with open(csv_path, newline="") as f:
        yield from csv.DictReader(f)


def _known(spk: str) -> bool:
    return spk not in ("", "Unknown", "unknown", "NA", "nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path,
                    default=Path("/rds/project/rds-xyBFuSj0hm0/dataset/"
                                 "MSP-Podcast-2.0/Labels.zip"),
                    help="labels_consensus.csv, its dir, or Labels.zip")
    ap.add_argument("--emotions", nargs="+", default=["N", "H", "S", "A"],
                    help="primary-emotion codes to keep (default big-4)")
    ap.add_argument("--n_speakers", type=int, default=300)
    ap.add_argument("--min_utts", type=int, default=30,
                    help="min target-emotion utts for a speaker to be eligible")
    ap.add_argument("--min_emotions", type=int, default=3,
                    help="min distinct target emotions a speaker must carry")
    ap.add_argument("--cap_per_speaker", type=int, default=120,
                    help="max utts kept per speaker (balanced across its emotions)")
    ap.add_argument("--val_frac", type=float, default=0.10,
                    help="fraction of each speaker's utts held out for val")
    ap.add_argument("--test_frac", type=float, default=0.10,
                    help="fraction of each speaker's utts held out for test "
                         "(same-speaker, closed-set: PR+SID+emotion all scorable)")
    ap.add_argument("--test_speakers", type=int, default=0,
                    help="OPTIONAL # of UNSEEN speakers from Dev for a secondary "
                         "speaker-independent emotion-generalization eval "
                         "(split='test_unseen', SID not scorable there). 0=off.")
    ap.add_argument("--audio_prefix", default="Audios",
                    help="path prefix of wavs inside Audios.tar.gz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    keep = {c.upper() for c in args.emotions}
    for c in keep:
        if c not in EMO_CODE:
            ap.error(f"--emotions code {c!r} not in {sorted(EMO_CODE)}")
    rng = random.Random(args.seed)

    # ---- bucket rows by split/speaker -------------------------------------
    # by_split_spk[split][spk] = list of (FileName, emo_code, Act, Val, Dom)
    by_split_spk: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    n_rows = n_kept = 0
    for r in _read_consensus(args.labels):
        n_rows += 1
        emo = r["EmoClass"].strip().upper()
        if emo not in keep:
            continue
        spk = r["SpkrID"].strip()
        if not _known(spk):
            continue
        split = r["Split_Set"].strip()
        if split not in ("Train", "Development"):
            continue
        by_split_spk[split][spk].append(
            (r["FileName"].strip(), emo,
             r.get("EmoAct", ""), r.get("EmoVal", ""), r.get("EmoDom", "")))
        n_kept += 1
    print(f"[msp] scanned {n_rows:,} rows; {n_kept:,} in {sorted(keep)} with known speaker")

    def eligible(pool: dict[str, list]) -> list[str]:
        out = []
        for spk, rows in pool.items():
            if len(rows) < args.min_utts:
                continue
            emos = {e for _, e, *_ in rows}
            if len(emos) < args.min_emotions:
                continue
            out.append(spk)
        return out

    def select_utts(rows: list) -> list:
        """Cap a speaker's utts at cap_per_speaker, balanced across emotions
        (round-robin) so the kept set is not all-neutral."""
        by_emo: dict[str, list] = defaultdict(list)
        for row in rows:
            by_emo[row[1]].append(row)
        for v in by_emo.values():
            rng.shuffle(v)
        order = sorted(by_emo)               # deterministic emotion order
        kept, i = [], 0
        while len(kept) < min(args.cap_per_speaker, len(rows)):
            progressed = False
            for e in order:
                if i < len(by_emo[e]):
                    kept.append(by_emo[e][i]); progressed = True
                    if len(kept) >= min(args.cap_per_speaker, len(rows)):
                        break
            if not progressed:
                break
            i += 1
        return kept

    # ---- pick TRAIN speakers ---------------------------------------------
    train_elig = eligible(by_split_spk["Train"])
    # rank by emotion balance: prefer speakers whose rarest target emotion is
    # well represented (=> within-speaker emotion variation), tie-break random.
    def balance_key(spk: str):
        rows = by_split_spk["Train"][spk]
        per = defaultdict(int)
        for _, e, *_ in rows:
            per[e] += 1
        return (len(per), min(per.values()), rng.random())
    train_elig.sort(key=balance_key, reverse=True)
    chosen = train_elig[: args.n_speakers]
    print(f"[msp] train: {len(train_elig)} eligible speakers -> chose {len(chosen)}")

    def split_assign(utts: list):
        """Per-emotion stratified train/val/test on ONE speaker's kept utts, so
        every held-out split sees each emotion (when the speaker has enough)."""
        by_emo: dict[str, list] = defaultdict(list)
        for row in utts:
            by_emo[row[1]].append(row)
        out = []
        for lst in by_emo.values():
            rng.shuffle(lst)
            n = len(lst)
            n_test = int(round(n * args.test_frac))
            n_val = int(round(n * args.val_frac))
            for k, row in enumerate(lst):
                s = "test" if k < n_test else "val" if k < n_test + n_val else "train"
                out.append((row, s))
        return out

    rows_out = []   # (FileName, wav, spk_idx, spk_orig, emo_name, emo_idx, split, A,V,D)
    spk_rows = []   # (spk_idx, spk_orig, n_train, n_val, n_test, per_emotion_train)
    for idx, spk in enumerate(chosen):
        utts = select_utts(by_split_spk["Train"][spk])
        per = defaultdict(lambda: defaultdict(int))     # split -> emo -> count
        for (fn, emo, a, v, d), split in split_assign(utts):
            name, ei = EMO_CODE[emo]
            rows_out.append((fn, f"{args.audio_prefix}/{fn}", idx, spk, name, ei, split, a, v, d))
            per[split][name] += 1
        spk_rows.append((idx, spk,
                         sum(per["train"].values()), sum(per["val"].values()),
                         sum(per["test"].values()), dict(per["train"])))

    # ---- OPTIONAL secondary unseen-speaker emotion eval (split='test_unseen') --
    # Off by default.  SID is NOT scorable here (speaker_idx=-1, out of the closed
    # set); use only for a cross-speaker emotion-generalization figure.
    if args.test_speakers > 0:
        dev_elig = eligible(by_split_spk["Development"])
        rng.shuffle(dev_elig)
        test_spk = dev_elig[: args.test_speakers]
        print(f"[msp] dev: {len(dev_elig)} eligible -> {len(test_spk)} unseen speakers "
              f"(secondary emotion-generalization eval)")
        for spk in test_spk:
            for fn, emo, a, v, d in select_utts(by_split_spk["Development"][spk]):
                name, ei = EMO_CODE[emo]
                rows_out.append((fn, f"{args.audio_prefix}/{fn}", -1, spk,
                                 name, ei, "test_unseen", a, v, d))

    # ---- write outputs ----------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    man = args.out / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FileName", "wav", "speaker_idx", "spkr_id_orig",
                    "emotion", "emotion_idx", "split", "Act", "Val", "Dom"])
        w.writerows(rows_out)
    members = args.out / "members.txt"
    with open(members, "w") as f:
        f.writelines(f"{r[1]}\n" for r in rows_out)
    with open(args.out / "speakers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker_idx", "spkr_id_orig", "n_train", "n_val", "n_test",
                    "per_emotion_train"])
        for idx, spk, ntr, nval, nte, per in spk_rows:
            w.writerow([idx, spk, ntr, nval, nte,
                        ";".join(f"{k}:{v}" for k, v in sorted(per.items()))])

    # ---- summary ----------------------------------------------------------
    def counts(split):
        c = defaultdict(int)
        for r in rows_out:
            if r[6] == split:
                c[r[4]] += 1
        return dict(sorted(c.items()))
    n_tr = sum(1 for r in rows_out if r[6] == "train")
    n_va = sum(1 for r in rows_out if r[6] == "val")
    n_te = sum(1 for r in rows_out if r[6] == "test")
    n_tu = sum(1 for r in rows_out if r[6] == "test_unseen")
    print(f"\n[msp] subset written -> {args.out}")
    print(f"  speakers: {len(chosen)} (seen, closed-set)   "
          f"utts: train={n_tr}  val={n_va}  test={n_te}")
    print(f"  train emotions: {counts('train')}")
    print(f"  val   emotions: {counts('val')}")
    print(f"  test  emotions: {counts('test')}  (same speakers; PR+SID+emotion all scorable)")
    if n_tu:
        print(f"  test_unseen   : {counts('test_unseen')}  "
              f"(OPTIONAL emotion-generalization only; SID not scorable)")
    print(f"  ~audio hours (train, @5.6s/utt est.): {n_tr * 5.6 / 3600:.1f} h  "
          f"(NOT wall-clock — that's steps×step-cost)")
    print(f"\n  manifest : {man}")
    print(f"  extract  : tar -xzf <MSP>/Audios.tar.gz -T {members} -C <DEST>")


if __name__ == "__main__":
    main()
