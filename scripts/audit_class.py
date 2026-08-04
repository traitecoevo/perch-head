"""Second-opinion a labeled audio class with Perch's NATIVE eBird head -> per-clip CSV.

No trained head involved: Perch's `serving_default` returns `label` (14795 eBird-2021
logits) alongside `embedding`, so a class assembled elsewhere can be audited without
this repo knowing anything about that library's label vocabulary.

Read the result ASYMMETRICALLY. "this is actually <common species>" is evidence;
"yes, it's the target" is NOT clearance -- Perch's reliability on a species tracks that
species' recording supply, so a thinly-recorded target has a weak head here. Two further
cautions: the class list is eBird 2021, so codes lag current taxonomy (Australian Pipit
is `auspip1` here, `auspip3` now); and a high target logit does not mean the target is
the loudest thing in the window, which is why this reports RANK and MARGIN rather than
the target score alone.

Results group by parent recording, because contamination arrives one whole recording at
a time -- a recording whose clips are consistently outranked by the same other species is
the unit worth quarantining, not the individual clip.

Usage:
  .venv/bin/python scripts/audit_class.py \
      --class-dir "/path/to/reallybig/Tachybaptus novaehollandiae_Australasian Grebe" \
      --code ausgre1 --out grebe_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import Counter, defaultdict

import numpy as np

from perch_head.audio import AUDIO_EXT

# '<score>_<rank>_' mining prefix and the trailing '_<start>s_<end>s' window offsets both
# have to come off before two clips of one parent recording group together.
PREFIX = re.compile(r"^[0-9.]+_\d+_")
OFFSETS = re.compile(r"_\d+(?:\.\d+)?s_\d+(?:\.\d+)?s.*$")


def parent_recording(filename: str) -> str:
    """The recording a clip was cut from, for both clip-naming schemes in use.

    The cutters (`cut_stratified.py` / `cut_topup.py`) write
    '<batch>_<idx>__<recording>__t<offset>__<scorer>', where the recording is
    already delimited; older mined clips are '<score>_<rank>_<recording>_<a>s_<b>s'.
    Stripping only the old decoration leaves a new-scheme name untouched, which
    makes every clip its own singleton group -- and the by-recording report,
    which exists because contamination arrives one whole recording at a time,
    silently stops telling you anything.
    """
    stem = os.path.splitext(filename)[0]
    if "__" in stem:
        return stem.split("__")[1]
    return OFFSETS.sub("", PREFIX.sub("", stem))


def centre_window(sig: np.ndarray, n: int) -> np.ndarray:
    """One `n`-sample window centred on the clip.

    Deliberately NOT `windows_for_file`: these are 3 s or 5 s clips already cut around a
    detection, so the call sits in the middle. End-aligned padding would push a 3 s call
    off-centre and score a window that is 40% silence.
    """
    if len(sig) < n:
        pad = n - len(sig)
        return np.pad(sig, (pad // 2, pad - pad // 2))
    start = (len(sig) - n) // 2
    return sig[start:start + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-dir", required=True, help="folder of clips for ONE class.")
    ap.add_argument("--code", default=None,
                    help="eBird-2021 code for the target, e.g. ausgre1 (see the "
                         "checkpoint's assets/perch_v2_ebird_classes.csv).")
    ap.add_argument("--label", default=None,
                    help="scientific name from assets/labels.csv, e.g. 'Capra hircus'. "
                         "Use this for the ~5100 classes that carry no eBird code -- the "
                         "mammals, frogs and insects Perch also scores. A non-bird class "
                         "cannot be named with --code.")
    ap.add_argument("--out", required=True, help="output per-clip audit CSV.")
    ap.add_argument("--checkpoint", default=None, help="Perch checkpoint (default: kagglehub cache).")
    ap.add_argument("--suspect-rank", type=int, default=20,
                    help="a recording whose MEDIAN target rank is worse than this is flagged.")
    ap.add_argument("--limit", type=int, default=0, help="audit only the first N clips (smoke test).")
    args = ap.parse_args()

    import tensorflow as tf
    from perch_head.audio import open_audio_file
    from perch_head.embed import WINDOW_SAMPLES, default_checkpoint_path

    if bool(args.code) == bool(args.label):
        raise SystemExit("give exactly one of --code (eBird) or --label (scientific name)")

    ckpt = args.checkpoint or default_checkpoint_path()
    classes_csv = os.path.join(ckpt, "assets", "perch_v2_ebird_classes.csv")
    labels_csv = os.path.join(ckpt, "assets", "labels.csv")
    with open(classes_csv) as fh:
        codes = [line.strip() for line in fh][1:]   # first line is the 'ebird2021' header
    # Parallel to `codes`, but naming every class rather than only the birds: the
    # eBird column is 'no_ebird_code' for ~5100 of them.
    with open(labels_csv) as fh:
        names = [line.strip() for line in fh][1:]

    if args.code:
        if args.code not in codes:
            raise SystemExit(f"{args.code!r} is not in the Perch eBird-2021 class list ({classes_csv})")
        target = codes.index(args.code)
    else:
        if args.label not in names:
            raise SystemExit(f"{args.label!r} is not in the Perch class list ({labels_csv})")
        target = names.index(args.label)
    print(f"target: {names[target]} ({codes[target]})  index {target}")

    serving = tf.saved_model.load(ckpt).signatures["serving_default"]

    files = sorted(f for f in os.listdir(args.class_dir) if f.lower().endswith(AUDIO_EXT))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit(f"no audio in {args.class_dir}")

    rows = []
    for i, fname in enumerate(files, 1):
        try:
            sig, _ = open_audio_file(os.path.join(args.class_dir, fname))
        except Exception as exc:                      # a single unreadable clip must not
            print(f"  !! skipped {fname}: {exc}")     # abort an audit of hundreds
            continue
        window = centre_window(sig, WINDOW_SAMPLES)
        logits = serving(inputs=tf.constant(window[None, :], dtype=tf.float32))["label"].numpy()[0]
        order = np.argsort(logits)[::-1]
        rows.append({
            "file": fname,
            "recording": parent_recording(fname),
            "target_logit": round(float(logits[target]), 3),
            "target_rank": int(np.where(order == target)[0][0]) + 1,
            "top1": codes[order[0]],
            "top1_name": names[order[0]],
            "top1_logit": round(float(logits[order[0]]), 3),
            "top2": codes[order[1]],
            "top2_name": names[order[1]],
            "margin": round(float(logits[target] - logits[order[0]]), 3),
        })
        if i % 25 == 0:
            print(f"  ... {i}/{len(files)}", flush=True)

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ranks = np.array([r["target_rank"] for r in rows])
    print(f"\n{os.path.basename(args.class_dir.rstrip('/'))}  ({names[target]})  n={len(rows)}")
    print(f"  target is Perch top-1: {(ranks == 1).sum()}   top-5: {(ranks <= 5).sum()}   "
          f"rank>{args.suspect_rank}: {(ranks > args.suspect_rank).sum()}")

    by_recording = defaultdict(list)
    for r in rows:
        by_recording[r["recording"]].append(r)

    print("\n  by parent recording (worst median rank first):")
    for rec, rs in sorted(by_recording.items(),
                          key=lambda kv: -np.median([x["target_rank"] for x in kv[1]])):
        med = np.median([x["target_rank"] for x in rs])
        outranked = Counter(x["top1_name"] for x in rs if x["target_rank"] > 1)
        flag = "  <-- SUSPECT" if med > args.suspect_rank else ""
        print(f"    {rec[:46]:48s} n={len(rs):3d} median_rank={med:6.0f} "
              f"median_logit={np.median([x['target_logit'] for x in rs]):6.2f} "
              f"outranked_by={', '.join(f'{k}:{v}' for k, v in outranked.most_common(3)) or '-'}{flag}")

    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
