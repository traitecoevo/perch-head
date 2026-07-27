"""Score audio with Perch's NATIVE eBird head -> a score CSV the assembly-pipeline cutters read.

Sibling of predict.py, which runs a *trained* head. This one needs no head at all: Perch's
`serving_default` exposes 14795 eBird-2021 logits directly, so a species can be mined
before any custom head knows about it -- useful for a class being rebuilt, and as a second
opinion independent of whichever recognizer mined the class in the first place.

Output is the `predict.py` schema (`file,start_s,end_s,scientific,common,score`), which
`Training_library_assembly_pipeline/species/clip_common.py` sniffs. Pair it with
`cut_topup.py`, NOT `cut_stratified.py`: these are raw logits on an arbitrary per-recording
scale, and equal-width confidence bands over that are meaningless.

Windows are Perch-native 5 s. `--hop` under 5 gives overlapping windows so a call is
localised to better than one window length -- the cutter centres its clip on the window
midpoint, so a coarse hop puts the call off-centre.

Codes are eBird 2021 and lag current taxonomy (Australian Pipit is `auspip1` here,
`auspip3` now). `--scientific` is what gets written to the CSV, so it can carry the
current name regardless of what the checkpoint calls it.

Usage:
  .venv/bin/python scripts/predict_ebird.py --audio DIR --code ausgre1 \
      --scientific "Tachybaptus novaehollandiae" --common "Australasian Grebe" \
      --hop 1.0 --out scores.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


def iter_audio(root: str):
    if os.path.isfile(root):
        yield root
        return
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if n.lower().endswith(AUDIO_EXT) and not n.startswith("."):
                yield os.path.join(dirpath, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="audio file or folder (walked recursively).")
    ap.add_argument("--code", required=True, help="eBird-2021 code, e.g. ausgre1.")
    ap.add_argument("--scientific", required=True, help="scientific name to WRITE (current taxonomy).")
    ap.add_argument("--common", default="", help="common name to write.")
    ap.add_argument("--out", required=True, help="output score CSV.")
    ap.add_argument("--hop", type=float, default=1.0,
                    help="seconds between window starts; <5 overlaps (default 1.0).")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    import tensorflow as tf
    from perch_head.audio import open_audio_file
    from perch_head.embed import SAMPLE_RATE, WINDOW_SAMPLES, default_checkpoint_path

    ckpt = args.checkpoint or default_checkpoint_path()
    with open(os.path.join(ckpt, "assets", "perch_v2_ebird_classes.csv")) as fh:
        codes = [line.strip() for line in fh][1:]
    if args.code not in codes:
        raise SystemExit(f"{args.code!r} is not a Perch eBird-2021 class")
    target = codes.index(args.code)

    serving = tf.saved_model.load(ckpt).signatures["serving_default"]
    hop_samples = max(1, int(args.hop * SAMPLE_RATE))

    rows = []
    for path in iter_audio(args.audio):
        try:
            sig, _ = open_audio_file(path)
        except Exception as exc:
            print(f"  !! {os.path.basename(path)}: {exc}")
            continue
        # a clip shorter than one window is zero-padded rather than skipped: short
        # Macaulay/XC cuts are often the entire vocalisation
        if len(sig) < WINDOW_SAMPLES:
            sig = np.pad(sig, (0, WINDOW_SAMPLES - len(sig)))
        starts = list(range(0, len(sig) - WINDOW_SAMPLES + 1, hop_samples))
        batch = np.stack([sig[s:s + WINDOW_SAMPLES] for s in starts]).astype("float32")
        logits = serving(inputs=tf.constant(batch))["label"].numpy()[:, target]
        for s, lg in zip(starts, logits):
            t = s / SAMPLE_RATE
            rows.append({
                "file": path,
                "start_s": round(t, 3),
                "end_s": round(t + WINDOW_SAMPLES / SAMPLE_RATE, 3),
                "scientific": args.scientific,
                "common": args.common,
                "score": round(float(lg), 4),
            })
        print(f"  {os.path.basename(path):44s} {len(starts):4d} windows  "
              f"peak {logits.max():6.2f}  median {np.median(logits):6.2f}")

    if not rows:
        raise SystemExit("no windows scored")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
