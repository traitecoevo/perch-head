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

MEMORY, and the failure that cost 2.6 h on 2026-08-12. This script used to stack *every
window of a file* into a single model call. That is fine for XC-length clips and lethal on
an archive pool: a 504 s Macaulay asset at `--hop 1.0` is a 500-window call, and the
activations for that on a 16 GB machine drove swap to 16.5 of 17.4 GB. Throughput collapsed
to ~0.7x real-time -- roughly 25x slower than the same backbone's documented 17.8-20.6x on
the Fowlers and Wild Deserts passes -- and the tell was pathological, because the process
looked healthy the whole time: 400%+ CPU, RSS climbing, no error. Watch `sysctl
vm.swapusage`, not %CPU. Windows are now chunked at `--batch` (64, as in predict.py), which
makes peak memory independent of how long the longest recording is.

`--hop` costs time linearly, so pick it for the job: 1.0 to localise a call tightly inside a
known-good recording, 2.5 to mine a large pool for its best window per recording (what the
Fowlers/Wild Deserts passes used). At 2.5 the 5 s windows still overlap 50%, so nothing
shorter than 2.5 s can straddle two windows and be missed by both.

Codes are eBird 2021 and lag current taxonomy (Australian Pipit is `auspip1` here, and
`auspip2` now -- NOT `auspip3`, which is New Zealand Pipit, a different bird).
`--scientific` is what gets written to the CSV, so it can carry the current name regardless
of what the checkpoint calls it.

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

from perch_head.audio import AUDIO_EXT


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
    ap.add_argument("--code", help="eBird-2021 code, e.g. ausgre1.")
    ap.add_argument("--label", help="Perch class by SCIENTIFIC NAME, e.g. 'Psephotellus varius'. "
                    "Use when the species has no eBird code in the checkpoint: Perch carries "
                    "14795 classes but `perch_v2_ebird_classes.csv` says `no_ebird_code` for "
                    "some of them (Mulga Parrot, row 11239), so a code-only lookup makes a "
                    "species unscoreable that the model actually has a head for.")
    ap.add_argument("--scientific", required=True, help="scientific name to WRITE (current taxonomy).")
    ap.add_argument("--common", default="", help="common name to write.")
    ap.add_argument("--out", required=True, help="output score CSV.")
    ap.add_argument("--hop", type=float, default=1.0,
                    help="seconds between window starts; <5 overlaps (default 1.0).")
    ap.add_argument("--batch", type=int, default=64,
                    help="windows per model call (default 64, as predict.py). See the "
                         "note in the module docstring -- this is a memory bound, not a "
                         "speed knob, and raising it is how the script used to thrash.")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    import tensorflow as tf
    from perch_head.audio import open_audio_file
    from perch_head.embed import SAMPLE_RATE, WINDOW_SAMPLES, default_checkpoint_path

    ckpt = args.checkpoint or default_checkpoint_path()
    if not args.code and not args.label:
        raise SystemExit("pass --code or --label")
    if args.label:
        with open(os.path.join(ckpt, "assets", "labels.csv")) as fh:
            labels = [line.strip() for line in fh][1:]
        if args.label not in labels:
            near = [l for l in labels if l.split()[0] == args.label.split()[0]][:5]
            raise SystemExit(f"{args.label!r} is not a Perch class. Near: {near}")
        target = labels.index(args.label)
    else:
        with open(os.path.join(ckpt, "assets", "perch_v2_ebird_classes.csv")) as fh:
            codes = [line.strip() for line in fh][1:]
        if args.code not in codes:
            raise SystemExit(f"{args.code!r} is not a Perch eBird-2021 class")
        target = codes.index(args.code)

    serving = tf.saved_model.load(ckpt).signatures["serving_default"]
    hop_samples = max(1, int(args.hop * SAMPLE_RATE))

    n_rows = 0
    fields = ["file", "start_s", "end_s", "scientific", "common", "score"]
    # Stream to the CSV instead of accumulating every row: a long pool is hours of work,
    # and a run that only writes at the end loses all of it to one kill or one OOM.
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for path in iter_audio(args.audio):
            try:
                sig, _ = open_audio_file(path)
            except Exception as exc:
                print(f"  !! {os.path.basename(path)}: {exc}", flush=True)
                continue
            # a clip shorter than one window is zero-padded rather than skipped: short
            # Macaulay/XC cuts are often the entire vocalisation
            if len(sig) < WINDOW_SAMPLES:
                sig = np.pad(sig, (0, WINDOW_SAMPLES - len(sig)))
            starts = list(range(0, len(sig) - WINDOW_SAMPLES + 1, hop_samples))
            logits = np.empty(len(starts), dtype="float32")
            for i in range(0, len(starts), args.batch):
                chunk = starts[i:i + args.batch]
                block = np.stack([sig[s:s + WINDOW_SAMPLES]
                                  for s in chunk]).astype("float32")
                logits[i:i + len(chunk)] = (
                    serving(inputs=tf.constant(block))["label"].numpy()[:, target])
            for s, lg in zip(starts, logits):
                t = s / SAMPLE_RATE
                w.writerow({
                    "file": path,
                    "start_s": round(t, 3),
                    "end_s": round(t + WINDOW_SAMPLES / SAMPLE_RATE, 3),
                    "scientific": args.scientific,
                    "common": args.common,
                    "score": round(float(lg), 4),
                })
                n_rows += 1
            fh.flush()
            print(f"  {os.path.basename(path):44s} {len(starts):4d} windows  "
                  f"peak {logits.max():6.2f}  median {np.median(logits):6.2f}", flush=True)

    if not n_rows:
        raise SystemExit("no windows scored")
    print(f"\nwrote {n_rows} rows -> {args.out}")


if __name__ == "__main__":
    main()
