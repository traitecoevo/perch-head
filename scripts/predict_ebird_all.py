"""Score audio with Perch's own classifier head, keeping ALL 14795 logits.

Sibling of `predict_ebird.py`, which takes `--code` and keeps one species. That is the right
tool when you know what you are mining. This one is for the other case: **one pass over field
audio that many classes can be mined from afterwards**, without paying for inference again.

Output is one `.npz` per input recording, holding the full window x class logit matrix in
float16 (the logits span roughly -10..+15, so float16's ~3 decimal digits is far finer than
anything downstream cares about, and it halves a 767 MB pass to 384 MB):

    scores              (n_windows, 14795) float16
    starts              (n_windows,)       float64   window start, seconds into the file
    columns_code        (14795,)           <U…       eBird-2021 codes, checkpoint order;
                                                     `no_ebird_code` for 5089 of them
    columns_scientific  (14795,)           <U…       class names from assets/labels.csv
    meta                ()                 <U…       json: file, sr, channels, hop, duration…

**One npz per recording, and existing ones are skipped**, so the run is resumable. That is not
a nicety: the field audio lives on an SMB mount that drops mid-run, and a single combined
output means a drop at hour 17 loses everything.

`--stage-dir` copies each recording to local disk before decoding for the same reason — a
dropped mount mid-decode is a corrupt read, not an error. The staged copy is deleted after.

Usage:
  .venv/bin/python scripts/predict_ebird_all.py \
      --file-list selection.txt --out-dir scores/ --hop 2.5 --stage-dir /tmp/stage
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np


def probe(path):
    """(sample_rate, channels, duration) via ffprobe; (None, None, None) if unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=120).stdout.split()
        vals = [v for v in ",".join(out).split(",") if v]
        return int(vals[0]), int(vals[1]), float(vals[2])
    except Exception:
        return None, None, None


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", help="file or folder (walked recursively).")
    src.add_argument("--file-list", help="text file, one audio path per line.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hop", type=float, default=2.5,
                    help="seconds between window starts; <5 overlaps (default 2.5).")
    ap.add_argument("--stage-dir", default=None,
                    help="copy each file here before decoding (use for network mounts).")
    ap.add_argument("--batch", type=int, default=64, help="windows per forward pass.")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files (0 = all).")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    import tensorflow as tf
    from perch_head.audio import AUDIO_EXT, open_audio_file
    from perch_head.embed import SAMPLE_RATE, WINDOW_SAMPLES, default_checkpoint_path

    if args.file_list:
        paths = [l.strip() for l in open(args.file_list) if l.strip()]
    elif os.path.isfile(args.audio):
        paths = [args.audio]
    else:
        paths = [os.path.join(d, n)
                 for d, _, ns in os.walk(args.audio) for n in sorted(ns)
                 if n.lower().endswith(AUDIO_EXT) and not n.startswith(".")]
    if args.limit:
        paths = paths[:args.limit]

    ckpt = args.checkpoint or default_checkpoint_path()
    with open(os.path.join(ckpt, "assets", "perch_v2_ebird_classes.csv")) as fh:
        codes = np.array([l.strip() for l in fh][1:])
    # labels.csv is one bare column whose header is the NAME OF THE LABEL SPACE, not a
    # column description: in perch_v2 it reads `inat2024_fsd50k`. This used to sniff the
    # header for "scientific"/"name"/"label", find none of them, and silently fall through
    # to a column of empty strings -- so every npz written before 2026-08-27 has an unusable
    # `columns_scientific` and has to be re-joined to labels.csv by row index. Take column 0
    # and check the length instead; the row order is the checkpoint's logit order, which is
    # what makes the positional join to `codes` valid in the first place.
    lab = os.path.join(ckpt, "assets", "labels.csv")
    rows = [line.strip() for line in open(lab)][1:]
    if len(rows) != len(codes):
        raise SystemExit(
            f"{lab} has {len(rows)} classes but perch_v2_ebird_classes.csv has {len(codes)}; "
            "the two assets must be row-aligned to the same logit order.")
    sci = np.array(rows)

    serving = tf.saved_model.load(ckpt).signatures["serving_default"]
    os.makedirs(args.out_dir, exist_ok=True)
    if args.stage_dir:
        os.makedirs(args.stage_dir, exist_ok=True)

    hop_samples = max(1, int(args.hop * SAMPLE_RATE))
    t_all = time.time()
    done = skipped = failed = 0
    audio_s = 0.0

    for i, path in enumerate(paths, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_dir, stem + ".npz")
        if os.path.exists(out):
            print(f"[{i}/{len(paths)}] skip (exists) {stem}", flush=True)
            skipped += 1
            continue

        sr, ch, dur = probe(path)
        local = path
        t0 = time.time()
        if args.stage_dir:
            local = os.path.join(args.stage_dir, os.path.basename(path))
            try:
                shutil.copy2(path, local)
            except Exception as exc:
                print(f"[{i}/{len(paths)}] !! stage failed {stem}: {exc}", flush=True)
                failed += 1
                continue
        t_stage = time.time() - t0

        try:
            t0 = time.time()
            sig, _ = open_audio_file(local)
            t_dec = time.time() - t0
            if len(sig) < WINDOW_SAMPLES:
                sig = np.pad(sig, (0, WINDOW_SAMPLES - len(sig)))
            starts = list(range(0, len(sig) - WINDOW_SAMPLES + 1, hop_samples))

            t0 = time.time()
            chunks = []
            for b in range(0, len(starts), args.batch):
                win = np.stack([sig[s:s + WINDOW_SAMPLES]
                                for s in starts[b:b + args.batch]]).astype("float32")
                chunks.append(serving(inputs=tf.constant(win))["label"].numpy().astype("float16"))
            scores = np.concatenate(chunks, 0) if chunks else np.zeros((0, len(codes)), "float16")
            t_inf = time.time() - t0

            np.savez_compressed(
                out, scores=scores,
                starts=np.array([s / SAMPLE_RATE for s in starts], dtype="float64"),
                columns_code=codes, columns_scientific=sci,
                meta=json.dumps({"file": path, "source_sample_rate": sr, "channels": ch,
                                 "duration_s": dur, "hop_s": args.hop,
                                 "window_s": WINDOW_SAMPLES / SAMPLE_RATE,
                                 "model": "perch_v2_native_head", "checkpoint": ckpt}))
            done += 1
            audio_s += dur or 0.0
            rate = (dur / (t_dec + t_inf)) if (dur and t_dec + t_inf) else float("nan")
            print(f"[{i}/{len(paths)}] {stem:38} {len(starts):5d} win  "
                  f"{sr}Hz/{ch}ch  stage {t_stage:5.1f}s  dec {t_dec:5.1f}s  inf {t_inf:6.1f}s  "
                  f"{rate:5.1f}x RT  -> {os.path.getsize(out)/1e6:.0f} MB", flush=True)
        except Exception as exc:
            print(f"[{i}/{len(paths)}] !! {stem}: {exc}", flush=True)
            failed += 1
        finally:
            if args.stage_dir and local != path and os.path.exists(local):
                os.remove(local)

    el = time.time() - t_all
    print(f"\ndone={done} skipped={skipped} failed={failed}  audio={audio_s/3600:.2f} h  "
          f"elapsed={el/60:.1f} min  overall {audio_s/el if el else 0:.1f}x real-time")


if __name__ == "__main__":
    main()
