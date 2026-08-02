#!/usr/bin/env python3
"""Which SOURCE RECORDINGS in a class sit nearest the soundscape events it missed?

Mining guidance. When a class cannot be made trustworthy at any threshold, the useful
question is not "which clips are bad" but "which of the recordings this class was built
from actually resemble the field audio we are failing on" -- because those are the
recordings worth going back to and cutting more windows out of.

The ranking is entirely embedding-driven: every missed event and every clip in the class
is embedded with Perch, and each miss votes for the source recordings of its nearest
clips. Filenames enter only afterwards, to group clips by the recording they were cut
from -- provenance metadata, not a claim about content. (A class holds two naming
schemes; `recording()` handles both, and a parser that knows only the old one silently
collapses every Xeno-Canto clip onto one bogus source.)

Read the output as a ratio, not a count. A source that is nearest to many misses while
holding few clips is under-mined -- that is the buy signal. A source already contributing
80 clips and still nearest to everything is saturated: the class needs new recordings,
not more windows from that one. `n_clips` is printed next to the vote count so the two
are never confused.

CAVEAT, and it has already burned one recommendation: `votes_per_clip` is confounded by
SOURCE DURATION, which this script cannot see -- it only ever sees clips. A 5 s recording
yields exactly one window, so it scores a huge votes/clip and looks maximally under-mined
when it is in fact fully consumed. That is what happened to the top Pied Butcherbird row
(iNat 2054328, 1 clip, nearest to 12 of 22 misses): the recording is 5.07 s long and there
was nothing left to cut. Before acting on a high votes/clip row, check the source's length.
The right conclusion in that case is not "mine this harder" but "find more recordings LIKE
this one" -- which for that row meant an iNat radius search around the soundscape site.

Also prints the distance scale. If the nearest library clip to a typical miss is farther
than library clips are from each other, no amount of mining from existing sources closes
the gap and the class needs genuinely new audio.

Usage:
  .venv/bin/python scripts/nearest_sources_for_misses.py \
      --fn-csv <exp>/false_negatives.csv --model run0-3-ph \
      --soundscape-dir <dir of labeled wavs> --lib <reallybig> \
      --class "Cracticus nigrogularis_Pied Butcherbird" --species "pied butcherbird" \
      --perch-name "Cracticus nigrogularis" --out butcherbird_sources.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re

import numpy as np


def recording(fn: str) -> str:
    """Originating recording, stripped of mining/cutting decoration.

    Two schemes coexist in the library and a rebuilt class holds a mix:
      old  '<score>_<rank>_<recording>_<start>s_<end>s.wav'
      new  '<batch>_<idx>__<recording>__t<offset>__<scorer>.wav'   (cut_*.py)
    """
    stem = os.path.splitext(fn)[0]
    if "__" in stem:
        return stem.split("__")[1]
    stem = re.sub(r'^[0-9.]+_\d+_', '', stem)                    # old mining prefix
    stem = re.sub(r'^call_[0-9.]+conf_[-0-9.]+dB_', '', stem)    # old audition prefix
    return re.sub(r'_\d+(\.\d+)?s(_\d+(\.\d+)?s)?$', '', stem)   # old cut suffix


def kind(rec: str) -> str:
    """Reference vs field provenance -- they are mined in completely different ways.

    Field must be tested BEFORE iNat: a Song Meter name is `<site>_<YYYYMMDD>_<HHMMSS>`,
    which satisfies any "<digits>_<digits> at the end" test written for iNat's
    `<taxon>_<photo id>_<observation id>`. Getting that order wrong labels every field
    recording iNat and makes the class look entirely reference-sourced.
    """
    if re.search(r'(^|[_\s-])XC\d{4,}', rec):
        return "XC"
    if "RecNo" in rec or re.match(r'^ML\d+', rec) or re.match(r'^\d+\.\s', rec):
        return "ML"
    if re.match(r'^[A-Za-z0-9-]+_\d{8}_\d{6}$', rec):
        return "field"                              # <site>_<date>_<time>
    if re.match(r'^[A-Za-z]+_[a-z]+_\d{5,}_\d{4,}$', rec):
        return "iNat"                               # <genus>_<species>_<photo>_<obs>
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn-csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--soundscape-dir", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--class", dest="cls", required=True, help="reallybig class directory")
    ap.add_argument("--species", required=True, help="species value in the FN csv")
    ap.add_argument("--perch-name", required=True, help="scientific name in Perch labels.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=5, help="library neighbours each miss votes for")
    ap.add_argument("--hop", type=float, default=2.5, help="s between windows scanned in a bout")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    import tensorflow as tf
    from perch_head.audio import open_audio_file
    from perch_head.embed import WINDOW_SAMPLES, default_checkpoint_path

    ckpt = args.checkpoint or default_checkpoint_path()
    with open(os.path.join(ckpt, "assets", "labels.csv")) as fh:
        names = [l.strip() for l in fh][1:]
    if args.perch_name not in names:
        raise SystemExit(f"{args.perch_name!r} not in Perch's vocabulary")
    k_idx = names.index(args.perch_name)
    serving = tf.saved_model.load(ckpt).signatures["serving_default"]

    def run(win):
        out = serving(inputs=tf.constant(win[None, :], dtype=tf.float32))
        return (float(out["label"].numpy()[0][k_idx]),
                np.asarray(out["embedding"].numpy()[0], dtype=np.float32).ravel())

    def window(sig, start_samp):
        w = sig[start_samp:start_samp + WINDOW_SAMPLES]
        return np.pad(w, (0, WINDOW_SAMPLES - len(w))) if len(w) < WINDOW_SAMPLES else w

    # ---- missed events: take the STRONGEST window in each bout, not the midpoint.
    # Bouts run well past one window (Butcherbird: half are >5 s), so a midpoint cut
    # lands in a gap as often as on a call and makes the class look worse than it is.
    # Max-over-bout is also what the evaluation itself scores.
    fn = [r for r in csv.DictReader(open(args.fn_csv))
          if r["model"] == args.model and r["species"] == args.species]
    if not fn:
        raise SystemExit(f"no {args.species!r} rows for model {args.model!r}")
    byfile = collections.defaultdict(list)
    for r in fn:
        byfile[r["file"]].append(r)

    miss_emb, miss_meta = [], []
    for i, (rel, rs) in enumerate(sorted(byfile.items()), 1):
        p = os.path.join(args.soundscape_dir, rel)
        if not os.path.isfile(p):
            print(f"  !! missing audio: {rel}")
            continue
        sig, sr = open_audio_file(p)
        print(f"  [{i}/{len(byfile)}] {os.path.basename(rel)}  {len(rs)} events", flush=True)
        for r in rs:
            t0, dur = float(r["local_start_s"]), float(r["duration_s"])
            hop = max(1, int(args.hop * sr))
            starts = list(range(int(t0 * sr),
                                max(int(t0 * sr) + 1, int((t0 + dur) * sr) - WINDOW_SAMPLES + 1),
                                hop)) or [int(t0 * sr)]
            best = max((run(window(sig, s)) + (s,) for s in starts), key=lambda x: x[0])
            miss_emb.append(best[1])
            miss_meta.append(dict(file=rel, start_s=round(best[2] / sr, 1),
                                  dur_s=dur, logit=round(best[0], 2)))

    # ---- the WHOLE class, not a sample: a source absent from a subsample would be
    # invisible to exactly the question being asked.
    d = os.path.join(args.lib, args.cls)
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".wav", ".flac", ".mp3")))
    print(f"  library {args.cls}: {len(fs)} clips", flush=True)
    lib_emb, lib_file = [], []
    for j, f in enumerate(fs, 1):
        if j % 50 == 0:
            print(f"    {j}/{len(fs)}", flush=True)
        try:
            sig, sr = open_audio_file(os.path.join(d, f))
        except Exception as e:
            print(f"    !! {f}: {e}")
            continue
        c = max(0, len(sig) // 2 - WINDOW_SAMPLES // 2)
        lib_emb.append(run(window(sig, c))[1])
        lib_file.append(f)

    M = np.stack(miss_emb); L = np.stack(lib_emb)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    D = 1.0 - M @ L.T                      # cosine distance, misses x library clips

    recs = np.array([recording(f) for f in lib_file])
    n_clips = collections.Counter(recs)

    # Distance scale: is the class even in range of these events? Both sides must be
    # NEAREST-neighbour distances. Comparing a miss's minimum over 348 clips against the
    # median of all library pairs is not a comparison -- a minimum over a large set is
    # small by construction, and that alone made the misses look closer to the class than
    # the class is to itself.
    LL = 1.0 - L @ L.T
    np.fill_diagonal(LL, np.inf)
    within = float(np.median(LL.min(1)))
    nearest = float(np.median(D.min(1)))
    print(f"\n  median library clip -> nearest other library clip : {within:.3f}")
    print(f"  median missed event -> nearest library clip       : {nearest:.3f} "
          f"({nearest/within:.2f}x)")

    votes = collections.Counter()
    dsum = collections.defaultdict(list)
    for i in range(D.shape[0]):
        for j in np.argsort(D[i])[:args.topk]:
            votes[recs[j]] += 1
            dsum[recs[j]].append(float(D[i, j]))

    rows = []
    for rec, v in votes.most_common():
        rows.append(dict(recording=rec, kind=kind(rec), votes=v,
                         n_misses_nearest=len({i for i in range(D.shape[0])
                                               if recs[np.argmin(D[i])] == rec}),
                         n_clips=n_clips[rec],
                         votes_per_clip=round(v / n_clips[rec], 2),
                         mean_dist=round(float(np.mean(dsum[rec])), 3)))
    rows.sort(key=lambda r: (-r["votes_per_clip"], r["mean_dist"]))

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\n  {len(miss_meta)} missed events vs {len(lib_file)} clips "
          f"from {len(n_clips)} source recordings")
    print(f"  {'recording':44s} {'kind':5s} {'votes':>5s} {'top1':>4s} {'clips':>5s} "
          f"{'v/clip':>6s} {'dist':>6s}")
    for r in rows[:20]:
        print(f"  {r['recording']:44.44s} {r['kind']:5s} {r['votes']:5d} "
              f"{r['n_misses_nearest']:4d} {r['n_clips']:5d} "
              f"{r['votes_per_clip']:6.2f} {r['mean_dist']:6.3f}")
    # Where the class's mass sits vs where the votes go. A class can be mostly reference
    # audio while every miss votes for the handful of field recordings, which says the
    # reference clips are holding the count up and the field ones are doing the work.
    print(f"\n  {'provenance':8s} {'clips':>6s} {'share':>6s} {'votes':>6s} {'share':>6s}")
    tv = sum(votes.values()) or 1
    for kd in ("XC", "ML", "iNat", "field", "other"):
        c = sum(n for r, n in n_clips.items() if kind(r) == kd)
        v = sum(n for r, n in votes.items() if kind(r) == kd)
        if c or v:
            print(f"  {kd:8s} {c:6d} {c/len(lib_file):6.0%} {v:6d} {v/tv:6.0%}")

    silent = [r for r in n_clips if r not in votes]
    print(f"\n  {len(silent)}/{len(n_clips)} source recordings are nearest to NO missed event "
          f"({sum(n_clips[r] for r in silent)} clips, "
          f"{sum(n_clips[r] for r in silent)/len(lib_file):.0%} of the class)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
