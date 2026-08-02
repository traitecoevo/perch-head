"""Characterise the soundscape events a recognizer MISSED against the library clips
it was trained on -> per-species feature comparison + an independent detectability read.

The question this answers: when a class at cap still scores ~1% recall in the field,
which way do the missed events differ from the clips in the class? Level? Wind?
Bandwidth? Duration? Or not at all -- in which case the library is not the problem.

Acoustic features are measured identically on both sides so the two are comparable:

  sig_dB     1-8 kHz in-band level, dBFS -- how much target-band signal is present.
             Deliberately NOT broadband RMS: a wind-masked window has high broadband
             RMS and almost no in-band signal, so broadband conflates the two failure
             conditions this script exists to separate.
  wind_dB    10*log10(E[50-500Hz] / E[1-8kHz]) -- low-frequency rumble sitting on top.
  snr_dB     p90-p10 of per-frame in-band energy -- signal against its own background.
  cent_Hz    in-band spectral centroid.
  bw_Hz      in-band 10-90 percentile energy bandwidth.

Perch's native head then scores both sides. This is the load-bearing part: if Perch
finds the target in a missed event, the event was detectable and the recognizer is at
fault; if Perch cannot either, the event is genuinely hard. Read it asymmetrically as
always -- Perch failing on a thinly-recorded species is not evidence about the audio.

Usage:
  .venv/bin/python scripts/compare_misses_to_library.py \
      --fn-csv <experiment>/false_negatives.csv --model run0-3-ph \
      --soundscape-dir <dir of the labeled wavs> --lib <reallybig> --out cmp.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import random

import numpy as np

# eval species name -> (reallybig class dir, Perch labels.csv scientific name)
SPECIES = {
    "australian magpie":      ("Gymnorhina tibicen_Australian Magpie", "Gymnorhina tibicen"),
    "singing honeyeater":     ("Gavicalis virescens_Singing Honeyeater", "Gavicalis virescens"),
    "goat":                   ("Capra hircus_Goat", "Capra hircus"),
    "australasian pipit":     ("Anthus novaeseelandiae_Australian Pipit", "Anthus novaeseelandiae"),
    "white-winged fairywren": ("Malurus leucopterus_White-winged Fairywren", "Malurus leucopterus"),
    "pied butcherbird":       ("Cracticus nigrogularis_Pied Butcherbird", "Cracticus nigrogularis"),
    "australian raven":       ("Corvus coronoides_Australian Raven", "Corvus coronoides"),
    "little corella":         ("Cacatua sanguinea_Little Corella", "Cacatua sanguinea"),
    # Perch's vocabulary is iNat2024/eBird-2021, so it predates the Zebra Finch split:
    # `Taeniopygia castanotis` (current, and what soundscape-eval's synonym list maps to)
    # is absent here and only `guttata` resolves.
    "zebra finch":            ("Taeniopygia guttata_Zebra Finch", "Taeniopygia guttata"),
    "emu":                    ("Dromaius novaehollandiae_Emu", "Dromaius novaehollandiae"),
}


def features(x, sr):
    if len(x) < 1024:
        return None
    n = 1 << int(np.floor(np.log2(len(x))))
    x = x[:n]
    f = np.fft.rfftfreq(n, 1 / sr)
    P = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    band = (f >= 1000) & (f < 8000)
    lo = P[(f >= 50) & (f < 500)].sum()
    mid = P[band].sum()
    scale = (x ** 2).mean() / (P.sum() + 1e-20)
    fb, Pb = f[band], P[band]
    c = float((fb * Pb).sum() / (Pb.sum() + 1e-20))
    cs = np.cumsum(Pb) / (Pb.sum() + 1e-20)
    bw = float(fb[np.searchsorted(cs, .9)] - fb[np.searchsorted(cs, .1)])

    w = int(0.05 * sr)
    fr = x[: len(x) // w * w].reshape(-1, w)
    e = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    snr = float(np.percentile(e, 90) - np.percentile(e, 10))
    return dict(sig_dB=10 * np.log10(mid * scale + 1e-12),
                wind_dB=10 * np.log10((lo + 1e-20) / (mid + 1e-20)),
                snr_dB=snr, cent_Hz=c, bw_Hz=bw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn-csv", required=True)
    ap.add_argument("--model", required=True, help="which model's rows in the FN csv")
    ap.add_argument("--soundscape-dir", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib-sample", type=int, default=90)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    import tensorflow as tf
    from perch_head.audio import open_audio_file
    from perch_head.embed import WINDOW_SAMPLES, default_checkpoint_path

    ckpt = args.checkpoint or default_checkpoint_path()
    with open(os.path.join(ckpt, "assets", "labels.csv")) as fh:
        names = [l.strip() for l in fh][1:]
    # Resolve every target up front. A name Perch does not know is a one-line
    # taxonomy lag, but discovered mid-run it throws away every window scored so
    # far -- so it must be caught before the first inference, not at the species
    # that happens to be last.
    idx, unknown = {}, []
    for sp, (_, sci) in SPECIES.items():
        (idx.setdefault(sp, names.index(sci)) if sci in names else unknown.append((sp, sci)))
    for sp, sci in unknown:
        print(f"  !! {sp}: {sci!r} not in Perch's vocabulary -- skipping this species")
    if not idx:
        raise SystemExit("no target species resolved")

    serving = tf.saved_model.load(ckpt).signatures["serving_default"]

    def score(win, k):
        out = serving(inputs=tf.constant(win[None, :], dtype=tf.float32))
        lg = out["label"].numpy()[0]
        emb = np.asarray(out["embedding"].numpy()[0], dtype=np.float32).ravel()
        return float(lg[k]), int((lg > lg[k]).sum()) + 1, emb

    def centred(sig, at, sr):
        c = int(at * sr)
        h = WINDOW_SAMPLES // 2
        s, e = max(0, c - h), max(0, c - h) + WINDOW_SAMPLES
        w = sig[s:e]
        return np.pad(w, (0, WINDOW_SAMPLES - len(w))) if len(w) < WINDOW_SAMPLES else w

    rows, embs = [], []

    # ---- missed events, grouped by file so each soundscape is decoded once
    fn = [r for r in csv.DictReader(open(args.fn_csv)) if r["model"] == args.model
          and r["species"] in idx]
    byfile = collections.defaultdict(list)
    for r in fn:
        byfile[r["file"]].append(r)
    for i, (rel, rs) in enumerate(sorted(byfile.items()), 1):
        p = os.path.join(args.soundscape_dir, rel)
        if not os.path.isfile(p):
            continue
        sig, sr = open_audio_file(p)
        print(f"  [{i}/{len(byfile)}] {os.path.basename(rel)}  {len(rs)} events", flush=True)
        for r in rs:
            at = float(r["local_start_s"]) + float(r["duration_s"]) / 2
            w = centred(sig, at, sr)
            ft = features(w, sr)
            if not ft:
                continue
            lg, rk, em = score(w, idx[r["species"]])
            embs.append(em)
            rows.append(dict(species=r["species"], side="missed_event", file=rel,
                             dur_s=float(r["duration_s"]), target_logit=round(lg, 3),
                             target_rank=rk, **{k: round(v, 2) for k, v in ft.items()}))

    # ---- library clips
    rnd = random.Random(0)
    for sp, (cls, sci) in SPECIES.items():
        if sp not in idx:
            continue
        d = os.path.join(args.lib, cls)
        if not os.path.isdir(d):
            continue
        fs = [f for f in os.listdir(d) if f.lower().endswith((".wav", ".flac", ".mp3"))]
        fs = rnd.sample(fs, min(len(fs), args.lib_sample))
        print(f"  library {sp}: {len(fs)} clips", flush=True)
        for f in fs:
            try:
                sig, sr = open_audio_file(os.path.join(d, f))
            except Exception:
                continue
            w = centred(sig, len(sig) / sr / 2, sr)
            ft = features(w, sr)
            if not ft:
                continue
            lg, rk, em = score(w, idx[sp])
            embs.append(em)
            rows.append(dict(species=sp, side="library_clip", file=f,
                             dur_s=round(len(sig) / sr, 2), target_logit=round(lg, 3),
                             target_rank=rk, **{k: round(v, 2) for k, v in ft.items()}))

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # embeddings are written parallel to the CSV, same row order: the point of
    # keeping them is asking which library clips a missed event is nearest to.
    # NB 'clip', not 'file': savez_compressed's own first parameter is named `file`,
    # so passing file=... as an array collides with it and raises.
    np.savez_compressed(os.path.splitext(args.out)[0] + "_emb.npz",
                        emb=np.stack(embs).astype(np.float32),
                        side=np.array([r["side"] for r in rows]),
                        species=np.array([r["species"] for r in rows]),
                        clip=np.array([r["file"] for r in rows]))
    print(f"\nwrote {args.out}  ({len(rows)} rows) + _emb.npz {np.stack(embs).shape}")


if __name__ == "__main__":
    main()
