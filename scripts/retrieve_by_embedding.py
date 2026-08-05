"""Mine audio for windows that RESEMBLE seed clips, ranked by embedding cosine rather
than by any classifier's score.

Why this exists: every other cutter in the pipeline picks windows a detector already
fires on, so it can only ever find more of what the detector already hears. The classes
that fail in the field fail on faint, wind-masked events -- exactly the windows no
detector fires on -- so score-driven mining cannot reach them by construction, and each
rebuild reproduces the gap. Cosine-to-a-seed has no such blind spot: a quiet call sitting
under wind still lands near a loud clean one in Perch space, several ranks below where any
head would put it.

Read the output the same way as the rest of the pipeline: `max_cos` says a window looks
like the seeds, NOT that the target is audible in it. Everything here is an audition
candidate. Stage to `call_library/_audition/`, never straight into `reallybig`.

Two things worth knowing before trusting a run:

  * **Seeds decide what you get.** Seeding from the class's own clean library clips
    retrieves more clean audio -- useful for coverage, useless for the wind gap. To reach
    the faint tail, seed from the events the recognizer MISSED (cut them out of the
    soundscape first) or from the faintest clips the class already holds.
  * **Cosine is not calibrated across species.** 0.55 was a sensible floor for the cat
    batch; it is not a constant. Sort by `max_cos`, look at where the ranking falls apart,
    and set `--min-cos` from that -- do not port a threshold between species.

`rms_dbfs` is recorded per clip so the faint end of a batch is visible without listening:
a retrieval run that returns only loud windows did not do the job it was built for.

Usage:
  .venv/bin/python scripts/retrieve_by_embedding.py \
      --seed "$CALL_LIBRARY/reallybig/Felis catus_Cat" \
      --search /Volumes/PREDATOR/.../Fowlers_Gap \
      --out-dir "$CALL_LIBRARY/_audition/Felis catus_Cat_retrieval_2026-08-05" \
      --min-cos 0.55 --per-source-cap 20
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np

from perch_head.audio import AUDIO_EXT, open_audio_file
from perch_head.embed import SAMPLE_RATE, WINDOW_SAMPLES, perch_embed

METHOD = "embedding_retrieval"
MANIFEST_COLUMNS = ["clip", "source_file", "start_s", "max_cos", "method", "rms_dbfs"]


def slug_for(seed_paths: list[str], explicit: str | None) -> str:
    """Clip-name prefix: 'Felis catus_Cat' -> 'felis_catus'. Genus/species only, so the
    prefix stays stable if the common name in the folder is ever revised."""
    if explicit:
        return explicit
    first = seed_paths[0]
    base = os.path.basename(first.rstrip("/")) if os.path.isdir(first) else \
        os.path.basename(os.path.dirname(first))
    return base.split("_", 1)[0].lower().replace(" ", "_") or "seed"


def audio_under(path: str) -> list[str]:
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _, files in os.walk(path):
        out += [os.path.join(root, f) for f in files
                if f.lower().endswith(AUDIO_EXT) and not f.startswith(".")]
    return sorted(out)


def embed_batched(windows: list[np.ndarray], batch: int) -> np.ndarray:
    """Embed in fixed-size batches. A one-hour file at a 2.5 s hop is ~1440 windows =
    ~0.9 GB of float32 if stacked whole, so the stacking is what needs bounding, not the
    model call."""
    out = []
    for i in range(0, len(windows), batch):
        out.append(perch_embed(np.stack(windows[i:i + batch])))
    return np.concatenate(out) if out else np.zeros((0, 1536), dtype="float32")


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def seed_matrix(seed_paths: list[str], batch: int, limit: int) -> tuple[np.ndarray, int]:
    """One centred window per seed clip -> L2-normalised (n_seeds, 1536).

    Centred, not the first window: library clips are short and a leading window of a 5 s
    clip padded from 3 s is 40% silence, which drags every cosine toward the silence
    direction instead of the species.
    """
    files = []
    for p in seed_paths:
        files += audio_under(p)
    if limit and len(files) > limit:
        # Deterministic thinning, so a rerun retrieves the same windows.
        step = len(files) / limit
        files = [files[int(i * step)] for i in range(limit)]
    wins = []
    for f in files:
        try:
            sig, _ = open_audio_file(f)
        except Exception as exc:
            print(f"  !! seed unreadable, skipping: {os.path.basename(f)} ({exc})")
            continue
        c = len(sig) // 2
        s = max(0, c - WINDOW_SAMPLES // 2)
        w = sig[s:s + WINDOW_SAMPLES]
        if len(w) < WINDOW_SAMPLES:
            w = np.pad(w, (0, WINDOW_SAMPLES - len(w)))
        wins.append(w.astype("float32"))
    if not wins:
        raise SystemExit("no readable seed clips")
    return unit(embed_batched(wins, batch)), len(wins)


def nms(cands: list[tuple[float, float]], min_gap_s: float) -> list[tuple[float, float]]:
    """Greedy best-first suppression over start times. With a hop below the 5 s window,
    neighbouring windows share most of their audio and would otherwise be cut as separate
    clips that are near-duplicates of each other -- the same failure the top-up path hit
    with a 1 s hop."""
    kept: list[tuple[float, float]] = []
    for cos, start in sorted(cands, key=lambda c: -c[0]):
        if all(abs(start - k[1]) >= min_gap_s for k in kept):
            kept.append((cos, start))
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", action="append", required=True,
                    help="seed clip file or directory; repeatable")
    ap.add_argument("--search", action="append", required=True,
                    help="audio file or directory to mine; repeatable")
    ap.add_argument("--out-dir", required=True, help="staging dir (an _audition batch)")
    ap.add_argument("--min-cos", type=float, default=0.55,
                    help="cosine floor; species-specific, do not port between classes")
    ap.add_argument("--hop", type=float, default=2.5, help="window hop, seconds")
    ap.add_argument("--nms-s", type=float, default=5.0,
                    help="minimum spacing between kept clips, seconds")
    ap.add_argument("--per-source-cap", type=int, default=20,
                    help="max clips from any one source recording")
    ap.add_argument("--max-clips", type=int, default=0, help="0 = no global cap")
    ap.add_argument("--seed-limit", type=int, default=200,
                    help="thin the seed set to at most this many clips")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tag", default=None, help="clip-name prefix (default: from seed dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="score and report, write no audio and no manifest")
    args = ap.parse_args()

    import soundfile as sf

    seeds, n_seed = seed_matrix(args.seed, args.batch, args.seed_limit)
    slug = slug_for(args.seed, args.tag)
    print(f"seeds: {n_seed} clips -> {seeds.shape}   prefix {slug!r}")

    targets = []
    for s in args.search:
        targets += audio_under(s)
    print(f"search: {len(targets)} files\n")

    hop_samples = max(1, int(args.hop * SAMPLE_RATE))
    rows, t0 = [], time.time()

    for i, path in enumerate(targets, 1):
        try:
            sig, _ = open_audio_file(path)
        except Exception as exc:
            print(f"  [{i}/{len(targets)}] {os.path.basename(path)}: unreadable ({exc})")
            continue
        if len(sig) < WINDOW_SAMPLES:
            sig = np.pad(sig, (0, WINDOW_SAMPLES - len(sig)))
        starts = list(range(0, len(sig) - WINDOW_SAMPLES + 1, hop_samples))
        wins = [sig[s:s + WINDOW_SAMPLES] for s in starts]
        cos = (unit(embed_batched(wins, args.batch)) @ seeds.T).max(axis=1)

        cands = [(float(c), s / SAMPLE_RATE) for c, s in zip(cos, starts)
                 if c >= args.min_cos]
        kept = nms(cands, args.nms_s)[:args.per_source_cap]
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"  [{i}/{len(targets)}] {stem}: {len(cands)} over floor -> {len(kept)} kept"
              f"  (best {cos.max():.3f})", flush=True)

        for c, start in sorted(kept, key=lambda k: k[1]):
            w = sig[int(start * SAMPLE_RATE):int(start * SAMPLE_RATE) + WINDOW_SAMPLES]
            rms = float(np.sqrt((w.astype("float64") ** 2).mean()))
            rows.append(dict(
                clip=f"{slug}__{stem}__{start:.1f}s__cos{c:.2f}.wav",
                source_file=os.path.basename(path), start_s=round(start, 1),
                max_cos=round(c, 4), method=METHOD,
                rms_dbfs=round(20 * math.log10(rms + 1e-12), 1),
                _win=w))

    rows.sort(key=lambda r: -r["max_cos"])
    if args.max_clips:
        rows = rows[:args.max_clips]

    if not rows:
        raise SystemExit("nothing over the cosine floor -- lower --min-cos or reseed")

    q = np.percentile([r["rms_dbfs"] for r in rows], [10, 50, 90])
    print(f"\n{len(rows)} clips from {len({r['source_file'] for r in rows})} recordings"
          f"   cos {rows[-1]['max_cos']:.2f}-{rows[0]['max_cos']:.2f}"
          f"   rms_dbfs p10/p50/p90 {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}"
          f"   [{time.time() - t0:.0f}s]")

    if args.dry_run:
        print("dry run -- nothing written")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for r in rows:
        sf.write(os.path.join(args.out_dir, r["clip"]), r.pop("_win"), SAMPLE_RATE,
                 subtype="PCM_16")
    manifest = os.path.join(args.out_dir, "clips_manifest.csv")
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
