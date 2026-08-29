#!/usr/bin/env python3
"""Resample some classes' rows at CONSTANT COUNT, to calibrate how far unrelated classes move.

WHY THIS EXISTS. Comparing a retrained head against a noise band measured from seed-only
replicates understates the variation a DATA change causes, because those replicates hold the
data and the train/val split fixed. Measured on the 2026-08-23 Arm B: after correcting an
upsampling confound, 3 of 43 species whose data had not changed at all still moved beyond
|z|>3 of the seed band, and 13 beyond |z|>2 (about 2 expected). So the shared hidden layer
genuinely re-organises when any class's data changes, and a per-species result of that size
cannot be read against the seed band.

This builds the missing null. It replaces the chosen classes' rows with a random subset of a
larger pool, keeping the SAME number of rows per class as the control. Class sizes are
therefore untouched -- which matters, because `_upsample_repeat` anchors its per-class floor
to the largest class, so any size change silently re-balances all 433 classes. The only thing
that differs from the control is WHICH clips those classes contribute. Refit, score, and the
spread of the UNCHANGED classes is the honest band for judging a real arm.

It is also the natural null for a composition arm: a diversity-optimised re-prune has to beat
a random re-draw at the same count, not merely differ from the control.

    python scripts/make_null_cache.py --control out/run0-8/perch_cache.npz \\
        --pool out/run0-9/mini_armB.npz --out out/run0-9/null_cache.npz --seed 101
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, help="the frozen cache to perturb.")
    ap.add_argument("--pool", required=True,
                    help="subset cache holding a LARGER pool of rows for the classes to "
                         "resample (e.g. the arm's mini cache).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=101,
                    help="seeds the row draw. Use a different value per null replicate.")
    args = ap.parse_args()

    c = {k: v for k, v in np.load(args.control, allow_pickle=False).items()}
    p = np.load(args.pool, allow_pickle=False)
    if list(c["labels"]) != list(p["labels"]):
        raise SystemExit("label vocabularies differ between control and pool.")

    cols = np.flatnonzero(p["Y"].any(axis=0))
    rng = np.random.default_rng(args.seed)
    keep = np.ones(len(c["X"]), dtype=bool)
    add_x, add_y, add_split = [], [], []

    for ci in cols:
        ctrl_rows = np.flatnonzero(c["Y"][:, ci] == 1)
        pool_rows = np.flatnonzero(p["Y"][:, ci] == 1)
        n = len(ctrl_rows)
        if len(pool_rows) < n:
            raise SystemExit(
                f"pool has {len(pool_rows)} rows for {c['labels'][ci]!r} but the control has "
                f"{n}; cannot resample at constant count from a smaller pool.")
        pick = rng.choice(pool_rows, size=n, replace=False)
        keep[ctrl_rows] = False
        add_x.append(p["X"][pick])
        add_y.append(p["Y"][pick])
        # Reuse the control rows' own split labels, so the train/val PROPORTION for this class
        # is identical too -- otherwise the null would carry a split change the arm does not.
        add_split.append(c["split"][ctrl_rows])
        print(f"  {str(c['labels'][ci]):45s} {n:5d} rows resampled from a pool of "
              f"{len(pool_rows)}")

    out = {
        "X": np.concatenate([c["X"][keep], *add_x], axis=0).astype("float32"),
        "Y": np.concatenate([c["Y"][keep], *add_y], axis=0).astype("float32"),
        "labels": c["labels"],
        "is_present": c["is_present"],
        "split": np.concatenate([c["split"][keep], *add_split]),
    }
    if len(out["X"]) != len(c["X"]):
        raise SystemExit(f"row count changed {len(c['X'])} -> {len(out['X'])}; not a constant-"
                         "count null.")

    np.savez_compressed(args.out, **out)
    print(f"\nSaved {args.out}")
    print(f"  X {out['X'].shape}  (unchanged from control -- class sizes held constant)")
    print(f"  split: {int((out['split'] == 'train').sum())} train / "
          f"{int((out['split'] == 'val').sum())} val")


if __name__ == "__main__":
    main()
