#!/usr/bin/env python3
"""Splice a few re-extracted classes into an existing training cache.

WHY. Adding clips to a handful of class folders currently means an ~8 h full re-extract to
recompute ~85k vectors that come back numerically identical, because Perch's backbone is
frozen. This merges a small cache built by `make_subset_library.py` + `extract_embeddings.py`
into a control cache, replacing exactly the changed classes' rows.

WHY IT IS SAFE WITHOUT THE `source` ARRAY. The cache has no per-row provenance, which is the
blocker recorded in CLAUDE.md for a general append mode. It is not a blocker here: `Y` rows
are strictly one-hot (`y[:, ci] = 1.0`, one folder per row), so the control rows belonging to
species class `c` are exactly `Y[:, c] == 1`. That identification fails only for NON-EVENT
rows, which are all-zero and therefore indistinguishable -- so this script refuses to splice
a non-event folder. Species classes only.

THE SPLIT IS THE POINT, NOT A DETAIL. `extract_embeddings.py` draws `split` as
`default_rng(seed).permutation(len(X))`, keyed on len(X) alone, so a full re-extract re-draws
the train/val assignment of EVERY row -- changing the model and the yardstick at once, and
destroying any before/after comparison on unchanged classes. This script keeps each retained
row's existing split and draws only for the new rows. That is what preserves val AUPRC as a
usable early read and keeps the unchanged classes as a negative control.

Two levels of verification, both required before trusting an arm:
  * `tests/test_splice_cache.py` -- pure numpy, no checkpoint: the row algebra and the guards.
  * `verify_subset_extraction.py` -- routes one UNCHANGED class through the subset path and
    checks the vectors come back. NOT bit-identical: XLA picks different tiling for a
    different batch shape, so measured drift is ~1e-6 L2 (Gibberbird, 2026-08-22). The real
    criterion is that rows pair 1:1 and the pairing is unambiguous -- matched pairs sat
    4.6e5x closer than the nearest non-matching row. Row ORDER also legitimately differs,
    since `random.shuffle(files)` runs off a different RNG state in a subset pass.
"""

from __future__ import annotations

import argparse

import numpy as np

VAL_FRACTION = 0.2


def _load(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in ("X", "Y", "labels", "is_present", "split")}


def changed_columns(subset: dict) -> np.ndarray:
    """Columns the subset actually carries rows for."""
    return np.flatnonzero(subset["Y"].any(axis=0))


def splice(control: dict, subset: dict, seed: int = 0) -> dict:
    if list(control["labels"]) != list(subset["labels"]):
        raise SystemExit(
            "label vocabularies differ -- the subset was not extracted with --label-vocab "
            "pointing at the control's labels, or --n-distractors was not 0 (which reorders "
            "columns into present-then-distractor). Columns must match exactly.")
    if not np.array_equal(control["is_present"], subset["is_present"]):
        raise SystemExit("is_present differs between control and subset; species lists disagree.")

    zero_rows = int((subset["Y"].sum(axis=1) == 0).sum())
    if zero_rows:
        raise SystemExit(
            f"subset holds {zero_rows} all-zero (non-event) rows. Their source folder is not "
            "recoverable from Y, so they cannot be spliced. Re-extract with --no-nonevents.")
    multi = int((subset["Y"].sum(axis=1) > 1).sum())
    if multi:
        raise SystemExit(f"subset holds {multi} multi-hot rows; splice assumes one folder per row.")

    cols = changed_columns(subset)
    if cols.size == 0:
        raise SystemExit("subset carries no rows at all -- nothing to splice.")

    drop = control["Y"][:, cols].any(axis=1)
    keep = ~drop

    rng = np.random.default_rng(seed)
    n_new = len(subset["X"])
    n_val = int(round(VAL_FRACTION * n_new))
    perm = rng.permutation(n_new)
    new_split = np.array(["train"] * n_new)
    new_split[perm[:n_val]] = "val"

    out = {
        "X": np.concatenate([control["X"][keep], subset["X"]], axis=0).astype("float32"),
        "Y": np.concatenate([control["Y"][keep], subset["Y"]], axis=0).astype("float32"),
        "labels": control["labels"],
        "is_present": control["is_present"],
        "split": np.concatenate([control["split"][keep], new_split]),
    }

    names = [str(control["labels"][c]) for c in cols]
    print(f"changed classes ({len(cols)}): {names}")
    print(f"  control rows dropped: {int(drop.sum())}   subset rows added: {n_new}")
    print(f"  rows {len(control['X'])} -> {len(out['X'])}")
    print(f"  retained rows keep their original split; {n_val}/{n_new} new rows drawn as val")
    print(f"  split: {int((out['split'] == 'train').sum())} train / "
          f"{int((out['split'] == 'val').sum())} val")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, help="the frozen full cache to splice into.")
    ap.add_argument("--subset", required=True, help="cache covering only the changed classes.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="seeds the split draw for NEW rows only.")
    args = ap.parse_args()

    control, subset = _load(args.control), _load(args.subset)
    out = splice(control, subset, args.seed)
    np.savez_compressed(args.out, **out)
    print(f"\nSaved {args.out}")
    print(f"  X {out['X'].shape}  Y {out['Y'].shape}  classes {len(out['labels'])}")


if __name__ == "__main__":
    main()
