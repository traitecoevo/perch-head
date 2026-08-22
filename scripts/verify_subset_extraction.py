#!/usr/bin/env python3
"""Prove a subset extraction reproduces the control cache's vectors, before splicing.

Run this on an UNCHANGED class every time you build a subset cache. It is the check that
catches the failure mode `splice_cache.py`'s unit tests cannot see: the row algebra can be
perfect while the extraction itself silently produced different vectors -- a wrong
`--label-vocab`, a stale symlink, a changed window setting -- and the resulting cache trains
without complaint.

WHY NOT BIT-IDENTITY. Perch's backbone is frozen, so the vectors are mathematically the same,
but XLA selects different tiling for a different batch shape and float32 addition is not
associative. Measured drift on a 33-clip class was ~1e-6 L2 (2026-08-22). Bit-identity would
be the wrong assertion; ambiguity of the pairing is the right one, and it is not close --
matched pairs sat 4.6e5x nearer than the closest non-matching row.

    python scripts/verify_subset_extraction.py \
        --control out/run0-8/perch_cache.npz --subset /tmp/minilib_selftest.npz \
        --class "Ashbyia lovensis_Gibberbird"
"""

from __future__ import annotations

import argparse

import numpy as np

# A pairing this unambiguous cannot be an accident; anything near 1.0 means the rows are not
# the same audio. Kept far below the measured 4.6e5 so a genuine regression is loud.
MIN_SEPARATION_RATIO = 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--class", dest="cls", required=True,
                    help="an UNCHANGED class present in both caches.")
    args = ap.parse_args()

    c = np.load(args.control, allow_pickle=False)
    s = np.load(args.subset, allow_pickle=False)
    if list(c["labels"]) != list(s["labels"]):
        raise SystemExit("label vocabularies differ -- wrong --label-vocab on the subset pass.")

    where = np.flatnonzero(c["labels"] == args.cls)
    if not where.size:
        raise SystemExit(f"class not in the vocabulary: {args.cls!r}")
    col = int(where[0])

    A = c["X"][c["Y"][:, col] == 1]
    B = s["X"][s["Y"][:, col] == 1]
    print(f"{args.cls}: control {A.shape[0]} rows, subset {B.shape[0]} rows")
    if A.shape != B.shape:
        raise SystemExit(
            f"row COUNT differs ({A.shape[0]} vs {B.shape[0]}). The class is not unchanged, or "
            "the window settings differ -- --all-windows / --cap / --minlen-s must match the "
            "control's extraction exactly.")

    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    nn = d.argmin(1)
    if sorted(nn.tolist()) != list(range(len(B))):
        raise SystemExit("rows do not pair 1:1 -- two control rows claim the same subset row.")

    matched = d[np.arange(len(A)), nn]
    other = d.copy()
    other[np.arange(len(A)), nn] = np.inf
    ratio = float(other.min() / max(matched.max(), 1e-12))

    print(f"  matched-pair L2:  max {matched.max():.3e}  mean {matched.mean():.3e}")
    print(f"  nearest non-pair: {other.min():.3e}")
    print(f"  separation ratio: {ratio:.3e}x  (need > {MIN_SEPARATION_RATIO:g})")

    if ratio < MIN_SEPARATION_RATIO:
        raise SystemExit("FAIL: pairing is ambiguous; these are not the same vectors.")
    print("\nPASS -- subset extraction reproduces the control for this class.")


if __name__ == "__main__":
    main()
