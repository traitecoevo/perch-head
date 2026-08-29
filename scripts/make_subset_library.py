#!/usr/bin/env python3
"""Build a scratch library that exposes only the class folders you want re-embedded.

WHY THIS EXISTS. `extract_embeddings.py` has no "only these classes" flag, and a full pass
over `reallybig` costs ~8 h. But it does not need one: the vocabulary (and, with
`--n-distractors 0`, the column order) comes from `--label-vocab`, and a class whose folder
is EMPTY is skipped with "no usable clips" rather than dropped from the vocabulary. So a
directory holding one empty subdir per vocabulary class, plus symlinks to the real folders
for the handful that changed, yields a cache whose columns are byte-compatible with the
full one and whose rows cover only the changed classes. `splice_cache.py` then merges it.

THE EMPTY FOLDERS ARE NOT OPTIONAL. `_list_clips()` calls `os.listdir()` unguarded, so a
vocabulary class with no folder raises FileNotFoundError rather than being skipped.

SYMLINKS, NOT COPIES: the changed folders are large and the extractor only reads them.

Usage, for the 2-class Arm B:

    python scripts/make_subset_library.py \
        --vocab out/run0-9/cache_labels.txt \
        --library "$CALL_LIBRARY/reallybig" \
        --out /tmp/minilib_armB \
        --class "Cincloramphus cruralis_Brown Songlark" \
        --class "Gymnorhina tibicen_Australian Magpie"

Then extract with `--library /tmp/minilib_armB --label-vocab <the same vocab file>
--species-list <the FULL species list> --n-distractors 0 --all-windows --cap 0
--no-nonevents`.
"""

from __future__ import annotations

import argparse
import os
import shutil


def _read_lines(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", required=True,
                    help="one class-folder name per line; must be the SAME file passed to "
                         "extract_embeddings.py --label-vocab (write it from the control "
                         "cache's `labels` array).")
    ap.add_argument("--library", required=True, help="the real library the symlinks point into.")
    ap.add_argument("--out", required=True, help="scratch library to create.")
    ap.add_argument("--class", dest="classes", action="append", default=[], required=True,
                    help="class folder to expose for real; repeatable.")
    ap.add_argument("--force", action="store_true",
                    help="replace --out if it already exists.")
    args = ap.parse_args()

    vocab = _read_lines(args.vocab)
    if len(set(vocab)) != len(vocab):
        raise SystemExit("vocab file has duplicate entries -- column identity would be ambiguous.")

    missing = [c for c in args.classes if c not in set(vocab)]
    if missing:
        raise SystemExit(
            f"--class not in the vocabulary, so it would have no column: {missing}\n"
            "  A class that is genuinely new needs a full re-extract, not a splice: it "
            "changes the width of Y and every stored recognizer's output layer.")

    for c in args.classes:
        src = os.path.join(args.library, c)
        if not os.path.isdir(src):
            raise SystemExit(f"--class has no folder in the library: {src}")

    if os.path.exists(args.out):
        if not args.force:
            raise SystemExit(f"{args.out} exists; pass --force to replace it.")
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    for cls in vocab:
        os.mkdir(os.path.join(args.out, cls))
    for cls in args.classes:
        dst = os.path.join(args.out, cls)
        os.rmdir(dst)
        os.symlink(os.path.abspath(os.path.join(args.library, cls)), dst)

    n_clips = sum(len(os.listdir(os.path.join(args.out, c))) for c in args.classes)
    print(f"{args.out}: {len(vocab)} vocabulary folders, "
          f"{len(args.classes)} symlinked ({n_clips} clips), {len(vocab) - len(args.classes)} empty")
    for c in args.classes:
        print(f"  real: {c} ({len(os.listdir(os.path.join(args.out, c)))} clips)")


if __name__ == "__main__":
    main()
