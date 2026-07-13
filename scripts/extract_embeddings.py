"""Extract Perch embeddings from a training-clip library -> a cached npz for head training.

Walks a call/clip library (one folder per class, folder name "Genus species_Common Name"
by convention) and feeds each clip through Perch: `audio.open_audio_file(path, 32000)` +
`audio.split_signal(sig, 32000, 5.0, 0, SIG_MINLEN)` (zero-pads short clips to 5 s,
end-aligned, no peak-normalization) -> `perch_embed()` -> the 1536-d penultimate embedding.
Caches (X, Y) so a head can be trained on it repeatedly without re-running Perch.

Three class roles, each getting a column (or none, for non-events) in the label matrix Y:
  * PRESENT      — species you actually want to detect (e.g. the species confirmed in a
                   labeled evaluation soundscape). Given via --species-list. Always kept.
  * DISTRACTORS  — additional classes from the library, included so the head has to learn
                   real separability instead of an artificially small closed set that can't
                   leak into anything. --n-distractors controls how many; -1 = all of them.
  * NON-EVENTS   — helper/negative folders (rain, wind, human noise, ...), identified by
                   --nonevent-prefixes / --nonevent-exact. These get an ALL-ZERO label row
                   and NO output column: pure hard negatives that push every real class down,
                   never upsampled. See docs/training.md.

Species-list format (--species-list): one scientific binomial per line (case-insensitive
match against each library folder's "Genus species" prefix), blank lines and lines starting
with '#' ignored. See configs/species/ for a worked example.

Label-vocab format (--label-vocab), optional: one "Genus species_Common Name" folder name
per line, fixing the full class vocabulary + column order (e.g. to reproduce another
recognizer's exact class set for a fair comparison). If omitted, the vocabulary is derived
by listing --library's subfolders directly (excluding non-event folders), sorted for
determinism.

Output npz: X (n,1536) float32, Y (n,C) float32 multi-hot, labels (C class folder names),
is_present (C bool), split (n,) 'train'/'val' (seeded 80/20).
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np

import birdnet_analyzer.config as cfg
from birdnet_analyzer import audio

from perch_head.embed import perch_embed

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


def _read_lines(path: str) -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def _sci_prefix(folder: str) -> str:
    """Library folder is 'Genus species_Common'; scientific part is before the first '_'."""
    return folder.split("_", 1)[0].strip()


def _load_species_list(path: str) -> set[str]:
    return {s.strip().lower() for s in _read_lines(path)}


def _derive_vocab(library: str, nonevent_prefixes: tuple[str, ...], nonevent_exact: tuple[str, ...]) -> list[str]:
    return sorted(
        d for d in os.listdir(library)
        if os.path.isdir(os.path.join(library, d))
        and not d.startswith(nonevent_prefixes)
        and d not in nonevent_exact
    )


def _list_clips(library: str, folder: str) -> list[str]:
    fdir = os.path.join(library, folder)
    return [f for f in os.listdir(fdir) if f.lower().endswith(AUDIO_EXT)]


def _clip_windows(library: str, folder: str, files: list[str], cap: int, first_only: bool) -> list[np.ndarray]:
    """Up to `cap` clips -> list of (160000,) windows via Perch's feed path (zero-pad)."""
    fdir = os.path.join(library, folder)
    out: list[np.ndarray] = []
    for f in files:
        if len(out) >= cap:
            break
        try:
            sig, rate = audio.open_audio_file(os.path.join(fdir, f), sample_rate=32000)
            chunks = audio.split_signal(sig, rate, 5.0, 0, cfg.SIG_MINLEN)
        except Exception:
            continue
        if not chunks:
            continue
        picks = chunks[:1] if first_only else chunks
        for c in picks:
            out.append(np.asarray(c, dtype="float32"))
            if len(out) >= cap:
                break
    return out


def _embed_batched(windows: list[np.ndarray], batch: int) -> np.ndarray:
    embs = []
    for i in range(0, len(windows), batch):
        embs.append(perch_embed(np.asarray(windows[i:i + batch], dtype="float32")))
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1536), dtype="float32")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True, help="path to the clip library (one folder per class).")
    ap.add_argument("--species-list", required=True,
                    help="text file of present-species scientific names, one per line.")
    ap.add_argument("--label-vocab", default=None,
                    help="optional text file fixing the full class vocabulary (folder names); "
                         "default: derive from --library's subfolders.")
    ap.add_argument("--out", required=True, help="output npz path.")
    ap.add_argument("--n-distractors", type=int, default=150,
                    help="non-present classes to include beyond the present set; -1 = all.")
    ap.add_argument("--distractor-select", choices=("common", "random"), default="common",
                    help="'common' = largest folders first; 'random' = seeded random sample.")
    ap.add_argument("--cap", type=int, default=150, help="max clips per class folder.")
    ap.add_argument("--all-windows", action="store_true",
                    help="use every 5 s window of each clip (default: first window only).")
    ap.add_argument("--no-nonevents", action="store_true", help="skip helper/non-event folders.")
    ap.add_argument("--nonevent-prefixes", default="Environment_,Homo sapiens_",
                    help="comma-separated folder-name prefixes treated as non-events (all-zero rows).")
    ap.add_argument("--nonevent-exact", default="Noise",
                    help="comma-separated exact folder names treated as non-events.")
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-classes", type=int, default=0,
                    help="smoke test: cap total species classes (present kept first).")
    args = ap.parse_args()
    random.seed(args.seed)

    nonevent_prefixes = tuple(p.strip() for p in args.nonevent_prefixes.split(",") if p.strip())
    nonevent_exact = tuple(p.strip() for p in args.nonevent_exact.split(",") if p.strip())

    labels_all = (_read_lines(args.label_vocab) if args.label_vocab
                  else _derive_vocab(args.library, nonevent_prefixes, nonevent_exact))
    present_sci = _load_species_list(args.species_list)
    sci_of = {lbl: _sci_prefix(lbl).lower() for lbl in labels_all}

    present_lbls = [lbl for lbl in labels_all if sci_of[lbl] in present_sci]
    distractor_pool = [lbl for lbl in labels_all if lbl not in present_lbls]

    missing = present_sci - {sci_of[lbl] for lbl in present_lbls}
    if missing:
        print(f"WARNING: {len(missing)} species-list entries have no library folder: {sorted(missing)}")
    print(f"Present species: {len(present_sci)}; matched to a library folder: {len(present_lbls)}")

    if args.distractor_select == "common":
        distractor_pool.sort(key=lambda lbl: -len(_list_clips(args.library, lbl)))
    else:
        random.shuffle(distractor_pool)
    n_dist = len(distractor_pool) if args.n_distractors < 0 else args.n_distractors
    distractor_lbls = distractor_pool[:n_dist]

    species_lbls = present_lbls + distractor_lbls
    if args.limit_classes and len(species_lbls) > args.limit_classes:
        species_lbls = species_lbls[:args.limit_classes]
        present_lbls = [lbl for lbl in present_lbls if lbl in species_lbls]
    col = {lbl: i for i, lbl in enumerate(species_lbls)}
    n_classes = len(species_lbls)
    is_present = np.array([lbl in set(present_lbls) for lbl in species_lbls], dtype=bool)

    nonevent_lbls = []
    if not args.no_nonevents:
        nonevent_lbls = [d for d in sorted(os.listdir(args.library))
                         if d.startswith(nonevent_prefixes) or d in nonevent_exact]

    print(f"Species classes (output neurons): {n_classes} "
          f"({len(present_lbls)} present + {n_classes - len(present_lbls)} distractors)")
    print(f"Non-event helper folders (all-zero rows): {len(nonevent_lbls)}")
    print(f"Per-class cap: {args.cap}; windows/clip: {'all' if args.all_windows else 'first'}")

    X_parts, Y_parts = [], []
    total = 0
    all_folders = [(lbl, col[lbl]) for lbl in species_lbls] + [(lbl, None) for lbl in nonevent_lbls]
    for k, (folder, ci) in enumerate(all_folders):
        files = _list_clips(args.library, folder)
        random.shuffle(files)
        windows = _clip_windows(args.library, folder, files, args.cap, first_only=not args.all_windows)
        if not windows:
            print(f"  [{k + 1}/{len(all_folders)}] {folder}: no usable clips, skip")
            continue
        emb = _embed_batched(windows, args.embed_batch)
        y = np.zeros((len(emb), n_classes), dtype="float32")
        if ci is not None:
            y[:, ci] = 1.0
        X_parts.append(emb)
        Y_parts.append(y)
        total += len(emb)
        tag = "NON-EVENT" if ci is None else ("present" if is_present[ci] else "distractor")
        print(f"  [{k + 1}/{len(all_folders)}] {folder}: {len(emb)} emb  ({tag})  [total {total}]")

    X = np.concatenate(X_parts, axis=0).astype("float32")
    Y = np.concatenate(Y_parts, axis=0).astype("float32")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    n_val = int(round(0.2 * len(X)))
    val_idx = set(perm[:n_val].tolist())
    split = np.array(["val" if i in val_idx else "train" for i in range(len(X))])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, X=X, Y=Y,
        labels=np.array(species_lbls), is_present=is_present, split=split,
    )
    print(f"\nSaved {args.out}")
    print(f"  X {X.shape}  Y {Y.shape}  classes {n_classes}  present {int(is_present.sum())}")
    print(f"  rows: {int((split == 'train').sum())} train / {int((split == 'val').sum())} val")
    print(f"  positive rows: {int((Y.sum(1) > 0).sum())}  non-event (all-zero) rows: "
          f"{int((Y.sum(1) == 0).sum())}")


if __name__ == "__main__":
    main()
