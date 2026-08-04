"""Extract Perch embeddings for a WHOLE clip library -> per-clip npz + centroid CSV.

Sibling of `scripts/extract_embeddings.py`, and deliberately a different artifact:

    extract_embeddings.py          (X, Y, split) training cache -- rows are WINDOWS,
                                   capped per class, scoped by --species-list, no filenames
    extract_library_embeddings.py  per-species arrays + filenames -- one row per CLIP,
                                   every class, no labels, nothing to train on

The second shape is what clip-level tools need: curation
(`Training_library_assembly_pipeline/curation/`) moves individual files, and `birdnetEmbed`
plots per-species clouds, so both index embeddings BY FILENAME. The npz key layout here is
exactly the one BirdNET-Analyzer's `embedding_analysis/extract_head_embeddings.py` writes
(`species_list`, `emb_<class>`, `files_<class>`, `centroid_<class>`), so those tools read a
Perch artifact without a line of change.

Same shape does NOT mean same space: 1536-d Perch here, 2048-d BirdNET custom head there.
Centroids, cosine distances and near-duplicate thresholds computed in one are meaningless
against the other, which is what the `embedding_space` field in the npz is for -- check it
before comparing two artifacts. See BirdNET-Analyzer/CLAUDE.md, "Two embedding spaces".

Every folder is embedded, including the `Environment_*` / `Homo sapiens_*` / `Noise` helper
classes. They carry no label here -- there are no labels -- and curation specifically needs
them (noise clustering, noise-adjacency quarantine), so the non-event handling that matters
for training has nothing to do at this stage.

One row per clip, from the clip's MIDDLE window: the library's clips are short (most yield a
single 5 s window anyway) and a centred window is what the BirdNET artifact uses, so the two
stay comparable clip-for-clip.

Usage:
    .venv/bin/python scripts/extract_library_embeddings.py \
        --library "$CALL_LIBRARY/reallybig" \
        --output  "$CALL_LIBRARY/embeddings/reallybig_run0-3-ph"
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from perch_head.audio import AUDIO_EXT, open_audio_file, split_signal
from perch_head.embed import default_checkpoint_path, perch_embed


def short_name(class_dir: str) -> str:
    """Library folder is 'Genus species_Common'; the common name is after the first '_'."""
    parts = class_dir.split("_", 1)
    return parts[1] if len(parts) > 1 else class_dir


def _class_dirs(library: str) -> list[str]:
    return sorted(
        d for d in os.listdir(library)
        if os.path.isdir(os.path.join(library, d)) and not d.startswith(".")
    )


def _clip_windows(cls_dir: str, files: list[str], minlen_s: float) -> tuple[list[np.ndarray], list[str]]:
    """One centred window per readable clip, paired with the filename it came from."""
    windows, kept = [], []
    for fname in files:
        try:
            sig, rate = open_audio_file(os.path.join(cls_dir, fname))
            chunks = split_signal(sig, rate, minlen_s=minlen_s)
        except Exception:
            continue
        if not chunks:
            continue
        windows.append(np.asarray(chunks[len(chunks) // 2], dtype="float32"))
        kept.append(fname)
    return windows, kept


def _embed_batched(windows: list[np.ndarray], batch: int) -> np.ndarray:
    embs = []
    for i in range(0, len(windows), batch):
        embs.append(perch_embed(np.asarray(windows[i:i + batch], dtype="float32")))
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 1536), dtype="float32")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True, help="clip library (one folder per class).")
    ap.add_argument("--output", required=True, help="base name for output files (no extension).")
    ap.add_argument("--cap", type=int, default=0, help="max clips per class; 0 = no cap (default).")
    ap.add_argument("--minlen-s", type=float, default=1.0,
                    help="minimum real (non-padded) seconds required to keep a window.")
    ap.add_argument("--embed-batch", type=int, default=64)
    args = ap.parse_args()

    checkpoint = default_checkpoint_path()
    class_dirs = _class_dirs(args.library)

    print(f"Model:  {checkpoint}")
    print("Layer:  Perch penultimate embedding  (1536-d)")
    print(f"Input:  {args.library}")
    print(f"Found {len(class_dirs)} class directories\n")

    all_embeddings: dict[str, np.ndarray] = {}
    all_filenames: dict[str, list[str]] = {}
    total_clips_processed = total_clips_failed = 0
    start_time = time.time()

    for idx, cls in enumerate(class_dirs):
        cls_dir = os.path.join(args.library, cls)
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(AUDIO_EXT))
        if args.cap:
            files = files[:args.cap]
        windows, kept = _clip_windows(cls_dir, files, args.minlen_s)
        total_clips_failed += len(files) - len(kept)

        if windows:
            emb = _embed_batched(windows, args.embed_batch)
            all_embeddings[cls] = emb
            all_filenames[cls] = kept
            total_clips_processed += len(emb)

        elapsed = time.time() - start_time
        cps = total_clips_processed / elapsed if elapsed > 0 else 0
        print(
            f"  [{idx + 1}/{len(class_dirs)}] {short_name(cls):.<45s} "
            f"{len(kept):>4d} clips  (total: {total_clips_processed}, "
            f"{cps:.1f} clips/s, elapsed: {elapsed:.0f}s)",
            flush=True,
        )

    print("\nEmbedding extraction complete!")
    print(f"  Total clips embedded: {total_clips_processed}")
    print(f"  Failed: {total_clips_failed}")
    print(f"  Classes with embeddings: {len(all_embeddings)}")

    active = sorted(all_embeddings)
    centroids = {cls: all_embeddings[cls].mean(axis=0) for cls in active}

    save_dict: dict[str, np.ndarray] = {
        "species_list": np.array(active),
        "total_clips_processed": np.array([total_clips_processed]),
        "total_clips_failed": np.array([total_clips_failed]),
        "model_path": np.array([checkpoint]),
        "input_dir": np.array([args.library]),
        "embedding_space": np.array(["perch_penultimate"]),
    }
    for cls in active:
        save_dict[f"emb_{cls}"] = all_embeddings[cls]
        save_dict[f"centroid_{cls}"] = centroids[cls]
        save_dict[f"files_{cls}"] = np.array(all_filenames[cls])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    npz_path = f"{args.output}_embeddings.npz"
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved binary embeddings: {npz_path}")

    emb_dim = len(next(iter(centroids.values()))) if centroids else 0
    csv_path = f"{args.output}_centroids.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "short_name"] + [f"d{i}" for i in range(emb_dim)])
        for cls in active:
            w.writerow([cls, short_name(cls)] + [f"{v:.6f}" for v in centroids[cls]])
    print(f"Saved centroid CSV: {csv_path}")


if __name__ == "__main__":
    main()
