"""Run a trained perch-head on new audio -> a CSV of (file, window, species, score) rows.

Feeds each file through the same path embeddings were trained on: 32 kHz, 5 s windows,
zero-padded/end-aligned/not normalized (`perch_head.inference.windows_for_file`). Only rows
scoring >= --threshold are written, to keep the CSV a manageable size — the head returns a
score for every (window, class) pair, most of which are near zero.

This is retrospective/server-side inference (full compute, no edge/real-time constraint) —
see docs/inference.md for the deployment framing.
"""

from __future__ import annotations

import argparse
import csv

from perch_head.inference import Head, iter_audio_files, predict_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", required=True, help="trained head npz (from train_head.py).")
    ap.add_argument("--audio", required=True, help="an audio file or a folder to walk recursively.")
    ap.add_argument("--out", required=True, help="output predictions CSV.")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="minimum score to keep a row (default 0.1; the head's sigmoid output "
                         "is not calibrated to any particular operating point — tune per use).")
    ap.add_argument("--present-only", action="store_true",
                    help="only report classes flagged is_present in the training npz "
                         "(drops distractor/negative-only classes from the output).")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    head = Head.load(args.head)
    present_labels = {lbl for lbl, p in zip(head.labels, head.is_present) if p} if args.present_only else None

    n_files = n_rows = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "start_s", "end_s", "scientific", "common", "score"])
        writer.writeheader()
        for path in iter_audio_files(args.audio):
            n_files += 1
            rows = predict_file(head, path, batch=args.batch)
            for r in rows:
                if r["score"] < args.threshold:
                    continue
                if present_labels is not None and r["label"] not in present_labels:
                    continue
                writer.writerow({
                    "file": path, "start_s": r["start_s"], "end_s": r["end_s"],
                    "scientific": r["scientific"], "common": r["common"], "score": round(r["score"], 4),
                })
                n_rows += 1
            print(f"  {path}: {len(rows)} (window,class) scored")

    print(f"\nScored {n_files} file(s); wrote {n_rows} row(s) >= threshold {args.threshold} -> {args.out}")


if __name__ == "__main__":
    main()
