"""Run a trained head (from scripts/train_head.py) on audio: Perch embedding -> pure-numpy
forward pass -> per-window species probabilities. No Keras/TF graph needed at inference time
beyond the frozen Perch checkpoint itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from perch_head.audio import windows_for_file as _windows_for_file
from perch_head.embed import WINDOW_SECONDS, perch_embed

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


@dataclass
class Head:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    labels: list[str]
    is_present: np.ndarray
    l2norm: bool

    @classmethod
    def load(cls, path: str) -> "Head":
        d = np.load(path)
        return cls(
            W1=d["W1"], b1=d["b1"], W2=d["W2"], b2=d["b2"],
            labels=[str(x) for x in d["labels"]],
            is_present=d["is_present"],
            l2norm=bool(d["l2norm"]),
        )

    def predict_embeddings(self, emb: np.ndarray) -> np.ndarray:
        """(n, 1536) Perch embeddings -> (n, n_classes) sigmoid probabilities."""
        x = emb
        if self.l2norm:
            x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
        hidden = np.maximum(0.0, x @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))

    def predict_windows(self, windows: np.ndarray, batch: int = 64) -> np.ndarray:
        """(n, 160000) raw audio windows -> (n, n_classes) sigmoid probabilities."""
        embs = []
        for i in range(0, len(windows), batch):
            embs.append(perch_embed(np.asarray(windows[i:i + batch], dtype="float32")))
        emb = np.concatenate(embs, axis=0) if embs else np.zeros((0, self.W1.shape[0]), dtype="float32")
        return self.predict_embeddings(emb)


def windows_for_file(path: str, sig_minlen: float = 1.0) -> tuple[list[np.ndarray], list[float]]:
    """A single audio file -> (windows, start_times_s). Zero-pads the final short window,
    end-aligned, no peak-normalization — the same feed path embeddings were trained on."""
    chunks, starts = _windows_for_file(path, minlen_s=sig_minlen)
    return [np.asarray(c, dtype="float32") for c in chunks], starts


def predict_file(head: Head, path: str, batch: int = 64) -> list[dict]:
    """One audio file -> a list of {start_s, end_s, label, scientific, common, score} rows,
    one per (window, class) pair. Filter/threshold by score in the caller."""
    windows, starts = windows_for_file(path)
    if not windows:
        return []
    probs = head.predict_windows(np.asarray(windows), batch=batch)
    rows = []
    for w, start in enumerate(starts):
        for c, label in enumerate(head.labels):
            sci, _, common = label.partition("_")
            rows.append({
                "start_s": start,
                "end_s": start + WINDOW_SECONDS,
                "label": label,
                "scientific": sci,
                "common": common or sci,
                "score": float(probs[w, c]),
            })
    return rows


def iter_audio_files(path: str):
    if os.path.isfile(path):
        yield path
        return
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            if f.lower().endswith(AUDIO_EXT):
                yield os.path.join(root, f)
