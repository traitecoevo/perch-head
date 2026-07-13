"""Perch penultimate-embedding helper — shared by extraction, training-consistency checks,
and inference.

Perch's saved_model serving signature exposes both a native `label` (14795-class) head and
the penultimate `embedding` (1536-d) representation. This module pulls the latter, so a
custom head can be trained on it and scored against it later. Input geometry is fixed:
5 s @ 32 kHz = 160000 samples per window (see `WINDOW_SECONDS` / `SAMPLE_RATE`).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

SAMPLE_RATE = 32000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
EMBEDDING_DIM = 1536

_PERCH_EMBED_MODEL = None
_PERCH_MODEL_PATH = None


def default_checkpoint_path() -> str:
    """Perch v2 checkpoint bundled with the BirdNET-Analyzer fork.

    Requires `birdnet_analyzer` to be installed (editable) from
    https://github.com/wcornwell/BirdNET-Analyzer — see the top-level README.
    """
    import birdnet_analyzer.config as cfg
    return cfg.PERCH_V2_MODEL_PATH


def load(path: str | None = None) -> None:
    """Load (or reload, if `path` differs from what's cached) the Perch saved_model."""
    global _PERCH_EMBED_MODEL, _PERCH_MODEL_PATH
    path = path or default_checkpoint_path()
    if _PERCH_EMBED_MODEL is None or path != _PERCH_MODEL_PATH:
        _PERCH_EMBED_MODEL = tf.saved_model.load(path)
        _PERCH_MODEL_PATH = path


def perch_embed(data: np.ndarray, model_path: str | None = None) -> np.ndarray:
    """(n_windows, 160000) float32 -> (n_windows, 1536) Perch embeddings."""
    load(model_path)
    result = _PERCH_EMBED_MODEL.signatures["serving_default"](
        inputs=tf.constant(np.asarray(data, dtype="float32"))
    )
    return result["embedding"].numpy().astype("float32")
