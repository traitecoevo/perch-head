"""Perch penultimate-embedding helper — shared by extraction, training-consistency checks,
and inference.

Perch's saved_model serving signature exposes both a native `label` (14795-class) head and
the penultimate `embedding` (1536-d) representation. This module pulls the latter, so a
custom head can be trained on it and scored against it later. Input geometry is fixed:
5 s @ 32 kHz = 160000 samples per window (see `WINDOW_SECONDS` / `SAMPLE_RATE`).

The checkpoint is fetched via `kagglehub` — no BirdNET-Analyzer dependency. Kaggle Models
handle taken from `google/bird-vocalization-classifier` (Perch v2, CPU-serving variant),
the same handle BirdNET-Analyzer itself downloads at `birdnet_analyzer/utils.py::
ensure_perch_exists()`. `kagglehub.model_download` caches locally after the first call, so
repeated runs don't re-download.
"""

from __future__ import annotations

import os

import numpy as np
import tensorflow as tf

SAMPLE_RATE = 32000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
EMBEDDING_DIM = 1536

KAGGLE_MODEL_HANDLE = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"
_REQUIRED_FILES = (
    "fingerprint.pb",
    "saved_model.pb",
    "variables/variables.index",
    "variables/variables.data-00000-of-00001",
    "assets/labels.csv",
    "assets/perch_v2_ebird_classes.csv",
)

_PERCH_EMBED_MODEL = None
_PERCH_MODEL_PATH = None


def _valid_checkpoint(path: str) -> bool:
    return all(os.path.exists(os.path.join(path, f)) for f in _REQUIRED_FILES)


def default_checkpoint_path() -> str:
    """Perch v2 checkpoint path.

    Honors the `PERCH_MODEL_PATH` env var if set (e.g. an already-downloaded local copy —
    useful to skip a redundant download if another tool on the same machine already fetched
    one). Otherwise fetches via `kagglehub.model_download(KAGGLE_MODEL_HANDLE)`, which
    caches locally after the first call.
    """
    env_path = os.environ.get("PERCH_MODEL_PATH")
    if env_path:
        if not _valid_checkpoint(env_path):
            raise FileNotFoundError(
                f"PERCH_MODEL_PATH={env_path!r} is missing required Perch checkpoint files "
                f"(expected e.g. {_REQUIRED_FILES[0]}, {_REQUIRED_FILES[-1]})."
            )
        return env_path
    import kagglehub
    try:
        return kagglehub.model_download(KAGGLE_MODEL_HANDLE)
    except Exception as exc:
        # A complete local cache is enough to run; kagglehub still calls the API to
        # resolve the version, so a dropped link would otherwise strand a machine
        # that already has the checkpoint. Prefer the newest cached version.
        cached = _newest_cached_checkpoint()
        if cached is None:
            raise
        print(f"  WARNING: kagglehub unreachable ({type(exc).__name__}); "
              f"using cached checkpoint {cached}")
        return cached


def _newest_cached_checkpoint() -> str | None:
    """Newest valid Perch checkpoint in the local kagglehub cache, if any."""
    root = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "models",
                        *KAGGLE_MODEL_HANDLE.split("/"))
    if not os.path.isdir(root):
        return None
    versions = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if name.isdigit() and os.path.isdir(p) and _valid_checkpoint(p):
            versions.append((int(name), p))
    return max(versions)[1] if versions else None


def load(path: str | None = None) -> None:
    """Load (or reload, if `path` differs from what's cached) the Perch saved_model.

    The early return is load-bearing, not an optimisation. `perch_embed` calls this
    on EVERY batch, and `default_checkpoint_path()` reaches the network --
    `kagglehub.model_download` contacts api.kaggle.com to resolve the version even
    when the local cache is complete. Resolving per batch therefore put a network
    round-trip in front of every embedding call and made a multi-hour extraction
    fail the moment the link dropped: a whole-library run died at class 308/448
    on 2026-08-09 with a DNS error, hours after the model was already in memory.
    Once the model is loaded and no explicit path is requested, there is nothing
    to re-resolve.
    """
    global _PERCH_EMBED_MODEL, _PERCH_MODEL_PATH
    if path is None and _PERCH_EMBED_MODEL is not None:
        return
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
