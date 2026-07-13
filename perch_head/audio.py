"""Audio loading + windowing, self-contained (no BirdNET-Analyzer dependency).

`split_signal` is a direct, faithful port of BirdNET-Analyzer's `audio.split_signal`
(overlap=0 case only — the only configuration this project ever uses): non-overlapping
`window_s`-second chunks, zero-padded and end-aligned so every chunk is the same length,
with the final chunk dropped if less than `minlen_s` of real (non-padded) signal remains.
`open_audio_file` matches its resampling settings (librosa, `res_type="kaiser_fast"`, mono).

Keeping this port exact matters: the already-trained heads in this project (see
docs/design_plan.md §3) were extracted through this exact feed path — zero-pad, end-align,
no peak-normalization, `kaiser_fast` resampling. A different resampler or padding scheme
would shift embeddings away from what the heads were trained on.
"""

from __future__ import annotations

import numpy as np

from perch_head.embed import SAMPLE_RATE, WINDOW_SECONDS


def open_audio_file(path: str, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at `sample_rate` Hz."""
    import librosa
    sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type="kaiser_fast")
    return sig.astype("float32"), rate


def split_signal(sig: np.ndarray, rate: int, window_s: float = WINDOW_SECONDS,
                  minlen_s: float = 1.0) -> list[np.ndarray]:
    """Non-overlapping `window_s`-second chunks, zero-padded/end-aligned. Drops the final
    chunk if less than `minlen_s` of real signal remains in it."""
    chunksize = int(rate * window_s)
    minsize = int(rate * minlen_s)
    lastchunkpos = int((sig.size - 1) / chunksize) * chunksize if sig.size else 0
    if lastchunkpos < 0:
        lastchunkpos = 0
    elif sig.size - lastchunkpos < minsize:
        lastchunkpos -= chunksize
    padded = np.concatenate([sig, np.zeros(chunksize, dtype=sig.dtype)])
    return [padded[i:i + chunksize] for i in range(0, lastchunkpos + 1, chunksize)]


def windows_for_file(path: str, sample_rate: int = SAMPLE_RATE, window_s: float = WINDOW_SECONDS,
                      minlen_s: float = 1.0) -> tuple[list[np.ndarray], list[float]]:
    """A single audio file -> (windows, start_times_s)."""
    sig, rate = open_audio_file(path, sample_rate)
    chunks = split_signal(sig, rate, window_s, minlen_s)
    starts = [i * window_s for i in range(len(chunks))]
    return chunks, starts
