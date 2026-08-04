"""Audio loading + windowing, self-contained (no BirdNET-Analyzer dependency).

`split_signal` closely follows BirdNET-Analyzer's `audio.split_signal` (overlap=0 case
only — the only configuration this project ever uses): non-overlapping `window_s`-second
chunks, zero-padded at the end (real signal at the front, trailing zeros) so every chunk is
the same length, with the final chunk dropped if less than `minlen_s` of real (non-padded)
signal remains. `open_audio_file` matches its resampling settings (librosa,
`res_type="kaiser_fast"`, mono).

One deliberate deviation from BirdNET-Analyzer: a clip whose *entire* signal is shorter than
`minlen_s` yields no windows here (returns `[]`), whereas BirdNET keeps a single padded chunk
for it. Dropping clips under `minlen_s` (default 1 s of real audio) is intended — too little
real signal to trust the embedding — so extraction skips them rather than training on a
mostly-zero window.

Keeping the rest of this feed path behavior-consistent matters: the already-trained heads in
this project (see docs/design_plan.md §3) were extracted through it — zero-pad the tail,
no peak-normalization, `kaiser_fast` resampling. A different resampler or padding scheme
would shift embeddings away from what the heads were trained on.
"""

from __future__ import annotations

import numpy as np

from perch_head.embed import SAMPLE_RATE, WINDOW_SECONDS

# The one place that decides which files the walkers pick up. It belongs next to
# `open_audio_file` because it is a claim about what librosa can decode, and it must not be
# narrower than that: an extension missing here is not an error anywhere, the file is simply
# never opened, so a whole source silently contributes zero windows.
#
# `.m4a`/`.mp4` matter in practice — iNaturalist serves most of its audio as `audio/mp4`,
# so a list without them drops most of an iNat batch without a word. Verified 2026-08-04:
# librosa 0.11.0 decodes .m4a and .mpga here at the correct duration (55 files checked
# against ffprobe, 0 mismatches).
AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mp4", ".mpga", ".aac", ".opus")


def open_audio_file(path: str, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at `sample_rate` Hz."""
    import librosa
    sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type="kaiser_fast")
    return sig.astype("float32"), rate


def split_signal(sig: np.ndarray, rate: int, window_s: float = WINDOW_SECONDS,
                  minlen_s: float = 1.0) -> list[np.ndarray]:
    """Non-overlapping `window_s`-second chunks; the tail chunk is zero-padded at the end
    (real signal first, trailing zeros). Drops a chunk if less than `minlen_s` of real signal
    remains in it — including the degenerate case where the whole clip is under `minlen_s`,
    which yields no windows at all (returns `[]`)."""
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
