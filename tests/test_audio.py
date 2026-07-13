"""Windowing invariants for perch_head.audio.split_signal.

These guard the feed-path behavior the trained heads depend on (see docs/design_plan.md §3):
trailing zero-pad, and the deliberate drop of clips shorter than minlen_s. No Perch checkpoint
or network access is needed — split_signal is pure numpy.
"""

import numpy as np
import pytest

from perch_head.audio import split_signal
from perch_head.embed import SAMPLE_RATE, WINDOW_SAMPLES

RATE = SAMPLE_RATE


@pytest.mark.parametrize(
    "clip_s, expected_windows",
    [
        (0.5, 0),    # whole clip under minlen_s -> dropped entirely (intended deviation)
        (0.9, 0),    # ditto
        (1.0, 1),    # exactly minlen_s of real signal -> kept, padded
        (1.25, 1),
        (3.0, 1),    # the common training-clip case: 3 s audio + 2 s zeros
        (5.0, 1),    # exactly one window
        (6.25, 2),   # spills into a second window with 1.25 s real signal -> kept
        (5.3, 1),    # second window has only 0.3 s real signal (< minlen_s) -> dropped
        (10.0, 2),
        (15.0, 3),
    ],
)
def test_window_count(clip_s, expected_windows):
    sig = np.ones(int(clip_s * RATE), dtype=np.float32)
    assert len(split_signal(sig, RATE)) == expected_windows


def test_empty_signal_yields_no_windows():
    assert split_signal(np.zeros(0, dtype=np.float32), RATE) == []


def test_every_window_is_full_length():
    sig = np.ones(int(6.25 * RATE), dtype=np.float32)
    for w in split_signal(sig, RATE):
        assert w.shape == (WINDOW_SAMPLES,)


def test_trailing_pad_signal_at_front_zeros_at_end():
    # 3 s of ones -> one window: 3 s real signal at the front, 2 s of zeros appended.
    real = int(3.0 * RATE)
    sig = np.ones(real, dtype=np.float32)
    (window,) = split_signal(sig, RATE)
    assert np.all(window[:real] == 1.0)
    assert np.all(window[real:] == 0.0)


def test_minlen_boundary_is_inclusive():
    # A clip with exactly minlen_s of real signal is kept; a hair under is dropped.
    assert len(split_signal(np.ones(RATE, dtype=np.float32), RATE, minlen_s=1.0)) == 1
    assert len(split_signal(np.ones(RATE - 1, dtype=np.float32), RATE, minlen_s=1.0)) == 0
