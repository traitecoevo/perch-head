"""Row algebra and guards for scripts/splice_cache.py.

The splice replaces exactly the changed classes' rows in a training cache and leaves every
other row -- crucially including its train/val assignment -- untouched. Both halves matter:
if it dropped the wrong rows the model silently trains on stale vectors, and if it re-drew
`split` the unchanged classes would stop being a valid negative control, which is the whole
reason for splicing instead of re-extracting.

Pure numpy: no Perch checkpoint, no audio. The end-to-end check that the SUBSET EXTRACTION
reproduces the control's vectors is a separate manual pass (see the module docstring) --
it needs the checkpoint and minutes of compute, so it does not belong here.
"""

import importlib.util
import os

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "splice_cache",
    os.path.join(os.path.dirname(__file__), os.pardir, "scripts", "splice_cache.py"),
)
splice_cache = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(splice_cache)


N_CLASSES = 4
DIM = 3


def _cache(rows, labels=None, split=None):
    """rows = list of (class_index_or_None, fill_value); None means an all-zero non-event row."""
    X = np.array([[v] * DIM for _, v in rows], dtype="float32")
    Y = np.zeros((len(rows), N_CLASSES), dtype="float32")
    for i, (c, _) in enumerate(rows):
        if c is not None:
            Y[i, c] = 1.0
    return {
        "X": X,
        "Y": Y,
        "labels": np.array(labels if labels is not None else [f"c{i}" for i in range(N_CLASSES)]),
        "is_present": np.ones(N_CLASSES, dtype=bool),
        "split": np.array(split if split is not None else ["train"] * len(rows)),
    }


def test_replaces_only_the_changed_class_rows():
    control = _cache([(0, 1.0), (1, 2.0), (1, 2.5), (2, 3.0), (None, 9.0)])
    subset = _cache([(1, 7.0), (1, 7.5), (1, 7.75)])

    out = splice_cache.splice(control, subset)

    # class 1's two control rows are gone, its three subset rows are present
    assert sorted(out["X"][out["Y"][:, 1] == 1][:, 0]) == [7.0, 7.5, 7.75]
    # every other row survives untouched, non-event row included
    assert sorted(out["X"][out["Y"].sum(1) == 0][:, 0]) == [9.0]
    assert sorted(out["X"][out["Y"][:, 0] == 1][:, 0]) == [1.0]
    assert sorted(out["X"][out["Y"][:, 2] == 1][:, 0]) == [3.0]
    assert len(out["X"]) == 3 + 3


def test_retained_rows_keep_their_original_split():
    # the reason this script exists: a full re-extract re-draws every row's split
    control = _cache(
        [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)],
        split=["val", "train", "val", "train"],
    )
    subset = _cache([(1, 7.0)])

    out = splice_cache.splice(control, subset)

    retained = {x: s for x, s in zip(out["X"][:3, 0], out["split"][:3])}
    assert retained == {1.0: "val", 3.0: "val", 4.0: "train"}


def test_multiple_changed_classes_are_all_replaced():
    control = _cache([(0, 1.0), (1, 2.0), (2, 3.0)])
    subset = _cache([(0, 8.0), (2, 9.0)])

    out = splice_cache.splice(control, subset)

    assert np.array_equal(np.sort(splice_cache.changed_columns(subset)), np.array([0, 2]))
    assert sorted(out["X"][:, 0]) == [2.0, 8.0, 9.0]


def test_rejects_non_event_rows_in_the_subset():
    # an all-zero row's source folder is unrecoverable from Y, so it can never be spliced
    control = _cache([(0, 1.0)])
    subset = _cache([(0, 8.0), (None, 9.0)])

    with pytest.raises(SystemExit, match="non-event"):
        splice_cache.splice(control, subset)


def test_rejects_a_reordered_or_different_vocabulary():
    control = _cache([(0, 1.0)])
    subset = _cache([(0, 8.0)], labels=["c1", "c0", "c2", "c3"])

    with pytest.raises(SystemExit, match="label vocabularies differ"):
        splice_cache.splice(control, subset)


def test_rejects_multi_hot_rows():
    control = _cache([(0, 1.0)])
    subset = _cache([(0, 8.0)])
    subset["Y"][0, 2] = 1.0

    with pytest.raises(SystemExit, match="multi-hot"):
        splice_cache.splice(control, subset)


def test_rejects_an_empty_subset():
    with pytest.raises(SystemExit, match="no rows"):
        splice_cache.splice(_cache([(0, 1.0)]), _cache([]))


def test_new_rows_get_the_same_val_fraction():
    control = _cache([(0, 1.0)])
    subset = _cache([(1, float(i)) for i in range(100)])

    out = splice_cache.splice(control, subset, seed=0)

    new = out["split"][1:]
    assert (new == "val").sum() == int(round(splice_cache.VAL_FRACTION * 100))
