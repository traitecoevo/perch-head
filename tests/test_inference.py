"""Forward-pass and load round-trip checks for perch_head.inference.Head.

The scoring path is pure numpy (no Perch checkpoint needed), so these run offline. The
round-trip test also pins the fact that a saved head loads without pickle (allow_pickle
defaults to False), i.e. sharing a head .npz never executes code.
"""

import numpy as np

from perch_head.inference import Head


def _toy_head(l2norm=False):
    rng = np.random.default_rng(0)
    in_dim, hidden, n_classes = 6, 4, 3
    return Head(
        W1=rng.standard_normal((in_dim, hidden)).astype("float32"),
        b1=rng.standard_normal(hidden).astype("float32"),
        W2=rng.standard_normal((hidden, n_classes)).astype("float32"),
        b2=rng.standard_normal(n_classes).astype("float32"),
        labels=["Genus species_Common Name", "Aaa bbb_Cee", "Xxx yyy_Zzz"],
        is_present=np.array([True, False, True]),
        l2norm=l2norm,
    )


def _reference_forward(head, emb):
    x = emb
    if head.l2norm:
        x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    hidden = np.maximum(0.0, x @ head.W1 + head.b1)
    logits = hidden @ head.W2 + head.b2
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))


def test_forward_pass_matches_reference():
    head = _toy_head()
    emb = np.random.default_rng(1).standard_normal((5, 6)).astype("float32")
    got = head.predict_embeddings(emb)
    assert got.shape == (5, 3)
    np.testing.assert_allclose(got, _reference_forward(head, emb), rtol=1e-6, atol=1e-7)


def test_output_is_a_probability():
    head = _toy_head()
    emb = np.random.default_rng(2).standard_normal((8, 6)).astype("float32") * 100  # huge logits
    got = head.predict_embeddings(emb)
    assert np.all(got >= 0.0) and np.all(got <= 1.0)


def test_l2norm_changes_the_output():
    emb = np.random.default_rng(3).standard_normal((4, 6)).astype("float32")
    plain = _toy_head(l2norm=False).predict_embeddings(emb)
    normed = _toy_head(l2norm=True).predict_embeddings(emb)
    np.testing.assert_allclose(normed, _reference_forward(_toy_head(l2norm=True), emb),
                               rtol=1e-6, atol=1e-7)
    assert not np.allclose(plain, normed)


def test_save_load_round_trip_without_pickle(tmp_path):
    head = _toy_head(l2norm=True)
    path = tmp_path / "toy_head.npz"
    np.savez_compressed(
        path, W1=head.W1, b1=head.b1, W2=head.W2, b2=head.b2,
        labels=np.array(head.labels), is_present=head.is_present,
        l2norm=np.array(head.l2norm),
    )
    loaded = Head.load(str(path))  # Head.load uses np.load with allow_pickle=False
    assert loaded.labels == head.labels
    assert loaded.l2norm is True
    np.testing.assert_array_equal(loaded.is_present, head.is_present)
    emb = np.random.default_rng(4).standard_normal((3, 6)).astype("float32")
    np.testing.assert_allclose(loaded.predict_embeddings(emb), head.predict_embeddings(emb))
