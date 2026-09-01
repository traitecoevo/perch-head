"""Head.load's score-ensemble branch, and EnsembleHead's forward pass.

No network and no Perch checkpoint needed: `perch_embed_and_label` and `stock_perch_labels`
are monkeypatched, since the only thing under test here is the WIRING between a
score-ensemble artifact and perch-head's inference interface, not Perch itself.
"""

import numpy as np
import pytest
from score_ensemble.artifact import write_artifact

from perch_head import inference as inf
from perch_head.inference import EnsembleHead, Head

# Matches EnsembleHead.load's own key: the head's labels are 'Genus species_Common Name',
# stock Perch's are plain binomials. write_artifact and bind_partner must agree on this, or
# `expect_shared` (computed at write time) describes a different join than the one that runs.
_STRIP_COMMON_NAME = lambda s: s.split("_", 1)[0].strip()


def _fake_head_npz(path, labels, n_hidden=4, in_dim=6):
    n = len(labels)
    rng = np.random.default_rng(0)
    np.savez_compressed(
        path,
        W1=rng.standard_normal((in_dim, n_hidden)).astype("float32"),
        b1=np.zeros(n_hidden, "float32"),
        W2=rng.standard_normal((n_hidden, n)).astype("float32"),
        b2=np.zeros(n, "float32"),
        labels=np.array(labels), is_present=np.ones(n, dtype=bool),
        l2norm=np.array(False),
    )


def test_head_load_detects_a_plain_head_without_pickle(tmp_path):
    path = tmp_path / "plain.npz"
    _fake_head_npz(path, ["Genus a_Common A"])
    head = Head.load(str(path))
    assert isinstance(head, Head)


def test_head_load_detects_an_ensemble_artifact_and_returns_ensemble_head(tmp_path, monkeypatch):
    head_path = tmp_path / "head.npz"
    labels = ["Genus a_Common A", "Genus b_Common B"]
    _fake_head_npz(head_path, labels)
    # Realistic: stock Perch's own labels are PLAIN binomials, no common-name suffix -- unlike
    # the head's own labels. A test that gives the partner suffixed labels too would not have
    # caught the real bug (bind_partner joining raw strings, which matched almost nothing).
    partner_labels = ["Genus a", "Genus z", "Genus b"]

    art_path = tmp_path / "head-ens.npz"
    write_artifact(str(art_path), head_npz_path=str(head_path), blend_with="perch",
                   combiner="gmean", columns_from="head", head_space="logit",
                   partner_space="probability", output_space="logit",
                   partner_labels=partner_labels, key=_STRIP_COMMON_NAME)

    monkeypatch.setattr(inf, "stock_perch_labels", lambda: partner_labels)

    loaded = Head.load(str(art_path))
    assert isinstance(loaded, EnsembleHead)
    assert loaded.labels == labels
    np.testing.assert_array_equal(loaded.is_present, np.ones(2, dtype=bool))


def test_ensemble_head_predict_windows_combines_both_taps(tmp_path, monkeypatch):
    head_path = tmp_path / "head.npz"
    labels = ["Genus a_Common A", "Genus b_Common B"]
    _fake_head_npz(head_path, labels)
    # Realistic: stock Perch's own labels are PLAIN binomials, no common-name suffix -- unlike
    # the head's own labels. A test that gives the partner suffixed labels too would not have
    # caught the real bug (bind_partner joining raw strings, which matched almost nothing).
    partner_labels = ["Genus a", "Genus z", "Genus b"]

    art_path = tmp_path / "head-ens.npz"
    write_artifact(str(art_path), head_npz_path=str(head_path), blend_with="perch",
                   combiner="max", columns_from="head", head_space="logit",
                   partner_space="probability", output_space="logit",
                   partner_labels=partner_labels, key=_STRIP_COMMON_NAME)

    monkeypatch.setattr(inf, "stock_perch_labels", lambda: partner_labels)
    loaded = Head.load(str(art_path))

    n_windows = 3
    fake_emb = np.random.default_rng(1).standard_normal((n_windows, 6)).astype("float32")
    fake_partner_probs = np.random.default_rng(2).uniform(
        0.01, 0.99, size=(n_windows, len(partner_labels))).astype("float32")

    def fake_embed_and_label(windows):
        return fake_emb[:len(windows)], fake_partner_probs[:len(windows)]

    monkeypatch.setattr(inf, "perch_embed_and_label", fake_embed_and_label)

    windows = np.zeros((n_windows, 160000), dtype="float32")
    probs = loaded.predict_windows(windows, batch=64)

    assert probs.shape == (n_windows, len(labels))
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Reference: run the SAME arithmetic score_ensemble.EnsembleScorer runs, independently.
    from score_ensemble.runtime import EnsembleScorer
    scorer = EnsembleScorer.load(str(art_path))
    scorer.bind_partner(partner_labels, key=_STRIP_COMMON_NAME)
    head_logits = scorer.head_logits(fake_emb)
    want_logits = scorer.score_from_taps(head_logits, fake_partner_probs)
    want_probs = 1.0 / (1.0 + np.exp(-np.clip(want_logits, -20, 20)))
    np.testing.assert_array_equal(probs, want_probs)


def test_ensemble_head_rejects_non_perch_partner(tmp_path, monkeypatch):
    head_path = tmp_path / "head.npz"
    _fake_head_npz(head_path, ["Genus a_Common A"])
    art_path = tmp_path / "bad.npz"
    with pytest.raises(NotImplementedError):
        write_artifact(str(art_path), head_npz_path=str(head_path), blend_with="birdnet-global",
                       combiner="gmean", columns_from="head", head_space="logit",
                       partner_space="birdnet_flat", output_space="logit",
                       partner_labels=["Sp A"])
