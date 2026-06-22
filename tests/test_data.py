"""Tests for the synthetic before and after pair generator."""

import numpy as np

from src.data import make_dataset, make_pair


def test_dataset_shapes_and_balance():
    before, after, labels = make_dataset(n=40, size=16, seed=1)
    assert before.shape == (40, 1, 16, 16)
    assert after.shape == (40, 1, 16, 16)
    assert labels.shape == (40,)
    assert before.dtype == np.float32
    # Balanced labels.
    assert labels.sum() == 20
    # Pixel values stay in [0, 1].
    assert before.min() >= 0.0 and before.max() <= 1.0
    assert after.min() >= 0.0 and after.max() <= 1.0


def test_damage_changes_the_after_image_more():
    """A damaged pair should show a larger before to after difference than an
    undamaged pair on average, since damage plants a real signature."""
    rng = np.random.default_rng(7)
    dmg_diffs = []
    clean_diffs = []
    for _ in range(50):
        b, a, _ = make_pair(rng, size=16, damaged=True)
        dmg_diffs.append(np.abs(a - b).mean())
        b, a, _ = make_pair(rng, size=16, damaged=False)
        clean_diffs.append(np.abs(a - b).mean())
    assert np.mean(dmg_diffs) > np.mean(clean_diffs)


def test_label_not_recoverable_from_single_image_means():
    """The label must not be trivially encoded in a single image. The mean
    brightness of the after image alone should not separate the classes by
    much, so a single image model has to work for its signal."""
    before, after, labels = make_dataset(n=200, size=16, seed=3)
    after_means = after.reshape(len(labels), -1).mean(axis=1)
    pos = after_means[labels == 1].mean()
    neg = after_means[labels == 0].mean()
    # The two class means of after brightness are close.
    assert abs(pos - neg) < 0.05
