"""End to end tests: damage classification beats chance, and using both images
beats using one."""

import torch

from src.data import make_dataset
from src.model import SiameseDamageNet, SingleImageDamageNet
from src.train import evaluate, to_tensors, train_model


def _splits():
    btr, atr, ytr = make_dataset(n=240, size=16, seed=0)
    bte, ate, yte = make_dataset(n=120, size=16, seed=99)
    return to_tensors(btr, atr, ytr), to_tensors(bte, ate, yte)


def test_siamese_beats_chance():
    (btr, atr, ytr), (bte, ate, yte) = _splits()
    model = train_model(SiameseDamageNet(), btr, atr, ytr, seed=0)
    acc = evaluate(model, bte, ate, yte)
    # Chance is 0.5 on a balanced two class problem. Require a clear margin.
    assert acc > 0.75, f"siamese accuracy {acc:.3f} did not beat chance"


def test_using_both_images_beats_single_image():
    """The siamese model, which sees before and after, must outperform the
    after only ablation that has the same capacity. This is the core claim:
    the damage signal lives in the change between the two images."""
    (btr, atr, ytr), (bte, ate, yte) = _splits()

    siamese = train_model(SiameseDamageNet(), btr, atr, ytr, seed=0)
    single = train_model(SingleImageDamageNet(which="after"), btr, atr, ytr, seed=0)

    acc_siamese = evaluate(siamese, bte, ate, yte)
    acc_single = evaluate(single, bte, ate, yte)

    assert acc_siamese > acc_single + 0.05, (
        f"siamese {acc_siamese:.3f} did not clearly beat single image "
        f"{acc_single:.3f}"
    )


def test_single_image_is_near_chance():
    """The after only model should be close to chance, confirming the dataset
    really hides the label from any single image."""
    (btr, atr, ytr), (bte, ate, yte) = _splits()
    single = train_model(SingleImageDamageNet(which="after"), btr, atr, ytr, seed=0)
    acc = evaluate(single, bte, ate, yte)
    assert acc < 0.70, f"single image accuracy {acc:.3f} was too high"
