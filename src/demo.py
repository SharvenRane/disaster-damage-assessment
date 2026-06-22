"""Runnable demo: train the siamese model and the single image ablation.

Run with: python -m src.demo
"""

from __future__ import annotations

from .data import make_dataset
from .model import SiameseDamageNet, SingleImageDamageNet
from .train import evaluate, to_tensors, train_model


def main() -> None:
    btr, atr, ytr = make_dataset(n=240, size=16, seed=0)
    bte, ate, yte = make_dataset(n=120, size=16, seed=99)

    btr_t, atr_t, ytr_t = to_tensors(btr, atr, ytr)
    bte_t, ate_t, yte_t = to_tensors(bte, ate, yte)

    siamese = train_model(SiameseDamageNet(), btr_t, atr_t, ytr_t, seed=0)
    single = train_model(SingleImageDamageNet(which="after"), btr_t, atr_t, ytr_t, seed=0)

    acc_siamese = evaluate(siamese, bte_t, ate_t, yte_t)
    acc_single = evaluate(single, bte_t, ate_t, yte_t)

    print(f"siamese (uses both images) test accuracy: {acc_siamese:.3f}")
    print(f"single image (after only) test accuracy:  {acc_single:.3f}")


if __name__ == "__main__":
    main()
