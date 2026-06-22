"""Training and evaluation helpers shared by the demo and the tests."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def to_tensors(before: np.ndarray, after: np.ndarray, labels: np.ndarray):
    return (
        torch.from_numpy(before).float(),
        torch.from_numpy(after).float(),
        torch.from_numpy(labels).long(),
    )


def train_model(
    model: nn.Module,
    before: torch.Tensor,
    after: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 40,
    lr: float = 1e-2,
    batch_size: int = 32,
    seed: int = 0,
) -> nn.Module:
    """Train a damage model in place and return it.

    Works for both SiameseDamageNet and SingleImageDamageNet since both take
    (before, after) and return class logits.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = before.shape[0]
    g = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            logits = model(before[idx], after[idx])
            loss = loss_fn(logits, labels[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    before: torch.Tensor,
    after: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Return classification accuracy on the given split."""
    model.eval()
    logits = model(before, after)
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()
