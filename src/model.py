"""Models for before and after damage classification.

Two architectures share one small convolutional encoder:

SiameseDamageNet
    Runs the shared encoder over the before image and the after image
    separately, then classifies from the difference of their embeddings. This
    is the natural siamese form: identical weights on both branches, and the
    decision is driven by how the embedding moved between the two times.

SingleImageDamageNet
    An ablation that sees only one image (after by default). It uses the same
    encoder and head capacity, so any gap in accuracy against the siamese model
    comes from access to the pair, not from extra parameters. We use it in the
    tests to show the siamese model genuinely exploits both images.

The encoder is a real two layer CNN. There is no pretrained backbone here; the
whole network trains from scratch on the synthetic data, which keeps the tests
fully offline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Small CNN that maps a single channel image to an embedding vector."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        # Global average pool to a fixed length vector regardless of input size.
        x = x.mean(dim=(2, 3))
        return self.fc(x)


class SiameseDamageNet(nn.Module):
    """Siamese model that classifies from the before to after embedding shift."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.encoder = ConvEncoder(in_channels=1, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        eb = self.encoder(before)
        ea = self.encoder(after)
        diff = ea - eb
        return self.head(diff)


class SingleImageDamageNet(nn.Module):
    """Ablation that classifies from one image only (the after image)."""

    def __init__(self, embed_dim: int = 32, which: str = "after"):
        super().__init__()
        if which not in ("before", "after"):
            raise ValueError("which must be 'before' or 'after'")
        self.which = which
        self.encoder = ConvEncoder(in_channels=1, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        x = after if self.which == "after" else before
        emb = self.encoder(x)
        return self.head(emb)
