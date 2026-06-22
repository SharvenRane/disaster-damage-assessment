"""Synthetic before and after image pairs with planted damage.

The generator produces a small dataset of paired satellite style image patches.
Each pair shows the same scene at two times. In a damaged pair the after image
has a planted damage signature: a textured rubble patch overlaid on part of the
building footprint, plus a localized darkening from debris and soot. In an
undamaged pair the after image differs only by nuisance changes (global
illumination shift, small additive sensor noise, slight registration jitter)
that are deliberately uncorrelated with the label.

The key property we rely on for testing is that the damage label is only
recoverable by comparing the two images. The before image alone and the after
image alone each carry no reliable signal about the label, because the base
scene content and the nuisance changes are drawn the same way regardless of
label. Only the planted *difference* between before and after encodes damage.
"""

from __future__ import annotations

import numpy as np


def _make_base_scene(rng: np.random.Generator, size: int) -> np.ndarray:
    """Build a single channel base scene with a building footprint.

    Returns a float array in [0, 1] of shape (size, size). The scene has a
    smooth ground gradient, a brighter rectangular building, and some texture.
    The building location and brightness are randomized so that no fixed pixel
    region is informative on its own.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # Smooth ground background with a random low frequency gradient.
    gx = rng.uniform(-1.0, 1.0)
    gy = rng.uniform(-1.0, 1.0)
    ground = 0.35 + 0.1 * (gx * (xx / size) + gy * (yy / size))
    scene = ground.astype(np.float32)

    # A brighter rectangular building footprint at a random position.
    bw = rng.integers(size // 4, size // 2)
    bh = rng.integers(size // 4, size // 2)
    bx = rng.integers(0, size - bw)
    by = rng.integers(0, size - bh)
    scene[by:by + bh, bx:bx + bw] += rng.uniform(0.25, 0.4)

    # Mild static texture so the scene is not flat.
    scene += 0.03 * rng.standard_normal((size, size)).astype(np.float32)

    return np.clip(scene, 0.0, 1.0), (bx, by, bw, bh)


def _apply_nuisance(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    """Apply label independent changes: illumination, noise, small jitter."""
    out = img.copy()
    # Global illumination shift.
    out = out * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05)
    # Small registration jitter via integer roll.
    sx = int(rng.integers(-1, 2))
    sy = int(rng.integers(-1, 2))
    out = np.roll(out, shift=(sy, sx), axis=(0, 1))
    # Additive sensor noise.
    out = out + 0.02 * rng.standard_normal(img.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def _plant_damage(rng: np.random.Generator, after: np.ndarray, footprint) -> np.ndarray:
    """Overlay a damage signature on part of the building footprint."""
    bx, by, bw, bh = footprint
    out = after.copy()
    # Damage patch covers a random sub region of the building.
    dw = max(2, int(bw * rng.uniform(0.4, 0.8)))
    dh = max(2, int(bh * rng.uniform(0.4, 0.8)))
    dx = bx + int(rng.integers(0, max(1, bw - dw)))
    dy = by + int(rng.integers(0, max(1, bh - dh)))

    region = out[dy:dy + dh, dx:dx + dw]
    # Rubble texture: high frequency speckle replacing smooth roof.
    rubble = 0.3 + 0.25 * rng.standard_normal(region.shape).astype(np.float32)
    # Debris darkening: pull the mean down.
    region = 0.5 * region + 0.5 * np.clip(rubble, 0.0, 1.0) - 0.1
    out[dy:dy + dh, dx:dx + dw] = np.clip(region, 0.0, 1.0)
    return out


def make_pair(rng: np.random.Generator, size: int = 16, damaged: bool = False):
    """Create one (before, after, label) sample.

    before and after are float32 arrays of shape (1, size, size) in [0, 1].
    label is 1 for damaged, 0 for undamaged.
    """
    base, footprint = _make_base_scene(rng, size)
    before = _apply_nuisance(rng, base)
    after = _apply_nuisance(rng, base)
    if damaged:
        after = _plant_damage(rng, after, footprint)
    label = 1 if damaged else 0
    return (
        before[None, :, :].astype(np.float32),
        after[None, :, :].astype(np.float32),
        label,
    )


def make_dataset(n: int = 200, size: int = 16, seed: int = 0):
    """Generate a balanced dataset of before and after pairs.

    Returns:
        before: float32 array (n, 1, size, size)
        after:  float32 array (n, 1, size, size)
        labels: int64 array (n,)
    """
    rng = np.random.default_rng(seed)
    befores = []
    afters = []
    labels = []
    for i in range(n):
        damaged = (i % 2 == 0)
        b, a, y = make_pair(rng, size=size, damaged=damaged)
        befores.append(b)
        afters.append(a)
        labels.append(y)
    before = np.stack(befores, axis=0).astype(np.float32)
    after = np.stack(afters, axis=0).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    return before, after, labels
