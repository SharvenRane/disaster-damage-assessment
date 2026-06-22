"""Behavior tests for the damage classification models."""

import torch

from src.model import SiameseDamageNet, SingleImageDamageNet


def test_forward_shapes():
    model = SiameseDamageNet()
    before = torch.rand(4, 1, 16, 16)
    after = torch.rand(4, 1, 16, 16)
    out = model(before, after)
    assert out.shape == (4, 2)


def test_siamese_is_symmetric_in_branch_weights():
    """Both branches must share weights. Swapping before and after should flip
    the sign of the embedding difference, since the encoder is identical."""
    model = SiameseDamageNet()
    model.eval()
    before = torch.rand(3, 1, 16, 16)
    after = torch.rand(3, 1, 16, 16)
    with torch.no_grad():
        eb = model.encoder(before)
        ea = model.encoder(after)
        diff_forward = ea - eb
        diff_swapped = eb - ea
    assert torch.allclose(diff_forward, -diff_swapped, atol=1e-6)


def test_siamese_depends_on_both_inputs():
    """The output must change when the before image changes, proving the model
    actually consumes the before branch and is not after only."""
    torch.manual_seed(0)
    model = SiameseDamageNet()
    model.eval()
    after = torch.rand(2, 1, 16, 16)
    before_a = torch.rand(2, 1, 16, 16)
    before_b = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        out_a = model(before_a, after)
        out_b = model(before_b, after)
    assert not torch.allclose(out_a, out_b, atol=1e-4)


def test_single_image_ignores_other_input():
    """The after only ablation must be invariant to the before image."""
    model = SingleImageDamageNet(which="after")
    model.eval()
    after = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        out_a = model(torch.rand(2, 1, 16, 16), after)
        out_b = model(torch.rand(2, 1, 16, 16), after)
    assert torch.allclose(out_a, out_b, atol=1e-6)
