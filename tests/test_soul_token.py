from __future__ import annotations

import torch

from soundpost.archings.soul import RemnantStateToken
from soundpost.settings import RstConfig


def _module() -> RemnantStateToken:
    torch.manual_seed(0)
    return RemnantStateToken(RstConfig(n_queries=4, n_heads=2, hbp_hidden=16), d_model=32).eval()


def test_output_shapes() -> None:
    module = _module()
    features = torch.randn(3, 8, 32)
    token, hbp = module(features)
    assert token.shape == (3, 32)
    assert hbp.shape == (3,)


def test_invariant_to_token_permutation() -> None:
    module = _module()
    features = torch.randn(2, 8, 32)
    perm = torch.randperm(8)
    token_a, hbp_a = module(features)
    token_b, hbp_b = module(features[:, perm])
    assert torch.allclose(token_a, token_b, atol=1e-5)
    assert torch.allclose(hbp_a, hbp_b, atol=1e-5)
