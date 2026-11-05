"""Fused attention.  Kept in its own module because it is the one operator whose
tolerance class is materially looser than everything else."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from support import params, run_case

from torch_mbt.seeding import randn
from torch_mbt.testing import OpCase


def _sdpa(dtype: torch.dtype, generator: torch.Generator):
    query = randn((2, 4, 8, 16), dtype, generator)
    key = randn((2, 4, 8, 16), dtype, generator)
    value = randn((2, 4, 8, 16), dtype, generator)
    return (query, key, value), {}


def _sdpa_causal(dtype: torch.dtype, generator: torch.Generator):
    query = randn((2, 2, 8, 16), dtype, generator)
    key = randn((2, 2, 8, 16), dtype, generator)
    value = randn((2, 2, 8, 16), dtype, generator)
    return (query, key, value), {"is_causal": True}


CASES = [
    OpCase(
        id="sdpa/full",
        family="attention",
        op="scaled_dot_product_attention",
        fn=F.scaled_dot_product_attention,
        build=_sdpa,
        dtypes=(torch.float32,),
        kind="attention",
        grad=True,
        grad_inputs=(0, 1, 2),
    ),
    OpCase(
        id="sdpa/causal",
        family="attention",
        op="scaled_dot_product_attention",
        fn=F.scaled_dot_product_attention,
        build=_sdpa_causal,
        dtypes=(torch.float32,),
        kind="attention",
    ),
]


@pytest.mark.parametrize("case,dtype", params(CASES))
def test_attention(case, dtype, backend):
    run_case(case, dtype, backend)
