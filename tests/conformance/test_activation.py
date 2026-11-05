"""Activation functions."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from support import params, run_case, unary

from torch_mbt.seeding import randn
from torch_mbt.testing import OpCase

F32_BF16 = (torch.float32, torch.bfloat16)


def _glu(dtype: torch.dtype, generator: torch.Generator):
    return (randn((8, 16), dtype, generator),), {"dim": -1}


def _prelu(dtype: torch.dtype, generator: torch.Generator):
    return (
        randn((8, 16), dtype, generator),
        (randn((16,), dtype, generator).abs() + 0.1),
    ), {}


CASES = [
    OpCase(
        id="relu",
        family="activation",
        op="relu",
        fn=torch.relu,
        build=unary(),
        dtypes=F32_BF16,
        grad=True,
        grad_inputs=(0,),
    ),
    OpCase(
        id="leaky-relu",
        family="activation",
        op="leaky_relu",
        fn=lambda t: F.leaky_relu(t, 0.01),
        build=unary(),
        dtypes=F32_BF16,
    ),
    OpCase(
        id="elu",
        family="activation",
        op="elu",
        fn=lambda t: F.elu(t, 1.0),
        build=unary(),
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="gelu",
        family="activation",
        op="gelu",
        fn=F.gelu,
        build=unary(),
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="sigmoid",
        family="activation",
        op="sigmoid",
        fn=torch.sigmoid,
        build=unary(),
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="silu",
        family="activation",
        op="silu",
        fn=F.silu,
        build=unary(),
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="softplus",
        family="activation",
        op="softplus",
        fn=F.softplus,
        build=unary(),
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="hardswish",
        family="activation",
        op="hardswish",
        fn=F.hardswish,
        build=unary(),
        dtypes=F32_BF16,
    ),
    OpCase(id="prelu", family="activation", op="prelu", fn=F.prelu, build=_prelu, dtypes=F32_BF16),
    OpCase(id="glu/dim-1", family="activation", op="glu", fn=F.glu, build=_glu, dtypes=F32_BF16),
]


@pytest.mark.parametrize("case,dtype", params(CASES))
def test_activation(case, dtype, backend):
    run_case(case, dtype, backend)
