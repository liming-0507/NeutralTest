"""Matrix products."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from torch_mbt.conformance.support import params, run_case
from torch_mbt.seeding import randn
from torch_mbt.testing import OpCase

F32_BF16 = (torch.float32, torch.bfloat16)


def _mm(dtype: torch.dtype, generator: torch.Generator):
    return (randn((8, 16), dtype, generator), randn((16, 4), dtype, generator)), {}


def _broadcast_batch(dtype: torch.dtype, generator: torch.Generator):
    return (randn((2, 8, 16), dtype, generator), randn((16, 4), dtype, generator)), {}


def _bmm(dtype: torch.dtype, generator: torch.Generator):
    return (randn((3, 8, 16), dtype, generator), randn((3, 16, 4), dtype, generator)), {}


def _linear(dtype: torch.dtype, generator: torch.Generator):
    return (
        randn((8, 16), dtype, generator),
        randn((4, 16), dtype, generator),
        randn((4,), dtype, generator),
    ), {}


CASES = [
    OpCase(
        id="matmul/2d",
        family="matmul",
        op="matmul",
        fn=torch.matmul,
        build=_mm,
        dtypes=F32_BF16,
        kind="matmul",
        grad=True,
        grad_inputs=(0, 1),
    ),
    OpCase(
        id="matmul/broadcast-batch",
        family="matmul",
        op="matmul",
        fn=torch.matmul,
        build=_broadcast_batch,
        dtypes=F32_BF16,
        kind="matmul",
    ),
    OpCase(
        id="bmm",
        family="matmul",
        op="bmm",
        fn=torch.bmm,
        build=_bmm,
        dtypes=F32_BF16,
        kind="matmul",
    ),
    OpCase(
        id="linear",
        family="matmul",
        op="linear",
        fn=F.linear,
        build=_linear,
        dtypes=F32_BF16,
        kind="matmul",
        grad=True,
        grad_inputs=(0, 1, 2),
    ),
]


@pytest.mark.parametrize("case,dtype", params(CASES))
def test_matmul(case, dtype, backend):
    run_case(case, dtype, backend)
