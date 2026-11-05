"""Shape and data-movement operators.  Exact by construction: no arithmetic."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from torch_mbt.conformance.support import params, run_case
from torch_mbt.seeding import randn
from torch_mbt.testing import OpCase

F32_BF16 = (torch.float32, torch.bfloat16)


def _pair(dtype: torch.dtype, generator: torch.Generator):
    return (randn((4, 8), dtype, generator),), {}


def _cat(dtype: torch.dtype, generator: torch.Generator):
    a = randn((4, 8), dtype, generator)
    b = randn((4, 4), dtype, generator)
    return ((a, b),), {"dim": 1}


def _stack(dtype: torch.dtype, generator: torch.Generator):
    a = randn((4, 8), dtype, generator)
    b = randn((4, 8), dtype, generator)
    return ([a, b],), {"dim": 0}


def _split(dtype: torch.dtype, generator: torch.Generator):
    return (randn((4, 8), dtype, generator),), {"split_size_or_sections": 4, "dim": 1}


def _pad(dtype: torch.dtype, generator: torch.Generator):
    return (randn((2, 3, 4, 4), dtype, generator),), {"pad": (1, 1, 1, 1), "value": 0.5}


def _transpose(dtype: torch.dtype, generator: torch.Generator):
    return (randn((4, 8), dtype, generator),), {"dim0": 0, "dim1": 1}


def _permute(dtype: torch.dtype, generator: torch.Generator):
    return (randn((2, 4, 8), dtype, generator),), {"dims": (2, 0, 1)}


def _flip(dtype: torch.dtype, generator: torch.Generator):
    return (randn((4, 8), dtype, generator),), {"dims": [1]}


def _roll(dtype: torch.dtype, generator: torch.Generator):
    return (randn((4, 8), dtype, generator),), {"shifts": 2, "dims": 1}


CASES = [
    OpCase(
        id="cat/dim-1",
        family="shape",
        op="cat",
        fn=torch.cat,
        build=_cat,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="stack/dim-0",
        family="shape",
        op="stack",
        fn=torch.stack,
        build=_stack,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="split/chunks",
        family="shape",
        op="split",
        fn=torch.split,
        build=_split,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="pad/constant",
        family="shape",
        op="pad",
        fn=F.pad,
        build=_pad,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="transpose",
        family="shape",
        op="transpose",
        fn=torch.transpose,
        build=_transpose,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="permute",
        family="shape",
        op="permute",
        fn=torch.permute,
        build=_permute,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="reshape",
        family="shape",
        op="reshape",
        fn=lambda t: t.reshape(-1),
        build=_pair,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="flip/dim-1",
        family="shape",
        op="flip",
        fn=torch.flip,
        build=_flip,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="roll/dim-1",
        family="shape",
        op="roll",
        fn=torch.roll,
        build=_roll,
        dtypes=F32_BF16,
        kind="exact",
    ),
]


@pytest.mark.parametrize("case,dtype", params(CASES))
def test_manipulation(case, dtype, backend):
    run_case(case, dtype, backend)
