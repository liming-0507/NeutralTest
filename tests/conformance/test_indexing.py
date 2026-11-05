"""Gather/scatter style operators.  Indices are generated on the CPU, so both
the device run and the reference see exactly the same ones."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from support import params, run_case

from torch_mbt.seeding import randn
from torch_mbt.testing import OpCase

F32_BF16 = (torch.float32, torch.bfloat16)


def _gather(dtype: torch.dtype, generator: torch.Generator):
    x = randn((4, 8), dtype, generator)
    index = torch.randint(0, 8, (4, 4), generator=generator)
    return (x, 1, index), {}


def _scatter(dtype: torch.dtype, generator: torch.Generator):
    x = randn((4, 8), dtype, generator)
    index = torch.randint(0, 8, (4, 4), generator=generator)
    src = randn((4, 4), dtype, generator)
    return (x, 1, index, src), {}


def _embedding(dtype: torch.dtype, generator: torch.Generator):
    weight = randn((10, 4), dtype, generator)
    index = torch.randint(0, 10, (3, 5), generator=generator)
    return (index, weight), {}


def _masked_fill(dtype: torch.dtype, generator: torch.Generator):
    x = randn((4, 8), dtype, generator)
    mask = randn((4, 8), dtype, generator) > 0
    return (x, mask, -1.0), {}


def _index_select(dtype: torch.dtype, generator: torch.Generator):
    x = randn((4, 8), dtype, generator)
    index = torch.randint(0, 4, (3,), generator=generator)
    return (x, 0, index), {}


CASES = [
    OpCase(
        id="gather/dim-1",
        family="indexing",
        op="gather",
        fn=torch.gather,
        build=_gather,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="scatter/dim-1",
        family="indexing",
        op="scatter",
        fn=torch.scatter,
        build=_scatter,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="embedding",
        family="indexing",
        op="embedding",
        fn=F.embedding,
        build=_embedding,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="masked-fill",
        family="indexing",
        op="masked_fill",
        fn=lambda t, mask, value: t.masked_fill(mask, value),
        build=_masked_fill,
        dtypes=F32_BF16,
        kind="exact",
    ),
    OpCase(
        id="index-select",
        family="indexing",
        op="index_select",
        fn=torch.index_select,
        build=_index_select,
        dtypes=F32_BF16,
        kind="exact",
    ),
]


@pytest.mark.parametrize("case,dtype", params(CASES))
def test_indexing(case, dtype, backend):
    run_case(case, dtype, backend)
