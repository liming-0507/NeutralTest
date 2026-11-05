"""Pooling."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neutraltest.conformance.support import params, run_case
from neutraltest.seeding import randn
from neutraltest.testing import OpCase

F32_BF16 = (torch.float32, torch.bfloat16)


def _x(dtype: torch.dtype, generator: torch.Generator):
    return randn((2, 4, 8, 8), dtype, generator)


def _max_pool2d(dtype: torch.dtype, generator: torch.Generator):
    return (_x(dtype, generator),), {"kernel_size": 2}


def _max_pool2d_pad(dtype: torch.dtype, generator: torch.Generator):
    return (_x(dtype, generator),), {"kernel_size": 3, "stride": 2, "padding": 1}


def _avg_pool2d(dtype: torch.dtype, generator: torch.Generator):
    return (_x(dtype, generator),), {"kernel_size": 2}


def _adaptive_avg_pool2d(dtype: torch.dtype, generator: torch.Generator):
    return (_x(dtype, generator),), {"output_size": (2, 2)}


def _max_pool2d_indices(dtype: torch.dtype, generator: torch.Generator):
    x = _x(dtype, generator)
    return (x,), {"kernel_size": 2, "return_indices": True}


CASES = [
    OpCase(
        id="max-pool2d/k2",
        family="pooling",
        op="max_pool2d",
        fn=F.max_pool2d,
        build=_max_pool2d,
        dtypes=F32_BF16,
        kind="elementwise",
    ),
    OpCase(
        id="max-pool2d/k3-s2-p1",
        family="pooling",
        op="max_pool2d",
        fn=F.max_pool2d,
        build=_max_pool2d_pad,
        dtypes=F32_BF16,
        kind="elementwise",
    ),
    OpCase(
        id="avg-pool2d/k2",
        family="pooling",
        op="avg_pool2d",
        fn=F.avg_pool2d,
        build=_avg_pool2d,
        dtypes=F32_BF16,
        kind="reduction",
    ),
    OpCase(
        id="adaptive-avg-pool2d",
        family="pooling",
        op="adaptive_avg_pool2d",
        fn=F.adaptive_avg_pool2d,
        build=_adaptive_avg_pool2d,
        dtypes=F32_BF16,
        kind="reduction",
    ),
    OpCase(
        id="max-pool2d/indices",
        family="pooling",
        op="max_pool2d",
        fn=F.max_pool2d,
        build=_max_pool2d_indices,
        dtypes=(torch.float32,),
        kind="elementwise",
        out_dtypes=(None, torch.int64),
    ),
]


@pytest.mark.parametrize("case,dtype", params(CASES))
def test_pooling(case, dtype, backend):
    run_case(case, dtype, backend)
