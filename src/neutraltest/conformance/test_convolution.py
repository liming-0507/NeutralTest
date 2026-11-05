"""Convolutions."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neutraltest.conformance.support import params, run_case
from neutraltest.seeding import randn
from neutraltest.testing import OpCase

F32_BF16 = (torch.float32, torch.bfloat16)
F32 = (torch.float32,)


def _conv2d(dtype: torch.dtype, generator: torch.Generator):
    x = randn((2, 3, 8, 8), dtype, generator)
    weight = randn((4, 3, 3, 3), dtype, generator)
    bias = randn((4,), dtype, generator)
    return (x, weight, bias), {"stride": 1, "padding": 1}


def _conv2d_groups(dtype: torch.dtype, generator: torch.Generator):
    x = randn((2, 4, 8, 8), dtype, generator)
    weight = randn((4, 2, 3, 3), dtype, generator)
    return (x, weight, None), {"stride": 1, "padding": 0, "groups": 2}


def _conv2d_strided(dtype: torch.dtype, generator: torch.Generator):
    x = randn((1, 3, 9, 9), dtype, generator)
    weight = randn((2, 3, 3, 3), dtype, generator)
    return (x, weight, None), {"stride": 2, "padding": 1}


def _conv_transpose2d(dtype: torch.dtype, generator: torch.Generator):
    x = randn((2, 3, 4, 4), dtype, generator)
    weight = randn((3, 2, 3, 3), dtype, generator)
    bias = randn((2,), dtype, generator)
    return (x, weight, bias), {"stride": 1, "padding": 0}


def _conv1d(dtype: torch.dtype, generator: torch.Generator):
    x = randn((2, 3, 10), dtype, generator)
    weight = randn((4, 3, 3), dtype, generator)
    return (x, weight, None), {"stride": 1, "padding": 1}


CASES = [
    OpCase(
        id="conv2d/pad1",
        family="convolution",
        op="conv2d",
        fn=F.conv2d,
        build=_conv2d,
        dtypes=F32_BF16,
        kind="convolution",
        grad=True,
        grad_inputs=(0, 1, 2),
    ),
    OpCase(
        id="conv2d/groups-2",
        family="convolution",
        op="conv2d",
        fn=F.conv2d,
        build=_conv2d_groups,
        dtypes=F32,
        kind="convolution",
    ),
    OpCase(
        id="conv2d/stride-2",
        family="convolution",
        op="conv2d",
        fn=F.conv2d,
        build=_conv2d_strided,
        dtypes=F32_BF16,
        kind="convolution",
    ),
    OpCase(
        id="conv-transpose2d",
        family="convolution",
        op="conv_transpose2d",
        fn=F.conv_transpose2d,
        build=_conv_transpose2d,
        dtypes=F32,
        kind="convolution",
    ),
    OpCase(
        id="conv1d/pad1",
        family="convolution",
        op="conv1d",
        fn=F.conv1d,
        build=_conv1d,
        dtypes=F32_BF16,
        kind="convolution",
    ),
]


@pytest.mark.parametrize("case,dtype", params(CASES))
def test_convolution(case, dtype, backend):
    run_case(case, dtype, backend)
