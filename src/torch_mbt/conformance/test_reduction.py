"""Reductions, including the ones that return indices."""

from __future__ import annotations

import pytest
import torch

from torch_mbt.conformance.support import params, run_case
from torch_mbt.seeding import randn
from torch_mbt.testing import OpCase

F32_BF16 = (torch.float32, torch.bfloat16)
SHAPE = (8, 16)


def _sum(dtype: torch.dtype, generator: torch.Generator):
    return (randn(SHAPE, dtype, generator),), {"dim": 1}


def _mean(dtype: torch.dtype, generator: torch.Generator):
    return (randn(SHAPE, dtype, generator),), {"dim": 1}


def _max(dtype: torch.dtype, generator: torch.Generator):
    return (randn(SHAPE, dtype, generator),), {"dim": 1}


def _amax(dtype: torch.dtype, generator: torch.Generator):
    return (randn(SHAPE, dtype, generator),), {"dim": 1}


def _topk(dtype: torch.dtype, generator: torch.Generator):
    return (randn(SHAPE, dtype, generator),), {"k": 4, "dim": 1}


def _argmax(dtype: torch.dtype, generator: torch.Generator):
    return (randn(SHAPE, dtype, generator),), {"dim": 1}


def _var(dtype: torch.dtype, generator: torch.Generator):
    return (randn(SHAPE, dtype, generator),), {"dim": 1}


def _logsumexp(dtype: torch.dtype, generator: torch.Generator):
    return (randn(SHAPE, dtype, generator),), {"dim": 1}


def _vector_norm(dtype: torch.dtype, generator: torch.Generator):
    return (randn(SHAPE, dtype, generator),), {"dim": 1}


CASES = [
    OpCase(
        id="sum/dim",
        family="reduction",
        op="sum",
        fn=torch.sum,
        build=_sum,
        dtypes=F32_BF16,
        kind="reduction",
        grad=True,
        grad_inputs=(0,),
    ),
    OpCase(
        id="mean/dim",
        family="reduction",
        op="mean",
        fn=torch.mean,
        build=_mean,
        dtypes=F32_BF16,
        kind="reduction",
    ),
    OpCase(
        id="amax/dim",
        family="reduction",
        op="amax",
        fn=torch.amax,
        build=_amax,
        dtypes=F32_BF16,
        kind="reduction",
    ),
    OpCase(
        id="max/dim",
        family="reduction",
        op="max",
        fn=torch.max,
        build=_max,
        dtypes=F32_BF16,
        kind="elementwise",
        out_dtypes=(None, torch.int64),
    ),
    OpCase(
        id="min/dim",
        family="reduction",
        op="min",
        fn=torch.min,
        build=_max,
        dtypes=F32_BF16,
        kind="elementwise",
        out_dtypes=(None, torch.int64),
    ),
    OpCase(
        id="argmax/dim",
        family="reduction",
        op="argmax",
        fn=torch.argmax,
        build=_argmax,
        dtypes=F32_BF16,
        kind="elementwise",
        out_dtypes=(torch.int64,),
    ),
    OpCase(
        id="topk",
        family="reduction",
        op="topk",
        fn=torch.topk,
        build=_topk,
        dtypes=F32_BF16,
        kind="elementwise",
        out_dtypes=(None, torch.int64),
    ),
    OpCase(
        id="cumsum/dim",
        family="reduction",
        op="cumsum",
        fn=lambda t, dim: torch.cumsum(t, dim=dim),
        build=_sum,
        dtypes=F32_BF16,
        kind="reduction",
    ),
    OpCase(
        id="var/dim",
        family="reduction",
        op="var",
        fn=torch.var,
        build=_var,
        dtypes=F32_BF16,
        kind="reduction",
    ),
    OpCase(
        id="logsumexp/dim",
        family="reduction",
        op="logsumexp",
        fn=torch.logsumexp,
        build=_logsumexp,
        dtypes=F32_BF16,
        kind="reduction",
    ),
    OpCase(
        id="l2-norm/dim",
        family="reduction",
        op="vector_norm",
        fn=torch.linalg.vector_norm,
        build=_vector_norm,
        dtypes=F32_BF16,
        kind="reduction",
    ),
]


@pytest.mark.parametrize("case,dtype", params(CASES))
def test_reduction(case, dtype, backend):
    run_case(case, dtype, backend)
