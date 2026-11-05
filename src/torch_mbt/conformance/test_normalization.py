"""Normalisation and the softmax family."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from torch_mbt.conformance.support import params, run_case
from torch_mbt.seeding import randn
from torch_mbt.testing import OpCase

F32 = (torch.float32,)
F32_BF16 = (torch.float32, torch.bfloat16)


def _layer_norm(dtype: torch.dtype, generator: torch.Generator):
    x = randn((4, 16), dtype, generator)
    weight = randn((16,), dtype, generator)
    bias = randn((16,), dtype, generator)
    return (x, (16,), weight, bias), {}


def _rms_norm(dtype: torch.dtype, generator: torch.Generator):
    return (randn((4, 16), dtype, generator), (16,), randn((16,), dtype, generator)), {}


def _rms_norm_op(*args, **kwargs):
    # F.rms_norm landed in torch 2.4; the test below skips on anything older.
    return F.rms_norm(*args, **kwargs)


HAS_RMS_NORM = hasattr(F, "rms_norm")


def _group_norm(dtype: torch.dtype, generator: torch.Generator):
    x = randn((2, 4, 8, 8), dtype, generator)
    return (x, 2), {}


def _batch_norm(dtype: torch.dtype, generator: torch.Generator):
    x = randn((2, 4, 8, 8), dtype, generator)
    running_mean = torch.zeros(4, dtype=dtype)
    running_var = torch.ones(4, dtype=dtype)
    weight = randn((4,), dtype, generator)
    bias = randn((4,), dtype, generator)
    return (x, running_mean, running_var, weight, bias, True, 0.1, 1e-5), {}


def _softmax(dtype: torch.dtype, generator: torch.Generator):
    return (randn((8, 16), dtype, generator),), {"dim": -1}


def _log_softmax(dtype: torch.dtype, generator: torch.Generator):
    return (randn((8, 16), dtype, generator),), {"dim": -1}


def _normalize(dtype: torch.dtype, generator: torch.Generator):
    return (randn((8, 16), dtype, generator),), {}


CASES = [
    OpCase(
        id="layer-norm",
        family="normalization",
        op="layer_norm",
        fn=F.layer_norm,
        build=_layer_norm,
        dtypes=F32_BF16,
        kind="normalization",
        grad=True,
        grad_inputs=(0, 2, 3),
    ),
    OpCase(
        id="group-norm",
        family="normalization",
        op="group_norm",
        fn=F.group_norm,
        build=_group_norm,
        dtypes=F32,
        kind="normalization",
    ),
    OpCase(
        id="batch-norm/training",
        family="normalization",
        op="batch_norm",
        fn=F.batch_norm,
        build=_batch_norm,
        dtypes=F32,
        kind="normalization",
    ),
    OpCase(
        id="softmax/dim-1",
        family="normalization",
        op="softmax",
        fn=F.softmax,
        build=_softmax,
        dtypes=F32_BF16,
        kind="normalization",
        grad=True,
        grad_inputs=(0,),
    ),
    OpCase(
        id="log-softmax/dim-1",
        family="normalization",
        op="log_softmax",
        fn=F.log_softmax,
        build=_log_softmax,
        dtypes=F32_BF16,
        kind="normalization",
    ),
    OpCase(
        id="normalize/l2",
        family="normalization",
        op="normalize",
        fn=F.normalize,
        build=_normalize,
        dtypes=F32_BF16,
        kind="normalization",
    ),
]


# Kept apart from CASES so the operator matrix stays collectable on old torch.
RMS_CASES = [
    OpCase(
        id="rms-norm",
        family="normalization",
        op="rms_norm",
        fn=_rms_norm_op,
        build=_rms_norm,
        dtypes=F32,
        kind="normalization",
    ),
]


@pytest.mark.parametrize("case,dtype", params(CASES))
def test_normalization(case, dtype, backend):
    run_case(case, dtype, backend)


@pytest.mark.skipif(not HAS_RMS_NORM, reason="F.rms_norm requires torch>=2.4")
@pytest.mark.parametrize("case,dtype", params(RMS_CASES))
def test_rms_norm(case, dtype, backend):
    run_case(case, dtype, backend)
