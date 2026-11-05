"""Elementwise binary and unary operators."""

from __future__ import annotations

import pytest
import torch

from neutraltest.conformance.support import binary, params, positive, run_case, unary
from neutraltest.seeding import randn
from neutraltest.testing import OpCase

F32_BF16 = (torch.float32, torch.bfloat16)


def _pow(dtype: torch.dtype, generator: torch.Generator):
    base = randn((8, 16), dtype, generator).abs() + 0.5
    return (base, 3.0), {}


def _where(dtype: torch.dtype, generator: torch.Generator):
    condition = randn((8, 16), dtype, generator) > 0
    return (
        condition,
        randn((8, 16), dtype, generator),
        randn((16,), dtype, generator),
    ), {}


CASES = [
    OpCase(
        id="add/broadcast",
        family="elementwise",
        op="add",
        fn=torch.add,
        build=binary(),
        dtypes=F32_BF16,
        grad=True,
        grad_inputs=(0, 1),
    ),
    OpCase(
        id="sub/broadcast",
        family="elementwise",
        op="sub",
        fn=torch.sub,
        build=binary(),
        dtypes=F32_BF16,
    ),
    OpCase(
        id="mul/broadcast",
        family="elementwise",
        op="mul",
        fn=torch.mul,
        build=binary(),
        dtypes=F32_BF16,
    ),
    OpCase(
        id="div/broadcast",
        family="elementwise",
        op="div",
        fn=torch.div,
        build=binary(),
        dtypes=F32_BF16,
    ),
    OpCase(
        id="maximum/broadcast",
        family="elementwise",
        op="maximum",
        fn=torch.maximum,
        build=binary(),
        dtypes=F32_BF16,
    ),
    OpCase(id="abs", family="elementwise", op="abs", fn=torch.abs, build=unary(), dtypes=F32_BF16),
    OpCase(id="neg", family="elementwise", op="neg", fn=torch.neg, build=unary(), dtypes=F32_BF16),
    OpCase(
        id="pow/scalar",
        family="elementwise",
        op="pow",
        fn=torch.pow,
        build=_pow,
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="exp",
        family="elementwise",
        op="exp",
        fn=torch.exp,
        build=unary(),
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="log",
        family="elementwise",
        op="log",
        fn=torch.log,
        build=positive(),
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="sqrt",
        family="elementwise",
        op="sqrt",
        fn=torch.sqrt,
        build=positive(),
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="sin",
        family="elementwise",
        op="sin",
        fn=torch.sin,
        build=unary(),
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="cos",
        family="elementwise",
        op="cos",
        fn=torch.cos,
        build=unary(),
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="tanh",
        family="elementwise",
        op="tanh",
        fn=torch.tanh,
        build=unary(),
        dtypes=F32_BF16,
        kind="transcendental",
    ),
    OpCase(
        id="clamp",
        family="elementwise",
        op="clamp",
        fn=lambda t: torch.clamp(t, min=-0.5, max=0.5),
        build=unary(),
        dtypes=F32_BF16,
    ),
    OpCase(
        id="where/broadcast",
        family="elementwise",
        op="where",
        fn=torch.where,
        build=_where,
        dtypes=F32_BF16,
    ),
]


@pytest.mark.parametrize("case,dtype", params(CASES))
def test_elementwise(case, dtype, backend):
    run_case(case, dtype, backend)
