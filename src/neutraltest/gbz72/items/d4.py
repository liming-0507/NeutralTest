"""D4 -- the structural special cases.

Check: each family has one shape or calling convention that is not the plain
square case, and it is exercised here. The residual add must agree with the plain
add, a decode-style attention query of length one against a nine-key cache must
match the reference under both masking modes, softmax must be correct on a rank-3
tensor and along a size-one axis, and the rotation must accept an explicit
position table as well as a one-dimensional one, in both pair layouts.

Tolerance: the column's float32 budget against the float64 reference for every
value comparison, and exact equality where the two spellings are required to be
the same computation (residual add vs add, broadcast vs explicit positions).
"""

from __future__ import annotations

from typing import Any

import torch

from ...gbz73.inference import kernels
from .._support import budget, close, make_inputs, observe, reference

ID = "D4"
COLUMNS = None


def run(ctx: Any) -> None:
    if ctx.family == "add":
        _residual(ctx)
    elif ctx.family == "flash_attention":
        _attention(ctx)
    elif ctx.family == "softmax":
        _softmax(ctx)
    elif ctx.family == "rope":
        _rope(ctx)
    else:
        ctx.na(f"{ctx.column.name} has no structural special case in this clause")


def _residual(ctx: Any) -> None:
    args, kwargs = make_inputs(family="add", seed=ctx.seed)
    plain = observe(ctx, args, kwargs)[0]
    residual = kernels.residual_add(*args, **kwargs)
    if not torch.equal(residual, plain):
        raise AssertionError(f"{ctx.column.name}: residual_add disagrees with add")
    scaled = kwargs | {"alpha": 2.5}
    if not torch.equal(kernels.residual_add(*args, **scaled), observe(ctx, args, scaled)[0]):
        raise AssertionError(f"{ctx.column.name}: residual_add drops alpha")


def _attention(ctx: Any) -> None:
    generator = torch.Generator().manual_seed(ctx.seed)
    q = torch.randn(2, 2, 1, 8, generator=generator)
    k = torch.randn(2, 2, 9, 8, generator=generator)
    v = torch.randn(2, 2, 9, 8, generator=generator)
    for causal in (False, True):
        produced = kernels.flash_attention(q, k, v, is_causal=causal)
        close(
            produced,
            reference(ctx, (q, k, v), {"is_causal": causal}, dtype=torch.float64),
            rel=budget(ctx),
            msg=f"{ctx.column.name}: query length 1, key length 9, causal={causal}",
        )


def _softmax(ctx: Any) -> None:
    generator = torch.Generator().manual_seed(ctx.seed)
    for shape, dim in (
        ((2, 3, 4), -1),
        ((2, 3, 4), 1),
        ((4, 1, 7), 1),
        ((5, 6), 0),
    ):
        x = torch.randn(*shape, generator=generator)
        produced = kernels.softmax(x, dim=dim)
        close(
            produced,
            reference(ctx, (x,), {"dim": dim}, dtype=torch.float64),
            rel=budget(ctx),
            msg=f"{ctx.column.name}: shape {shape}, dim {dim}",
        )


def _rope(ctx: Any) -> None:
    args, _ = make_inputs(family="rope", seed=ctx.seed)
    x, position = args
    for interleaved in (True, False):
        close(
            kernels.rope(x, position, interleaved=interleaved),
            reference(ctx, (x, position), {"interleaved": interleaved}, dtype=torch.float64),
            rel=budget(ctx),
            msg=f"{ctx.column.name}: interleaved={interleaved}",
        )
    single = kernels.rope(x[:1], position[0])
    if not torch.equal(single, kernels.rope(x, position)[:1]):
        raise AssertionError(
            f"{ctx.column.name}: a one-dimensional position table was not honoured"
        )
