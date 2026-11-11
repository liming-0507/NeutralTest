"""B7 -- non-contiguous layouts.

Check: every column is given tensors that are views with a stride twice the
element size -- a column slice of a wider matrix, a transposed square, a strided
weight -- and its result is compared with the same call on a contiguous copy. A
kernel that reads through a raw pointer with the assumed stride, or that asks the
runtime for a contiguous buffer without materialising one, produces a different
answer; one that reads the tensor as the framework sees it does not.

Tolerance: exact. Both calls compute the same arithmetic on the same values, so
any difference is a layout bug rather than a rounding difference. Backward
columns additionally compare their analytic gradients between the strided and the
contiguous arguments at the same exact bound.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import close, flat, make_inputs, observe

ID = "B7"
COLUMNS = None

WIDTH = 16


def run(ctx: Any) -> None:
    args, kwargs = _strided(ctx)
    contiguous = tuple(
        value.contiguous() if isinstance(value, torch.Tensor) else value for value in args
    )
    if ctx.backward:
        from_strided, _ = ctx.gradients_with(args, kwargs)
        from_contiguous, _ = ctx.gradients_with(contiguous, kwargs)
        close(
            flat(from_strided),
            flat(from_contiguous),
            rel=0.0,
            msg=f"{ctx.column.name}: gradients depend on the layout",
        )
        return
    close(
        observe(ctx, args, kwargs),
        observe(ctx, contiguous, kwargs),
        rel=0.0,
        msg=f"{ctx.column.name}: the strided view and the contiguous copy disagree",
    )


def _strided(ctx: Any) -> tuple[Any, Any]:
    width = WIDTH * 2
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant, width=width)
    if ctx.family == "add":
        a, b = args
        return (a[:, ::2], b[::2]), kwargs
    if ctx.family == "flash_attention":
        return tuple(value[..., ::2] for value in args), kwargs
    if ctx.family == "kv_cache":
        keys, values, cache_k, cache_v, index = args
        return (keys[:, ::2], values[:, ::2], cache_k[:, ::2], cache_v[:, ::2], index), kwargs
    if ctx.family == "softmax":
        (x,) = args
        return (x.t(),), kwargs
    if ctx.family == "rope":
        x, position = args
        return (x[..., ::2], position), kwargs
    x, weight = args
    return (x[:, ::2], weight[::2]), kwargs
