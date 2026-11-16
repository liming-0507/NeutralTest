"""E7 -- the operator runs on the deployment it was measured on.

Deployment is not a version string. The clause asks whether what ships can be
asked to run again and give the same answer; recording the versions and the
device the answer came from is part of that, but only part -- a row that printed
versions and called nothing would pass on a machine where the operator does not
load at all, which is one condition this row exists to catch.

So the row calls the column twice on the same inputs and requires the same bits,
then records the environment the two calls came from. Bit equality is right here:
this is the same process, the same device, the same arguments, so a difference is
hidden state the deployment carries between calls -- a cached workspace, a
counter, an uninitialised buffer, a descriptor built on the first call and reused
wrongly on the second.

What the note records is what a failure would need to be reproduced: the Python
and torch builds, the backend and its device type, the device index, and the
data dtype. It is written whether the row passes or fails, because a pass without
the environment is not evidence about a deployment.

Tolerance: none. The two calls are the same computation on the same operands.
"""

from __future__ import annotations

import platform
from typing import Any

import torch

from .._support import make_inputs

ID = "E7"
COLUMNS = None


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    first = _answer(ctx, args, kwargs)
    if not _finite(first):
        raise AssertionError(
            f"{ctx.column.name}: the first deployment call produced non-finite values, so there "
            f"is no answer to compare the second call with"
        )
    second = _answer(ctx, args, kwargs)
    if not _same(first, second):
        raise AssertionError(
            f"{ctx.column.name}: two deployment calls on the same inputs differ by up to "
            f"{_worst(first, second):.3e}, so the deployment carries state between calls"
        )
    ctx.note(
        f"repeatable on this deployment: two calls on the same inputs are bit-identical; "
        f"python {platform.python_version()}, torch {torch.__version__}, "
        f"{ctx.backend.name} ({ctx.backend.device_type} device {ctx.device_index}), "
        f"{_dtype(args)}"
    )


def _answer(ctx: Any, args: Any, kwargs: Any) -> Any:
    """The column's answer as a tuple of tensors, forward or backward."""
    if ctx.backward:
        gradients, _ = ctx.gradients_with(args, kwargs)
        return tuple(gradients)
    out = ctx.call_with(args, kwargs)
    return out if isinstance(out, tuple) else (out,)


def _finite(answer: tuple[Any, ...]) -> bool:
    return all(
        bool(torch.isfinite(value).all()) if value.is_floating_point() else True
        for value in answer
        if isinstance(value, torch.Tensor)
    )


def _same(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        a.shape == b.shape and torch.equal(a.detach().cpu(), b.detach().cpu())
        for a, b in zip(left, right, strict=True)
    )


def _worst(left: tuple[Any, ...], right: tuple[Any, ...]) -> float:
    worst = 0.0
    for a, b in zip(left, right, strict=False):
        if a.shape != b.shape:
            return float("inf")
        if a.is_floating_point() and a.numel():
            gap = (a.detach().cpu().double() - b.detach().cpu().double()).abs().max()
            worst = max(worst, float(gap))
    return worst


def _dtype(args: Any) -> str:
    for value in args:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return str(value.dtype)
    return "no floating operand"
