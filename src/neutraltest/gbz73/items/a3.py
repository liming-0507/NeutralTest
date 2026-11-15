"""A3 -- what happens above the largest finite value.

A finite input must produce a finite output: a format without an infinity has
to saturate at its maximum, and a silent NaN (or a wrap-around) is the failure
this row exists to catch.

The row is about the *operator's* output, not about the cast.  :func:`quant.roundtrip`
clamps before it casts, so calling it directly measures this framework's clamp
and returns the same answer whatever operator is under test -- five green ticks
that say nothing about any of them.  The check here therefore runs the operator
first, pushes its result past the format's range with a deliberately small
scale, and only then asks what the quantiser did with it.

Which operators this can say anything about is decided by the operator, not by
this file: a result that is bounded by construction (a softmax mass, a
normalised vector) cannot reach the boundary by scaling its inputs, and the row
declares itself not applicable there rather than reporting a pass for a check
that never ran.
"""

from __future__ import annotations

import torch

from .. import quant
from ..context import RunContext

ID = "A3"
OPS = None


def run(ctx: RunContext) -> None:
    dtype = ctx.fp8_dtype
    top = quant.max_finite(dtype)

    # Run the operator itself, then force the result out of range.  A scale far
    # below the result's own magnitude is what an accumulator overflow looks
    # like to the quantiser: values that are finite in float32 and larger than
    # anything the format can hold.
    args, kwargs = ctx.inputs(torch.float32)
    produced = ctx.op.forward(*args, **kwargs)
    if produced.numel() == 0:
        ctx.na(f"{ctx.op.name}: the operator produced no elements to saturate")

    magnitude = float(produced.detach().abs().max())
    if magnitude == 0.0:
        ctx.na(f"{ctx.op.name}: the operator produced only zeros; nothing can exceed the range")

    #: Put the largest element at ``top * overshoot`` and let the rest follow.
    overshoot = 1000.0
    scale = magnitude / (top * overshoot)

    back = quant.roundtrip(produced.detach(), scale, dtype)
    finite_before = torch.isfinite(produced).all().item()
    if not finite_before:
        ctx.na(f"{ctx.op.name}: the operator's own output is not finite; A6 is the row for that")

    if torch.isnan(back).any():
        raise AssertionError(
            f"{ctx.op.name}/{dtype}: a finite input became NaN on saturation: "
            f"{back.flatten()[:8].tolist()}"
        )
    if torch.isinf(back).any():
        raise AssertionError(
            f"{ctx.op.name}/{dtype}: a finite input became Inf: {back.flatten()[:8].tolist()}"
        )

    limit = top * scale
    if not torch.all(back.abs() <= limit * (1 + 1e-6)):
        worst = float(back.abs().max())
        raise AssertionError(
            f"{ctx.op.name}/{dtype}: output left the representable range: {worst:g} > {limit:g}"
        )

    # The largest element is the one that had to saturate; if it came back much
    # smaller the quantiser wrapped instead of clamping.
    largest = produced.detach().abs().max()
    got = back[produced.detach().abs() == largest].abs().max()
    if float(got) < limit * 0.99:
        raise AssertionError(
            f"{ctx.op.name}/{dtype}: the largest element did not saturate -- it came back "
            f"{float(got):g} where the format's ceiling at this scale is {limit:g}"
        )
