"""A3 -- overflow saturates at the format's largest finite value.

Check: the column's result is quantised at its dynamic scale, which must not
saturate anything, and again at a scale sixteen times smaller, which must. Every
element whose magnitude exceeds the format's largest finite value after scaling
has to land exactly on ``+max_finite`` or ``-max_finite``; an element well below
that bound must not. Saturation to infinity or to NaN instead of the largest
finite value is the failure this row exists to catch, and it is observable only
by exact comparison against the format's own bound.

Tolerance: exact. Both sides of every comparison are points of the FP8 format,
so there is no arithmetic between them that could round.
"""

from __future__ import annotations

from typing import Any

import torch

from ...gbz73 import quant
from .._support import flat, make_inputs, observe

ID = "A3"
COLUMNS = None


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(
        family=ctx.family, dtype=torch.float32, seed=ctx.seed, variant=ctx.variant
    )
    out = flat(observe(ctx, args, kwargs)).to(torch.float32)
    if not bool(torch.isfinite(out).all()):
        ctx.na("the column's output is not finite, so there is no finite peak to saturate")
    if float(out.abs().max()) == 0.0:
        ctx.na("the column's output is all zeros, so there is nothing to saturate")
    for dtype in quant.FP8_DTYPES:
        top = float(quant.max_finite(dtype))
        safe = float(quant.dynamic_scale(out, dtype))
        gentle = quant.quantise(out, safe, dtype).to(torch.float32)
        if float(gentle.abs().max()) > top:  # pragma: no cover - defensive
            raise AssertionError(
                f"{ctx.column.name}: {dtype} produced {float(gentle.abs().max())} "
                f"above the format maximum {top} at the dynamic scale"
            )
        tight = safe / 16.0
        saturated = quant.quantise(out, tight, dtype).to(torch.float32)
        over = out.abs() > top * tight
        if not bool(over.any()):
            ctx.na(f"no element of the {ctx.column.name} output saturates at a scale/16")
        if not torch.equal(saturated[over].abs(), torch.full_like(saturated[over], top)):
            raise AssertionError(
                f"{ctx.column.name}: {dtype} did not saturate to {top} for "
                f"{int(over.sum())} out-of-range elements"
            )
        under = out.abs() < top * tight * 0.5
        if bool(under.any()) and bool((saturated[under].abs() >= top).any()):
            raise AssertionError(
                f"{ctx.column.name}: {dtype} saturated an element that was in range"
            )
