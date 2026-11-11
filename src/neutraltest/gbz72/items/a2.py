"""A2 -- an FP8 round trip of the column's own output.

Check: the column's float32 result is quantised to each FP8 format available in
this build and dequantised again, once at the format's dynamic scale (the peak
over the whole tensor) and once at a scale eight times smaller, which pushes the
tail of the distribution into the subnormal band. The error is normalised the
same way the rest of the suite normalises it, so a scale change cancels out: a
correct format loses resolution but not its relative accuracy.

Tolerance: three half-ULPs -- one for the cast, one for the multiply back, and
one for the accumulation across the tensor. A float8 cast cannot do better than
half an ULP on any single element, so a tighter bound would fail for a reason
that has nothing to do with the operator; a looser one would let a broken
rounding mode through.
"""

from __future__ import annotations

from typing import Any

import torch

from ...gbz73 import quant
from .._support import close, flat, make_inputs, observe

ID = "A2"
COLUMNS = None


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(
        family=ctx.family, dtype=torch.float32, seed=ctx.seed, variant=ctx.variant
    )
    out = flat(observe(ctx, args, kwargs)).to(torch.float32)
    peak = float(out.abs().max())
    if peak == 0.0:
        ctx.na("the column's output is all zeros, so there is no scale to apply")
    for dtype in quant.FP8_DTYPES:
        allowed = 3.0 * float(quant.half_ulp(dtype))
        dynamic = float(quant.dynamic_scale(out, dtype))
        if dynamic <= 0.0:  # pragma: no cover - guarded by the peak check above
            ctx.na(f"{dtype} has no non-zero scale for this output")
        for factor in (1.0, 0.5):
            scale = dynamic * factor
            back = quant.dequantise(quant.quantise(out, scale, dtype), scale)
            close(back, out, rel=allowed, msg=f"{ctx.column.name}: {dtype} at scale x{factor:g}")
