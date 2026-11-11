"""A4 -- subnormals survive, and half a subnormal rounds to zero.

Check: two things. First, the format itself: a value at the smallest subnormal,
its multiples, and a value exactly halfway between zero and the smallest
subnormal are pushed through the cast and the multiply back. The representable
ones must come back unchanged -- a format that flushed subnormals to zero would
turn a smooth decay into a cliff -- and the halfway value must round to zero,
because zero has an even significand and round-to-nearest-even picks it. Second,
the column: scaling its output so that its smallest non-zero magnitude sits in
the subnormal band must still produce non-zero subnormals, so the path does not
flush the band before the format sees it.

Tolerance: exact. Every value compared is a point of the format, so any
difference is a defect in the cast rather than rounding noise.
"""

from __future__ import annotations

from typing import Any

import torch

from ...gbz73 import quant
from .._support import flat, make_inputs, observe

ID = "A4"
COLUMNS = None


def run(ctx: Any) -> None:
    for dtype in quant.FP8_DTYPES:
        tiny = float(quant.min_subnormal(dtype))
        values = torch.tensor(
            [4.0 * tiny, 2.0 * tiny, tiny, tiny / 2.0, tiny / 4.0, 0.0, -tiny / 2.0],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [4.0 * tiny, 2.0 * tiny, tiny, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )
        back = quant.dequantise(quant.quantise(values, 1.0, dtype), 1.0)
        if not torch.equal(back, expected):
            raise AssertionError(
                f"{ctx.column.name}: {dtype} subnormals did not survive the round trip: "
                f"{back.tolist()} instead of {expected.tolist()}"
            )

    args, kwargs = make_inputs(
        family=ctx.family, dtype=torch.float32, seed=ctx.seed, variant=ctx.variant
    )
    out = flat(observe(ctx, args, kwargs)).to(torch.float32)
    non_zero = out[out != 0.0].abs()
    if non_zero.numel() == 0:
        ctx.na("the column's output is all zeros, so it has no smallest magnitude")
    for dtype in quant.FP8_DTYPES:
        tiny = float(quant.min_subnormal(dtype))
        target = float(non_zero.median())
        scale = target / tiny
        quantised = quant.quantise(out, scale, dtype).to(torch.float32)
        band = (quantised.abs() > 0.0) & (quantised.abs() < float(torch.finfo(dtype).tiny))
        if not bool(band.any()):
            ctx.na(f"{ctx.column.name}: no output element falls in the {dtype} subnormal band")
        if not bool(torch.isfinite(quantised).all()):
            raise AssertionError(
                f"{ctx.column.name}: {dtype} produced a non-finite quantised value"
            )
        if not bool((quantised[out.abs() >= target] != 0.0).all()):
            raise AssertionError(
                f"{ctx.column.name}: {dtype} flushed a value at or above the median to zero"
            )
