"""C3 -- the granularity of the scale.

Check: a column's output is quantised twice -- once with a single scale for the
whole tensor, once with one scale per row -- and the per-row round trip must be no
worse than the per-tensor one. The point is not that fine scales are always
better, but that a row whose magnitude is far from the tensor peak benefits when
the scale follows the row, which is what a per-channel scheme promises. A
quantiser that ignores the ``dim`` argument, or a scale that is broadcast over the
wrong axis, gives the same error for both and fails the inequality.

Tolerance: the per-row error must stay within 1.25 times the per-tensor error plus
1e-9, and the per-tensor error itself within two half-ulps of the format. The
quarter of headroom absorbs the rows whose peak is not much below the tensor
peak, where the finer scale buys nothing but rounds just as well.
"""

from __future__ import annotations

from typing import Any

import torch

from ...gbz73 import quant
from ...gbz73.metrics import compare
from .._support import flat, make_inputs, observe

ID = "C3"
COLUMNS = None

SLACK = 1.25


def run(ctx: Any) -> None:
    values = _values(ctx)
    if float(values.abs().max()) == 0.0:
        ctx.na("every value is zero, so no scale can be applied")
    for dtype in quant.FP8_DTYPES:
        global_scale = float(torch.as_tensor(quant.dynamic_scale(values, dtype)).reshape(-1).max())
        rows = torch.as_tensor(quant.dynamic_scale(values, dtype, dim=1)).reshape(-1, 1)
        tensor_error = compare(
            quant.dequantise(quant.quantise(values, global_scale, dtype), global_scale), values
        ).rel
        row_error = compare(quant.dequantise(quant.quantise(values, rows, dtype), rows), values).rel
        if tensor_error > 2.0 * float(quant.half_ulp(dtype)):
            raise AssertionError(
                f"{ctx.column.name}: the {dtype} round trip at one scale costs "
                f"{tensor_error:.3e}, more than two half-ulps"
            )
        if row_error > SLACK * tensor_error + 1e-9:
            raise AssertionError(
                f"{ctx.column.name}: the per-row {dtype} scale costs {row_error:.3e}, "
                f"worse than the per-tensor {tensor_error:.3e}"
            )


def _values(ctx: Any) -> torch.Tensor:
    args, kwargs = make_inputs(
        family=ctx.family, dtype=torch.float32, seed=ctx.seed, variant=ctx.variant
    )
    if ctx.backward:
        gradients, _ = ctx.gradients_with(args, kwargs)
        return flat(gradients).reshape(8, -1)
    produced = observe(ctx, args, kwargs)
    width = produced[0].shape[-1]
    return flat(produced).reshape(-1, width)
