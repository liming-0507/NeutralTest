"""C5 -- what comes back out, and in which dtype.

The quantisation is allowed to be visible: what is not allowed is a result that
looks like float but is not, or an output stage that silently did nothing.
"""

from __future__ import annotations

import torch

from .. import quant
from ..context import RunContext

ID = "C5"
OPS = None


def run(ctx: RunContext) -> None:
    raw = ctx.quantised(quantise_output=False)
    rounded = ctx.quantised(quantise_output=True)

    if raw.dtype != torch.float32 or rounded.dtype != torch.float32:
        raise AssertionError(
            f"{ctx.op.name}: output dtype is {raw.dtype}/{rounded.dtype}, expected float32"
        )

    encoded = quant.quantise(rounded, 1.0, ctx.fp8_dtype)
    if encoded.element_size() != 1:
        raise AssertionError(
            f"{ctx.op.name}: the FP8 view is {encoded.element_size()} bytes per element, expected 1"
        )
    if encoded.dtype != ctx.fp8_dtype:
        raise AssertionError(
            f"{ctx.op.name}: encoding gave {encoded.dtype}, expected {ctx.fp8_dtype}"
        )

    # An output that was already on the grid (an amax is one of its inputs) is
    # allowed to come back unchanged -- but only if it really is on the grid.
    scale = quant.dynamic_scale(raw.reshape(-1), ctx.fp8_dtype)
    on_grid = torch.equal(quant.roundtrip(raw, scale, ctx.fp8_dtype), raw)
    if torch.equal(raw, rounded) and not on_grid:
        raise AssertionError(
            f"{ctx.op.name}: quantising the output changed nothing, so the stage is not applied"
        )
    # Re-quantising with a fresh scale is not required to be idempotent: the
    # multiply-then-divide need not land on the same code point. What must hold
    # is that the deviation stayed within half an ULP.
    moved = float((raw - rounded).abs().max())
    allowed = quant.half_ulp(ctx.fp8_dtype) * float(
        raw.detach().abs().clamp_min(float(torch.finfo(ctx.fp8_dtype).tiny)).max()
    )
    if moved > allowed * 1.0005:
        raise AssertionError(
            f"{ctx.op.name}: output rounding moved the result by {moved:.3e}, more than half an ULP"
        )
    ctx.note(f"output rounding moved the result by {moved:.3e}")
