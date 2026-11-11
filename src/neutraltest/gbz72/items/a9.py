"""A9 -- the low-precision paths compute the same function.

Check: each forward column is evaluated in float16 and bfloat16 and compared
with the float32 result after both are divided by their own peak. Normalising by
the peak isolates the shape of the result from its magnitude, which is the part a
narrower exponent can legitimately change: a path that computes in float32 and
casts at the end agrees with the float32 shape, while one that rounds an
intermediate to the storage dtype does not. The output dtype itself is checked
too, because a narrow path that silently widens is not a narrow path.

Tolerance: the dtype budget -- 1e-3 for float16 and 1.6e-2 for bfloat16 -- times
the family amplification the column already declares. The values are normalised,
so an exponent difference does not consume the budget; only a difference in the
computed shape does.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import budget, close, flat, make_inputs, observe

ID = "A9"
COLUMNS = None


def run(ctx: Any) -> None:
    if ctx.backward:
        from .._support import check

        check(ctx)
        return
    base_args, kwargs = make_inputs(
        family=ctx.family, dtype=torch.float32, seed=ctx.seed, variant=ctx.variant
    )
    base = flat(observe(ctx, base_args, kwargs))
    base_peak = float(base.abs().max())
    if base_peak == 0.0 or not bool(torch.isfinite(base).all()):
        ctx.na("the float32 result has no finite peak to normalise by")
    for dtype in (torch.float16, torch.bfloat16):
        args, _ = make_inputs(family=ctx.family, dtype=dtype, seed=ctx.seed, variant=ctx.variant)
        produced = observe(ctx, args, kwargs)
        for value in produced:
            if value.dtype != dtype:
                raise AssertionError(
                    f"{ctx.column.name}: asked for {dtype} and received {value.dtype}"
                )
        flat_ = flat(produced)
        if not bool(torch.isfinite(flat_).all()):
            raise AssertionError(f"{ctx.column.name}: the {dtype} path produced a non-finite value")
        peak = float(flat_.abs().max())
        if peak == 0.0:
            raise AssertionError(f"{ctx.column.name}: the {dtype} path produced only zeros")
        close(
            flat_ / peak,
            base / base_peak,
            rel=budget(ctx, dtype),
            msg=f"{ctx.column.name}: {dtype} shape against float32",
        )
