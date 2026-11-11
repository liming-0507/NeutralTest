"""F2 -- the cost curve follows the work.

Check: the operator is timed at three sizes that grow by a factor of four from one
to the next, and the cost is required to grow by no more than the shape of the work
allows. A linear operator may cost up to 8x per 4x step (the slack absorbs cache
effects); the attention column declares itself quadratic and is allowed 20x. A
column that pays a fixed launch cost dominates the small sizes and the curve looks
flat; a column whose cost explodes faster than its declared complexity has an
algorithmic problem that no constant-factor tuning will fix.

Tolerance: the per-step growth factors, 8 x the noise slack for the linear
columns and 20 x for the quadratic one. Below the clock's resolution the run stops
with a recorded reason rather than guessing.
"""

from __future__ import annotations

from typing import Any

from .._support import NOISE_SLACK, make_inputs, measure

ID = "F2"
COLUMNS = None

STEPS = (8, 32, 128)
LINEAR_GROWTH = 8.0 * NOISE_SLACK
QUADRATIC_GROWTH = 20.0 * NOISE_SLACK


def run(ctx: Any) -> None:
    times = []

    def timed(args: Any, kwargs: Any) -> float:
        if ctx.backward:
            return measure(lambda: ctx.gradients_with(args, kwargs))
        return measure(lambda: ctx.call_with(args, kwargs))

    for size in STEPS:
        args, kwargs = _sized(ctx, size)
        seconds = timed(args, kwargs)
        if seconds < 1e-7:
            ctx.na(
                f"{ctx.column.name} at size {size} takes {seconds * 1e9:.0f} ns, below the "
                f"clock's resolution; the curve would be noise"
            )
        times.append(seconds)
    bound = QUADRATIC_GROWTH if ctx.column.quadratic else LINEAR_GROWTH
    for small, large in zip(times, times[1:], strict=False):
        growth = large / small
        if growth > bound:
            raise AssertionError(
                f"{ctx.column.name}: cost grew {growth:.1f}x between two sizes that differ "
                f"by 4x, over the {bound:.1f}x this shape of work allows"
            )
    ctx.note(
        f"{ctx.column.name}: sizes {STEPS}, relative cost "
        + ", ".join(f"{value / times[0]:.1f}x" for value in times)
    )


def _sized(ctx: Any, size: int) -> tuple[Any, Any]:
    if ctx.family == "flash_attention":
        return make_inputs(family=ctx.family, seq=size, seed=ctx.seed, variant=ctx.variant)
    if ctx.family == "kv_cache":
        return make_inputs(
            family=ctx.family, rows=max(size, 16), tokens=size, seed=ctx.seed, variant=ctx.variant
        )
    if ctx.family == "rope":
        return make_inputs(
            family=ctx.family, batch=max(2, size // 2), seed=ctx.seed, variant=ctx.variant
        )
    return make_inputs(family=ctx.family, rows=size, seed=ctx.seed, variant=ctx.variant)
