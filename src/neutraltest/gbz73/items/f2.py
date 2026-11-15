"""F2 -- the cost as the shape grows.

Doubling the work should not more than triple the time; if it does, something is
quadratic that has no business being quadratic.
"""

from __future__ import annotations

import torch

from ..context import RunContext
from ._support import NOISE_SLACK, adapt, measure, require_resolvable

ID = "F2"
OPS = None


def _time(ctx: RunContext, rows: int) -> float:
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(rows, 32, generator=generator)
    args, kwargs = adapt(ctx.op.name, x)
    moved = tuple(ctx.move(a) for a in args)
    return require_resolvable(measure(lambda: ctx.op.forward(*moved, **kwargs)), ctx.op.name)


def _median_of(ctx: RunContext, rows: int, *, rounds: int = 5) -> float:
    """Median over several independent measurements, not the one inside them.

    ``measure`` returns the median of one call shape's samples, so the pair of
    numbers that becomes a ratio still carries that shape's whole variance: the
    GIL is released around the operator, and one scheduler slice landing on a
    busy core moves a single 64-row batch by half again.  Five rounds cost a few
    milliseconds and take the ratio back to what it is meant to describe.
    """
    readings = sorted(_time(ctx, rows) for _ in range(rounds))
    return readings[rounds // 2]


def run(ctx: RunContext) -> None:
    small = _median_of(ctx, 64)
    large = _median_of(ctx, 256)
    if small <= 0:
        raise AssertionError(f"{ctx.op.name}: the small shape was too fast to time")
    growth = large / small
    # Four times the rows may cost at most 12x the time (3x the ideal 4x), plus
    # the noise the measurement protocol itself carries.
    bound = 12.0 * NOISE_SLACK
    if not growth <= bound:
        raise AssertionError(f"{ctx.op.name}: 4x the rows cost {growth:.1f}x the time")
    ctx.note(f"64 rows {small * 1e6:.1f} us, 256 rows {large * 1e6:.1f} us ({growth:.2f}x)")
