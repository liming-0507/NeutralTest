"""F1 -- the operator can be timed at all, and its latency is stable.

Check: the column is called through the measurement helper, the median of the
samples has to be large enough for this machine's clock to resolve, and a second
run of the same measurement has to land within a small factor of the first. The
throughput is then derived from the element count and the median, and must be a
finite positive number. A column that is too cheap to time is not a pass: it stops
with an explicit reason, because the verdict would otherwise be decided by timer
noise rather than by the operator.

Tolerance: the two medians may differ by up to 3 x the noise slack (4.5x); the
throughput has no tolerance, it only has to be positive and finite, and the
measured number is attached to the result as a note.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import NOISE_SLACK, make_inputs, measure

ID = "F1"
COLUMNS = None

STABILITY = 3.0 * NOISE_SLACK


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    if ctx.backward:
        call: Any = lambda: ctx.gradients_with(args, kwargs)  # noqa: E731
    else:
        call = lambda: ctx.call_with(args, kwargs)  # noqa: E731
    first = measure(call)
    _require(ctx, first)
    second = measure(call)
    _require(ctx, second)
    ratio = max(first, second) / min(first, second)
    if ratio > STABILITY:
        raise AssertionError(
            f"{ctx.column.name}: the same measurement moved by {ratio:.2f}x, over the "
            f"{STABILITY:.1f}x stability bound"
        )
    elements = sum(int(value.numel()) for value in args if isinstance(value, torch.Tensor))
    throughput = elements / second
    if not throughput > 0.0 or throughput != throughput:
        raise AssertionError(f"{ctx.column.name}: the derived throughput is not finite")
    ctx.note(
        f"{ctx.column.name}: {second * 1e6:.1f} us per call, {throughput / 1e6:.1f} Melement/s"
    )


def _require(ctx: Any, seconds: float) -> None:
    if seconds <= 0.0:
        raise AssertionError(f"{ctx.column.name}: the timer returned {seconds!r}")
    if seconds < 1e-7:
        ctx.na(
            f"{ctx.column.name} completes in {seconds * 1e9:.0f} ns, below the clock's "
            f"resolution; the measurement would decide the verdict, not the operator"
        )
