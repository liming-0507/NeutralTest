"""F1 -- how long one call takes.

No absolute threshold: a wall-clock bound tuned on one machine is a future false
alarm on another. What is checked is that a number exists, that it is stable
enough to quote, and that it is reported with the shape it was measured on.
"""

from __future__ import annotations

from ..context import RunContext
from ._support import device_call, measure

ID = "F1"
OPS = None


def run(ctx: RunContext) -> None:
    call = device_call(ctx)
    seconds = measure(call)
    if not seconds > 0:
        raise AssertionError(
            f"{ctx.op.name}: a call took {seconds:.3e} s, which cannot be a measurement"
        )

    # The median must be reproducible: two independent batches within 3x.
    again = measure(call)
    ratio = max(seconds, again) / max(min(seconds, again), 1e-12)
    if ratio > 3.0:
        raise AssertionError(f"{ctx.op.name}: two timings of the same call differ by {ratio:.1f}x")
    ctx.note(
        f"{ctx.op.name}/{ctx.variant}: {seconds * 1e3:.3f} ms/call (repeat {again * 1e3:.3f} ms)"
    )
