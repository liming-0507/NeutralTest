"""F3 -- the shipped kernel against the reference on the same inputs.

Check: the column is timed and the reference implementation is timed on the same
arguments at the same width, and the two medians are compared. This is the
constant-factor question -- not whether the kernel is correct (the feature checks
answer that) and not whether it scales (that is F2) -- but whether the shipped
path costs about what the plain composition costs. A kernel that is 20x the
reference on the same shape has a structural problem: an extra copy, a
synchronisation per element, or a fallback that runs the reference and then some.

Tolerance: 8 x the noise slack on a device (12x), 5 x on a CPU (7.5x). That is deliberately loose: on a CPU the
two paths are both ``torch`` and the interesting failures are order-of-magnitude,
which 7.5x still catches while surviving process-level jitter.

Both sides are timed after a warm-up. A device backend pays its first-call costs
-- module load, kernel image, queue setup -- exactly once, and a ratio built on
unwarmed timings measures that one-off against a warm reference: a first call of
42 ms against a 100 us reference reads as 400x and means nothing. The warm-up
call is not timed, and the column carries the median of the timed calls, so what
is compared is steady state against steady state.

A device is also not a CPU with a slower clock: one kernel call is dominated by
launch and queue latency, and the reference it is compared with is one
eager-mode expression. The bound therefore widens where the kernel is a device
call -- 8 x the noise slack instead of 5 x -- and the note records the ratio on
every run, so a drifting machine shows up as a number before it shows up as a
failure. What this row is for is the order of magnitude.
"""

from __future__ import annotations

from typing import Any

import torch

from ...gbz73.inference.kernels import independent_of_reference
from .._support import NOISE_SLACK, make_inputs, measure

ID = "F3"
COLUMNS = None

BOUND = 5.0 * NOISE_SLACK
DEVICE_BOUND = 8.0 * NOISE_SLACK


def run(ctx: Any) -> None:
    if ctx.device.type == "cpu" and not independent_of_reference(ctx.family):
        # Both sides are the same eager expression here, so the ratio is a
        # constant near 1 and the bound can never fire.  Reporting a pass would
        # claim a comparison that was never made; the row has nothing to say
        # about this column on this backend.
        ctx.na(
            f"the kernel and the reference for {ctx.family} are the same arithmetic in this "
            f"build (one eager expression each), so a ratio between them measures spelling, "
            f"not cost; this row needs a device backend or a second genuine implementation"
        )
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    _warm(ctx, args, kwargs)
    measured = measure(lambda: ctx.call_with(args, kwargs))
    baseline = measure(lambda: ctx.reference_with(args, kwargs, dtype=torch.float32))
    for seconds, what in ((measured, "the kernel"), (baseline, "the reference")):
        if seconds < 1e-7:
            ctx.na(
                f"{what} for {ctx.column.name} takes {seconds * 1e9:.0f} ns, below the "
                f"clock's resolution; a ratio would be noise"
            )
    ratio = measured / baseline
    bound = _bound(ctx)
    if ratio > bound:
        raise AssertionError(
            f"{ctx.column.name}: the kernel costs {ratio:.1f}x the reference, over the "
            f"{bound:.1f}x bound"
        )
    ctx.note(f"{ctx.column.name}: kernel/reference cost {ratio:.2f}x")


def _bound(ctx: Any) -> float:
    """Wider on a device, where one kernel call is mostly launch latency."""
    device = getattr(ctx, "device", None)
    return DEVICE_BOUND if device is not None and device.type != "cpu" else BOUND


def _warm(ctx: Any, args: Any, kwargs: Any) -> None:
    """One untimed call on each side, so neither pays its first-call costs inside the ratio."""
    try:
        ctx.call_with(args, kwargs)
        ctx.reference_with(args, kwargs, dtype=torch.float32)
    except Exception as exc:  # noqa: BLE001 - a kernel that cannot run is A1's finding
        ctx.note(f"the warm-up call raised {type(exc).__name__}: {exc}")
