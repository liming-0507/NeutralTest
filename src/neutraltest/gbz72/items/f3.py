"""F3 -- the shipped kernel against the reference on the same inputs.

Check: the column is timed and the reference implementation is timed on the same
arguments at the same width, and the two medians are compared. This is the
constant-factor question -- not whether the kernel is correct (the feature checks
answer that) and not whether it scales (that is F2) -- but whether the shipped
path costs about what the plain composition costs. A kernel that is 20x the
reference on the same shape has a structural problem: an extra copy, a
synchronisation per element, or a fallback that runs the reference and then some.

Tolerance: 5 x the noise slack (7.5x). That is deliberately loose: on a CPU the
two paths are both ``torch`` and the interesting failures are order-of-magnitude,
which 7.5x still catches while surviving process-level jitter.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import NOISE_SLACK, make_inputs, measure

ID = "F3"
COLUMNS = None

BOUND = 5.0 * NOISE_SLACK


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    measured = measure(lambda: ctx.call_with(args, kwargs))
    baseline = measure(lambda: ctx.reference_with(args, kwargs, dtype=torch.float32))
    for seconds, what in ((measured, "the kernel"), (baseline, "the reference")):
        if seconds < 1e-7:
            ctx.na(
                f"{what} for {ctx.column.name} takes {seconds * 1e9:.0f} ns, below the "
                f"clock's resolution; a ratio would be noise"
            )
    ratio = measured / baseline
    if ratio > BOUND:
        raise AssertionError(
            f"{ctx.column.name}: the kernel costs {ratio:.1f}x the reference, over the "
            f"{BOUND:.1f}x bound"
        )
    ctx.note(f"{ctx.column.name}: kernel/reference cost {ratio:.2f}x")
