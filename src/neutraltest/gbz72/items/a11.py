"""A11 -- error accumulation across a short training chain.

Check: the normalisation is applied five times in a row, each step feeding the
next, with the same weight, and every intermediate result is compared with a
float64 chain that is fed the same way. The point is drift: a per-step error that
is invisible on one call accumulates when the operator is stacked, and a chain
whose deviation grows faster than linearly in the step count is not usable for a
few hundred steps of training.

Tolerance: 1e-3 per step against the float64 chain -- each step is a full
normalisation, and the chain amplifies whatever the step got wrong -- and the
final deviation may be at most six times the first: five steps may grow the error
linearly, not geometrically. The total deviation is also held to 1e-3, and every
intermediate must stay finite.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import close, flat, make_inputs, observe, reference

ID = "A11"
COLUMNS = None

STEPS = 5
TOTAL = 1e-3
GROWTH = 6.0


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(
        family="rms_norm", dtype=torch.float32, rows=4, width=64, seed=ctx.seed
    )
    state, weight = args
    expected = state.to(torch.float64)
    deviations = []
    for step in range(STEPS):
        state = observe(ctx, (state, weight), kwargs)[0]
        expected = reference(ctx, (expected, weight), kwargs, dtype=torch.float64)[0]
        if not bool(torch.isfinite(state).all()):
            raise AssertionError(f"{ctx.column.name}: step {step} produced a non-finite value")
        deviations.append(close(state, expected, rel=TOTAL, msg=f"{ctx.column.name}: step {step}"))
    if deviations[-1] > GROWTH * deviations[0] + 1e-9:
        raise AssertionError(
            f"{ctx.column.name}: the deviation grew from {deviations[0]:.3e} to "
            f"{deviations[-1]:.3e} over {STEPS} steps, faster than linear accumulation"
        )
    if not bool(torch.isfinite(flat(state)).all()):
        raise AssertionError(f"{ctx.column.name}: the chain produced a non-finite value")
