"""D2 -- the parameter sweep.

Check: the two parameters that take a range of legal values are walked across it
-- ``alpha`` through negative, zero, unit and beyond-unit values, ``eps`` through
six orders of magnitude -- and every point is compared with the operation it
should reduce to. ``alpha=1`` must be a plain add and ``alpha=0`` must return the
first operand exactly; the rotation's damping must be monotone in ``eps``, since
a larger epsilon can only shrink the reciprocal root mean square, and the two
ends of the sweep must differ so that an ignored ``eps`` cannot pass.

Tolerance: 1e-6 for the add against its closed form (a single multiply and an
add, both in float32), and the column's float32 budget for the normalisation
against its reference. The monotonicity check allows 1e-6 of relative slack for
the two smallest epsilons, where the damping is below the rounding of the
statistic it regularises.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import budget, close, make_inputs, observe, reference

ID = "D2"
COLUMNS = None

ALPHAS = (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.5)
EPSILONS = (1e-12, 1e-9, 1e-6, 1e-3, 1e-1)


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    if ctx.family == "add":
        _add(ctx, args)
    else:
        _rms(ctx, args)


def _add(ctx: Any, args: Any) -> None:
    a, b = args
    for alpha in ALPHAS:
        produced = observe(ctx, args, {"alpha": alpha})[0]
        close(
            produced,
            a + alpha * b,
            rel=1e-6,
            msg=f"{ctx.column.name}: alpha={alpha:g}",
        )
    if not torch.equal(observe(ctx, args, {"alpha": 1.0})[0], a + b):
        raise AssertionError(f"{ctx.column.name}: alpha=1 is not a plain add")
    if not torch.equal(observe(ctx, args, {"alpha": 0.0})[0], a):
        raise AssertionError(f"{ctx.column.name}: alpha=0 is not the first operand")


def _rms(ctx: Any, args: Any) -> None:
    norms = []
    for eps in EPSILONS:
        produced = observe(ctx, args, {"eps": eps})[0]
        close(
            produced,
            reference(ctx, args, {"eps": eps}, dtype=torch.float32)[0],
            rel=budget(ctx),
            msg=f"{ctx.column.name}: eps={eps:g}",
        )
        norms.append(float(produced.double().norm()))
    for previous, current in zip(norms, norms[1:], strict=False):
        if current > previous * (1.0 + 1e-6):
            raise AssertionError(
                f"{ctx.column.name}: raising eps increased the norm from "
                f"{previous:.6e} to {current:.6e}"
            )
    if not norms[0] > norms[-1]:
        raise AssertionError(f"{ctx.column.name}: eps had no effect on the result")
