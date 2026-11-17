"""D2 -- the parameter sweep.

Check: every option a column takes that has a range of legal values is walked
across it, and every point is compared with the operation it should reduce to.
``add`` sweeps ``alpha`` through negative, zero, unit and beyond-unit values and
holds two of them to a closed form: ``alpha=1`` must be a plain add and
``alpha=0`` must return the first operand exactly. Attention sweeps
``softmax_scale`` and holds the result to the reference at each point, plus the
degenerate ``scale=0`` where every row becomes the mean of ``v``. The
normalisation sweeps ``eps`` through six orders of magnitude and requires the
damping to be monotone in it, since a larger epsilon can only shrink the
reciprocal root mean square, and the two ends of the sweep must differ so that
an ignored ``eps`` cannot pass.

A family whose only options are choices rather than numbers -- softmax's axis,
the rotation's layout switch, the cache's quantisation -- has nothing to sweep,
and the column says so instead of pretending a sweep happened.

Tolerance: 1e-6 for the add against its closed form (a single multiply and an
add, both in float32), and the column's budget for everything compared against a
reference. The monotonicity check allows 1e-6 of relative slack for the two
smallest epsilons, where the damping is below the rounding of the statistic it
regularises.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import budget, close, make_inputs, numeric_option, observe, reference

ID = "D2"
COLUMNS = None

#: Kept as module constants so a test can assert the sweep is the documented one.
ALPHAS = (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.5)
SCALES = (0.25, 0.5, 1.0, 1.0 / 4**0.5, 2.0)
EPSILONS = (1e-12, 1e-9, 1e-6, 1e-3, 1e-1)


def run(ctx: Any) -> None:
    name, values = numeric_option(ctx.family)
    if not name:
        # No continuum: the axis, the layout and the quantisation mode are
        # discrete choices, and D1 already compares the two call forms.
        ctx.note(
            f"{ctx.family} has no numeric option to sweep; its options are choices, "
            "which D1 and C4 cover"
        )
        return
    if ctx.family == "add":
        _add(ctx, values)
    elif ctx.family == "flash_attention":
        _attention(ctx, name, values)
    else:
        _rms(ctx, name, values)


def _add(ctx: Any, alphas: tuple[float, ...]) -> None:
    args, _ = make_inputs(family="add", seed=ctx.seed, variant=ctx.variant)
    a, b = args
    for alpha in alphas:
        produced = observe(ctx, args, {"alpha": alpha})[0]
        close(produced, a + alpha * b, rel=1e-6, msg=f"{ctx.column.name}: alpha={alpha:g}")
    if not torch.equal(observe(ctx, args, {"alpha": 1.0})[0], a + b):
        raise AssertionError(f"{ctx.column.name}: alpha=1 is not a plain add")
    if not torch.equal(observe(ctx, args, {"alpha": 0.0})[0], a):
        raise AssertionError(f"{ctx.column.name}: alpha=0 is not the first operand")


def _attention(ctx: Any, name: str, scales: tuple[float, ...]) -> None:
    args, kwargs = make_inputs(family="flash_attention", seed=ctx.seed, variant=ctx.variant)
    for scale in scales:
        options = dict(kwargs, **{name: scale})
        produced = observe(ctx, args, options)
        close(
            produced,
            reference(ctx, args, options, dtype=torch.float64),
            rel=budget(ctx),
            msg=f"{ctx.column.name}: {name}={scale:g}",
        )
    # scale=0 flattens the scores, so every row must collapse onto the mean of v:
    # the one point of the sweep whose answer is known without a reference.
    flat = dict(kwargs, **{name: 0.0})
    flattened = observe(ctx, args, flat)[0]
    mean_v = args[2].to(torch.float64).mean(dim=2, keepdim=True).to(flattened.dtype)
    spread = mean_v.expand_as(flattened).contiguous()
    close(flattened, spread, rel=budget(ctx), msg=f"{ctx.column.name}: {name}=0 is the mean of v")


def _rms(ctx: Any, name: str, epsilons: tuple[float, ...]) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    norms = []
    for eps in epsilons:
        options = dict(kwargs, **{name: eps})
        produced = observe(ctx, args, options)[0]
        close(
            produced,
            reference(ctx, args, options, dtype=torch.float64)[0],
            rel=budget(ctx),
            msg=f"{ctx.column.name}: {name}={eps:g}",
        )
        norms.append(float(produced.double().norm()))
    for previous, current in zip(norms, norms[1:], strict=False):
        if current > previous * (1.0 + 1e-6):
            raise AssertionError(
                f"{ctx.column.name}: raising {name} increased the norm from "
                f"{previous:.6e} to {current:.6e}"
            )
    if not norms[0] > norms[-1]:
        raise AssertionError(f"{ctx.column.name}: {name} had no effect on the result")
