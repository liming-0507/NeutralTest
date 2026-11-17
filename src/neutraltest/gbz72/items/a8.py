"""A8 -- a long reduction keeps its bits.

Check: the operator is run on the widest row the sample allows and compared with
the float64 reference on the same input. The error is measured two ways, because
an accumulation defect shows up differently depending on where in the operator it
sits:

* the normalised error of the whole output, which sees a summation order that
  drops low bits as the row grows, and
* the growth of that error with the width. The same call is made at a short
  width and a long one, and the long one may not be worse by more than the
  square root of the size ratio -- a sum's rounding grows like ``sqrt(n)``, so
  anything that grows like ``n`` is dropping high bits rather than rounding.

The second measure is what distinguishes an accumulation problem from ordinary
float32 rounding, which is why A8 does not simply tighten A1's tolerance: A1
compares at one width, and a sum that is fine at sixteen terms can be wrong at
four thousand without ever leaving A1's bound.

The rotation keeps one property the other operators do not have: it is
orthogonal, so each row's norm is preserved to rounding. That is checked
separately and only for ``rope``, where it is a structural fact rather than a
coincidence of the sample.

Tolerance: ``VALUE_RTOL`` is six times the float32 base, the room a 4096-term
float32 reduction needs; the growth ratio allows ``GROWTH`` times what pure
rounding would produce, and the norm drift is held at 1e-5 because a norm is a
sum of squares and therefore less sensitive than the elements it is made of.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .._support import close, flat, make_inputs, observe, reference

ID = "A8"
COLUMNS = None

#: Width of the long row, and of the short one the growth is measured against.
LONG = 4096
SHORT = 16
#: Normalised error allowed at the long width.
VALUE_RTOL = 8e-6
#: How much worse the long row may be than pure ``sqrt(n)`` rounding predicts.
GROWTH = 4.0
#: Relative change of a row's norm a rotation may produce.
NORM_DRIFT = 1e-5


def run(ctx: Any) -> None:
    long_args, kwargs = _sample(ctx, LONG)
    long_got = observe(ctx, long_args, kwargs)
    long_want = reference(ctx, long_args, kwargs, dtype=torch.float64)
    long_error = close(
        long_got, long_want, rel=VALUE_RTOL, msg=f"{ctx.column.name}: a {LONG}-wide row"
    )
    short_args, _ = _sample(ctx, SHORT)
    short_got = observe(ctx, short_args, kwargs)
    short_want = reference(ctx, short_args, kwargs, dtype=torch.float64)
    short_error = _quiet_error(short_got, short_want)
    if short_error > 0.0 and long_error > 0.0:
        predicted = math.sqrt(LONG / SHORT) * GROWTH
        ratio = long_error / short_error
        if ratio > predicted:
            raise AssertionError(
                f"{ctx.column.name}: the error grows {ratio:.1f}x from {SHORT} to {LONG} wide, "
                f"above the {predicted:.1f}x that rounding alone explains"
            )
    if ctx.family == "rope":
        _rotation_keeps_the_norm(ctx, long_args, long_got)


def _sample(ctx: Any, width: int) -> tuple[Any, Any]:
    """A one-row sample of ``width``, as wide as the family's shape allows.

    The batch, head and token counts drop to one so that the width is the only
    thing being lengthened: a wider row is the reduction this row is about, and
    growing the other axes would change how many reductions there are rather than
    how long they are.
    """
    args, kwargs = make_inputs(
        family=ctx.family,
        width=width,
        batch=1,
        heads=1,
        seq=1,
        rows=1,
        tokens=1,
        seed=ctx.seed,
        variant=ctx.variant,
    )
    return args, kwargs


def _quiet_error(got: Any, want: Any) -> float:
    """The normalised error, or 0 when the two are empty.

    ``close`` raises on a difference and is the wrong tool for the short width,
    where the error is only wanted as a denominator: a short row that already
    failed would have been caught by A1.
    """
    produced = flat(got)
    expected = flat(want)
    if produced.numel() == 0 or expected.numel() == 0:
        return 0.0
    return float((produced - expected).abs().sum() / (expected.abs().sum() + 1e-30))


def _rotation_keeps_the_norm(ctx: Any, args: Any, got: Any) -> None:
    """A rotation is orthogonal, so the row's norm survives the rounding."""
    source = flat(args[0])
    produced = flat(got)
    if source.numel() != produced.numel():  # pragma: no cover - shape is A1's job
        raise AssertionError(
            f"{ctx.column.name}: rotated {source.numel()} values into {produced.numel()}"
        )
    before = source.pow(2).sum()
    after = produced.pow(2).sum()
    drift = float((after - before).abs() / (before + 1e-30))
    if drift > NORM_DRIFT:
        raise AssertionError(
            f"{ctx.column.name}: the norm drifted by {drift:.3e} over a {LONG}-wide row, "
            f"above {NORM_DRIFT:.1e}"
        )
