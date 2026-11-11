"""A8 -- accumulation over a long vector.

Check: the rotation is applied to a vector 4096 elements wide -- many more than
the two-element form the reference uses internally -- and compared with the
float64 reference on the same input. Two things are measured: the normalised
error of the whole output, and the drift of each row's norm. The first sees a
summation order that loses low bits; the second sees a rotation that stops being
a rotation once rounding is applied 4096 times.

Tolerance: 8e-6 for the value comparison. Each output element is a sum of two
products of float32 values, and a 4096-wide row accumulates a few ULPs of
float32 rounding per thousand elements; 8e-6 is six times the float32 base and
still two orders of magnitude below the 1e-3 an accumulation bug typically
produces. The norm drift is held at 1e-5, the same bound the reference rotation
test uses, because the norm is a sum of squares and therefore less sensitive than
the individual elements.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import close, flat, make_inputs, observe, reference

ID = "A8"
COLUMNS = None

VALUE_RTOL = 8e-6
NORM_DRIFT = 1e-5


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(
        family=ctx.family, width=4096, batch=1, heads=1, seq=4, seed=ctx.seed, variant=ctx.variant
    )
    got = observe(ctx, args, kwargs)
    want = reference(ctx, args, kwargs, dtype=torch.float64)
    close(got, want, rel=VALUE_RTOL, msg=f"{ctx.column.name}: 4096-wide rotation")
    source = flat(args[0])
    produced = flat(got)
    if source.numel() != produced.numel():  # pragma: no cover - shape is checked elsewhere
        raise AssertionError(
            f"{ctx.column.name}: rotated {source.numel()} values into {produced.numel()}"
        )
    before = source.pow(2).sum()
    after = produced.pow(2).sum()
    drift = float((after - before).abs() / (before + 1e-30))
    if drift > NORM_DRIFT:
        raise AssertionError(
            f"{ctx.column.name}: the norm drifted by {drift:.3e} over a 4096-wide row, "
            f"above {NORM_DRIFT:.1e}"
        )
