"""E5 -- the wide case.

Check: the rotation is run at widths that a small scratch buffer could not hold --
256, 1024, 4096 and 8192 lanes, i.e. 128 to 4096 angle pairs per position -- and
the result is compared with the float64 reference at every width. A kernel that
indexes a fixed-size table of angles, or that allocates its tables once for the
first width it sees, fails here while passing the narrow case. The rotation norm
is checked at each width too, since a table that is truncated at a buffer
boundary can preserve values in the middle and lose the norm at the edges.

Tolerance: the column's float32 budget against the float64 reference, and 1e-5 of
relative drift for the rotation norm at every width. The norm bound is deliberately
constant across widths: if it has to grow with the width, the angle table is not
being indexed correctly.
"""

from __future__ import annotations

from typing import Any

import torch

from ...gbz73.inference import kernels
from .._support import budget, close, make_inputs, reference

ID = "E5"
COLUMNS = None

WIDTHS = (256, 1024, 4096, 8192)
NORM_DRIFT = 1e-5


def run(ctx: Any) -> None:
    if ctx.family != "rope":
        ctx.na(f"{ctx.column.name} has no resource ceiling recorded in this clause")
        return
    for width in WIDTHS:
        args, kwargs = make_inputs(family="rope", width=width, seed=ctx.seed)
        x, position = args
        produced = kernels.rope(x, position, **kwargs)
        if not bool(torch.isfinite(produced).all()):
            raise AssertionError(f"{ctx.column.name}: width {width} produced a non-finite value")
        close(
            produced,
            reference(ctx, args, kwargs, dtype=torch.float64),
            rel=budget(ctx, kind="normalization"),
            msg=f"{ctx.column.name}: width {width}",
        )
        before = x.double().norm(dim=-1)
        after = produced.double().norm(dim=-1)
        drift = float((after - before).abs().max() / before.max())
        if drift > NORM_DRIFT:
            raise AssertionError(
                f"{ctx.column.name}: width {width} drifted the norm by {drift:.3e}"
            )
