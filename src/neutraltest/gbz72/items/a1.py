"""A1 -- the column agrees with the reference implementation.

Check: each column's kernel is compared with the float64 reference on the same
call. A forward column is exercised in float32 and, so that the low-precision
paths are not left to the dtype row alone, in float16 and bfloat16 too. A
backward column has no value of its own, so it is compared by its vector-Jacobian
product against autograd.

Tolerance: the dtype budget from :mod:`neutraltest.tolerance` -- 1.3e-6 for
float32, 1e-3 for float16, 1.6e-2 for bfloat16 -- times the family amplification
(two for the residual add and softmax, sixteen for attention, one for the cache,
two for the rotation, four for the normalisation) and times two when the column
is a backward one.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import check, make_inputs

ID = "A1"
COLUMNS = None


def run(ctx: Any) -> None:
    if ctx.backward:
        check(ctx)
        return
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        args, kwargs = make_inputs(
            family=ctx.family, dtype=dtype, seed=ctx.seed, variant=ctx.variant
        )
        check(ctx, args, kwargs, dtype=dtype)
