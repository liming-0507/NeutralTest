"""C1 -- the dtype matrix.

Check: every column is evaluated in float32, float64, float16 and bfloat16 and
compared with the reference in the same width; the result dtype is required to be
the requested one. The clause's columns compute; they do not store. The FP8
formats that the 7.3 rows enumerate are therefore exercised where a column
stores them -- the cache in A2 and A3 -- rather than as compute types, because no
reference kernel for these six operators computes in FP8 on this backend, and a
comparison against a reference that widens to float32 first would measure the
cast rather than the operator.

Tolerance: the dtype budget. float64 is 1e-12, float32 1.3e-6, float16 1e-3 and
bfloat16 1.6e-2, each times the family amplification and twice for a backward
column. Backward columns have no value of their own and are compared by their
VJP against autograd at 1e-9 instead.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import check, make_inputs, observe

ID = "C1"
COLUMNS = None

DTYPES = (torch.float64, torch.float32, torch.float16, torch.bfloat16)


def run(ctx: Any) -> None:
    if ctx.backward:
        check(ctx)
        return
    ctx.note(
        "FP8 formats are storage types for these columns and are covered by the "
        "quantisation rows; no reference kernel here computes in FP8."
    )
    for dtype in DTYPES:
        args, kwargs = make_inputs(
            family=ctx.family, dtype=dtype, seed=ctx.seed, variant=ctx.variant
        )
        check(ctx, args, kwargs, dtype=dtype)
        for value in observe(ctx, args, kwargs):
            if value.dtype != dtype:
                raise AssertionError(
                    f"{ctx.column.name}: asked for {dtype} and received {value.dtype}"
                )
