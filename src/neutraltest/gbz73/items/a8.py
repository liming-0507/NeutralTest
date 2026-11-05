"""A8 -- accumulation error, separated from quantisation error.

Both runs below see exactly the same FP8 values; the only difference is the
precision the operator accumulates in. Whatever error appears here belongs to
the accumulation, not to the format.
"""

from __future__ import annotations

import torch

from .. import metrics
from ..context import RunContext

ID = "A8"
OPS = None


def run(ctx: RunContext) -> None:
    args, kwargs = ctx.inputs(torch.float32)
    quantised, kwargs, _ = ctx.quantise_inputs(args, kwargs)

    wide = tuple(
        a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a
        for a in quantised
    )
    narrow_out = ctx.op.forward(*quantised, **kwargs).to(torch.float32)
    wide_out = ctx.op.forward(*wide, **kwargs).to(torch.float64)

    diff = ctx.diff(narrow_out.to(torch.float64), wide_out)
    bound = 1e-5 if ctx.op.family in {"normalisation", "gemm"} else 1e-6
    metrics.assert_within(
        diff, rel=bound, msg=f"{ctx.op.name}: float32 accumulation vs float64 accumulation"
    )
    ctx.note(f"accumulation only: {diff}")
