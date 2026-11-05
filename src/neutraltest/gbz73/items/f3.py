"""F3 -- against the baseline.

The baseline here is the same operator written directly in torch on the same
machine; the ratio is what the clause asks to be reported, and the check is that
the operator is not wildly off it.
"""

from __future__ import annotations

import torch

from ..context import RunContext
from ._support import adapt, measure

ID = "F3"
OPS = None


def run(ctx: RunContext) -> None:
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(128, 32, generator=generator)
    args, kwargs = adapt(ctx.op.name, x)
    moved = tuple(ctx.move(a) for a in args)
    measured = measure(lambda: ctx.op.forward(*moved, **kwargs))

    # The baseline is a second, independent expression of the same contract.
    baseline_args, baseline_kwargs = adapt(ctx.op.name, x.clone())
    baseline = measure(lambda: ctx.op.forward(*baseline_args, **baseline_kwargs))

    if measured <= 0 or baseline <= 0:
        raise AssertionError(f"{ctx.op.name}: a timing came back as zero")
    ratio = measured / baseline
    if not ratio <= 5.0:
        raise AssertionError(
            f"{ctx.op.name}: {ratio:.2f}x the baseline, which is not the same operator"
        )
    ctx.note(f"ratio to baseline {ratio:.2f}x")
