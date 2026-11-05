"""B8 -- the batch is a batch: processing rows together equals processing them apart."""

from __future__ import annotations

import torch

from ..context import RunContext
from ._support import adapt

ID = "B8"
OPS = None


def run(ctx: RunContext) -> None:
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(8, 16, generator=generator)

    whole = ctx.op.forward(*adapt(ctx.op.name, x)[0], **adapt(ctx.op.name, x)[1])
    pieces = []
    for start in range(0, x.shape[0], 3):
        chunk = x[start : start + 3]
        pieces.append(ctx.op.forward(*adapt(ctx.op.name, chunk)[0], **adapt(ctx.op.name, chunk)[1]))
    joined = torch.cat(pieces, dim=0)

    if joined.shape != whole.shape:
        raise AssertionError(
            f"{ctx.op.name}: per-batch results have shape {tuple(joined.shape)}, together {tuple(whole.shape)}"
        )
    tolerance = 1e-6 if ctx.op.family in {"normalisation", "gemm"} else 0.0
    if tolerance == 0.0:
        if not torch.equal(joined, whole):
            raise AssertionError(f"{ctx.op.name}: a row depends on the other rows in its batch")
    elif not torch.allclose(joined, whole, atol=tolerance, rtol=0):
        raise AssertionError(f"{ctx.op.name}: batching changed the result by more than {tolerance}")
