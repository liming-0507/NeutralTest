"""C1 -- every format the clause names, not just the default one.

E4M3 and E5M2 differ by a factor of two in precision and a factor of four in
range, so each gets its own floor. MXFP8 is named by the clause but is not a
dtype this build can construct; that is disclosed rather than skipped quietly.
"""

from __future__ import annotations

import torch

from .. import metrics, quant
from ..context import RunContext
from ._support import adapt

ID = "C1"
OPS = None


def run(ctx: RunContext) -> None:
    formats = [d for d in quant.FP8_DTYPES if hasattr(torch, str(d).replace("torch.", ""))]
    if not formats:
        ctx.na("this build has no FP8 dtype")

    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(64, 32, generator=generator) * 2.0

    for dtype in formats:
        scale = quant.dynamic_scale(x.reshape(-1), dtype)
        quantised = quant.roundtrip(x, scale, dtype)
        args, kwargs = adapt(ctx.op.name, quantised)
        out = ctx.op.forward(*args, **kwargs)
        args64, kwargs64 = adapt(ctx.op.name, quantised.double())
        expected = ctx.op.forward(*args64, **kwargs64)
        diff = ctx.diff(out.double(), expected)
        metrics.assert_within(diff, rel=ctx.floor(dtype), msg=f"{ctx.op.name}: {dtype}")

    if not hasattr(torch, "float8_e8m0fnu"):
        ctx.note("MXFP8 (E8M0 scale, 32-element block) is not constructible in this build")
