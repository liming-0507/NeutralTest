"""F4 -- the host-side cost of making the call at all.

A tiny problem whose time is dominated by argument handling is the cheapest way
to see launch overhead without a profiler.
"""

from __future__ import annotations

import torch

from ..context import RunContext
from ._support import measure

ID = "F4"
OPS = None


def run(ctx: RunContext) -> None:
    if ctx.op.name != "grouped_gemm":
        ctx.na("the clause marks host-side overhead for the grouped operator only")

    generator = torch.Generator().manual_seed(ctx.seed)
    a = torch.randn(16, 4, generator=generator)
    b = torch.randn(4, 4, 4, generator=generator)
    offsets = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32)
    seconds = measure(lambda: ctx.op.forward(a, b, offsets))
    # Half a millisecond of bookkeeping for 16 rows would be a real problem.
    if not seconds <= 5e-4:
        raise AssertionError(
            f"grouped_gemm: a 16-row call costs {seconds * 1e6:.1f} us on the host"
        )
    ctx.note(f"16-row call: {seconds * 1e6:.1f} us")
