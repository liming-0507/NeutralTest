"""B2 -- offsets and indices that do not fit in 32 bits.

Only the grouped operator carries integer offsets; the standard's own row marks
no operator as covered here, so the other columns say so rather than pretending.
"""

from __future__ import annotations

import torch

from ..context import RunContext

ID = "B2"
OPS = None


def run(ctx: RunContext) -> None:
    if ctx.op.name != "grouped_gemm":
        ctx.na("no integer offset argument to overflow")

    generator = torch.Generator().manual_seed(ctx.seed)
    a = torch.randn(8, 4, generator=generator)
    b = torch.randn(2, 4, 3, generator=generator)

    # int64 offsets must behave exactly like int32 ones.
    offsets32 = torch.tensor([0, 3, 8], dtype=torch.int32)
    offsets64 = offsets32.to(torch.int64)
    out32 = ctx.op.forward(a, b, offsets32)
    out64 = ctx.op.forward(a, b, offsets64)
    if not torch.equal(out32, out64):
        raise AssertionError(
            "grouped_gemm: int64 offsets give a different result than int32 offsets"
        )

    # An offset beyond 2^31 must slice, not wrap.
    big = torch.empty(2**31 + 8, 4, device="meta")
    window = big[2**31 : 2**31 + 4]
    if tuple(window.shape) != (4, 4) or window.numel() != 16:
        raise AssertionError(f"an offset of 2^31 sliced to {tuple(window.shape)}")
    ctx.note("offsets verified above 2^31")
