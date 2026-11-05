"""E4 -- memory safety, on the evidence a CPU run can produce.

Bounds checking needs a sanitiser on the accelerator, which this environment
does not have. What it can show is the part that is a property of the code:
a write never escapes into the inputs, and an out-of-range access is refused
rather than silently wrapped.
"""

from __future__ import annotations

import torch

from ..context import RunContext
from ._support import adapt

ID = "E4"
OPS = None


def run(ctx: RunContext) -> None:
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(32, 16, generator=generator)
    guard = torch.full((32, 16), 1234.5)
    args, kwargs = adapt(ctx.op.name, x)
    out = ctx.op.forward(*args, **kwargs)
    if not bool((guard == 1234.5).all()):
        raise AssertionError(f"{ctx.op.name}: a write escaped into memory outside the result")

    # An index past the end must be refused, not wrapped.
    if ctx.op.name == "grouped_gemm":
        a = torch.randn(4, 4, generator=generator)
        b = torch.randn(1, 4, 4, generator=generator)
        try:
            ctx.op.forward(a, b, torch.tensor([0, 2**20], dtype=torch.int64))
        except (RuntimeError, ValueError, IndexError):
            pass
        else:
            raise AssertionError("grouped_gemm: an offset past the end of the tensor was accepted")

    if out.numel() and not torch.isfinite(out).all():
        raise AssertionError(
            f"{ctx.op.name}: the result contains non-finite values for finite input"
        )
