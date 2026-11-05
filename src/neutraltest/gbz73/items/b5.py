"""B5 -- the corners: one element, one row, a very wide row, a high rank."""

from __future__ import annotations

import torch

from ..context import RunContext
from ._support import adapt

ID = "B5"
OPS = None


def run(ctx: RunContext) -> None:
    shapes: list[tuple[int, ...]] = [(1, 1), (1, 1024), (64, 1)]
    if ctx.op.name != "grouped_gemm":
        shapes.append((2, 3, 4, 16))

    for shape in shapes:
        generator = torch.Generator().manual_seed(ctx.seed)
        x = torch.randn(*shape, generator=generator)
        args, kwargs = adapt(ctx.op.name, x)
        out = ctx.op.forward(*args, **kwargs)
        wide, kwargs2 = adapt(ctx.op.name, x.double())
        expected = ctx.op.forward(*wide, **kwargs2)
        diff = ctx.diff(out.double(), expected)
        if not diff.rel <= 1e-5:
            raise AssertionError(f"{ctx.op.name}: shape {shape} disagrees with float64 ({diff})")
