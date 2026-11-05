"""B4 -- shapes that are not powers of two.

Unaligned widths are the normal case in a real model and the normal source of
padding bugs: the operator must agree with the float64 computation on a width
that is not a multiple of anything.
"""

from __future__ import annotations

import torch

from ..context import RunContext
from ._support import adapt

ID = "B4"
OPS = None


def run(ctx: RunContext) -> None:
    for shape in ((3, 37), (7, 13), (1, 977), (5, 17)):
        generator = torch.Generator().manual_seed(ctx.seed + shape[-1])
        x = torch.randn(*shape, generator=generator)
        args, kwargs = adapt(ctx.op.name, x)
        out = ctx.op.forward(*args, **kwargs)

        wide, kwargs2 = adapt(ctx.op.name, x.double())
        expected = ctx.op.forward(*wide, **kwargs2)
        diff = ctx.diff(out.double(), expected)
        if not diff.rel <= 1e-5:
            raise AssertionError(f"{ctx.op.name}: shape {shape} disagrees with float64 ({diff})")
