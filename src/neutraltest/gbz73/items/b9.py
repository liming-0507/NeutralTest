"""B9 -- the answer must not depend on how the data is distributed.

Gaussian, uniform, heavy-tailed, sparse and constant inputs stress different
parts of the range. Each of them still has the same floor.
"""

from __future__ import annotations

import torch

from .. import metrics, quant
from ..context import RunContext
from ._support import adapt

ID = "B9"
OPS = None


def _distributions(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(256, 32, generator=generator)
    uniform = torch.rand(256, 32, generator=generator) * 2.0 - 1.0
    heavy = torch.exp(torch.randn(256, 32, generator=generator) * 2.0)
    sparse = torch.randn(256, 32, generator=generator) * (
        torch.rand(256, 32, generator=generator) > 0.8
    )
    constant = torch.full((256, 32), 0.25)
    return {
        "gaussian": gaussian,
        "uniform": uniform,
        "heavy": heavy,
        "sparse": sparse,
        "constant": constant,
    }


def run(ctx: RunContext) -> None:
    for label, x in _distributions(ctx.seed).items():
        scale = quant.dynamic_scale(x.reshape(-1), ctx.fp8_dtype)
        quantised = quant.roundtrip(x, scale, ctx.fp8_dtype)
        args, kwargs = adapt(ctx.op.name, quantised)
        out = ctx.op.forward(*args, **kwargs)
        wide, kwargs2 = adapt(ctx.op.name, quantised.double())
        expected = ctx.op.forward(*wide, **kwargs2)
        diff = ctx.diff(out.double(), expected)
        metrics.assert_within(diff, rel=ctx.floor(), msg=f"{ctx.op.name}: {label} inputs")
