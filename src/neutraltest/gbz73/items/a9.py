"""A9 -- the mixed-precision path: only one side of the operator in FP8.

This is the deployment shape most of the time: weights quantised, activations
left in float. It must be *at least* as accurate as quantising both, because it
pays one rounding instead of two.
"""

from __future__ import annotations

import torch

from .. import metrics
from ..context import RunContext

ID = "A9"
OPS = None


def run(ctx: RunContext) -> None:
    name = ctx.op.name
    args, kwargs = ctx.inputs(torch.float32)
    floats = [
        i for i, a in enumerate(args) if isinstance(a, torch.Tensor) and a.is_floating_point()
    ]
    if len(floats) < 2:
        ctx.na(f"{name} has a single floating-point input, so there is no mixed arrangement")

    golden = ctx.golden()
    quantised, quantised_kwargs, scale = ctx.quantise_inputs(args, kwargs)

    # Weights in FP8, activations in float: one rounding instead of two.
    mixed = list(quantised)
    for index in floats:
        mixed[index] = args[index] if index == floats[0] else quantised[index]
    out = ctx.op.forward(*mixed, **quantised_kwargs).to(torch.float32)

    diff = ctx.diff(out, golden)
    metrics.assert_within(diff, rel=ctx.floor(), msg=f"{name}: weights-only FP8 vs golden")
    ctx.note(f"weights-only: {diff} scale {float(scale):.6f}")
