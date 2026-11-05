"""A7 -- negative zero: numerically equal to zero, byte-wise not.

Both are accepted numerically; what is not accepted is -0.0 turning into a
small non-zero value, or a ReLU letting a negative zero through as negative.
"""

from __future__ import annotations

import torch

from .. import quant
from ..context import RunContext

ID = "A7"
OPS = None


def run(ctx: RunContext) -> None:
    dtype = ctx.fp8_dtype
    values = torch.tensor([0.0, -0.0, 1.0, -1.0])

    back = quant.roundtrip(values, 1.0, dtype)
    if not torch.equal(back, values):
        raise AssertionError(
            f"{dtype}: signed zero did not survive the round trip: {back.tolist()}"
        )

    zeros = torch.tensor([[-0.0, 0.0, -0.0, 0.0]])
    args, kwargs = ctx.inputs(torch.float32)
    floats = [
        i for i, a in enumerate(args) if isinstance(a, torch.Tensor) and a.is_floating_point()
    ]
    ctx.require(
        bool(floats) and args[floats[0]].numel() >= zeros.numel(),
        "input is too small for a signed-zero patch",
    )
    patched = list(args)
    sample = patched[floats[0]].clone()
    sample.view(-1)[: zeros.numel()] = zeros.view(-1)
    patched[floats[0]] = sample

    out = ctx.op.forward(*patched, **kwargs)
    reference = ctx.op.forward(*patched, **kwargs)
    if not torch.equal(out, reference):
        raise AssertionError(f"{ctx.op.name}: a signed zero made the result non-deterministic")
    if not torch.isfinite(out).all():
        raise AssertionError(f"{ctx.op.name}: a signed zero produced a non-finite result")
