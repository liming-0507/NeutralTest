"""E7 -- the operator runs on the deployment it was measured on.

Deployment is not a version string: the clause asks whether what ships can be
asked to run again and give the same answer. Recording the versions and the FP8
format is part of that -- but only part. A row that printed versions and called
nothing would pass on a machine where the operator does not load at all, which
is the one condition it exists to catch.

So the row calls the operator twice on the same inputs and requires the same
bits, then records the environment the two calls came from. Bit equality is
right here: this is the same process, the same device, the same arguments, so a
difference is a hidden state the deployment carries between calls -- a cached
workspace, a counter, an uninitialised buffer.
"""

from __future__ import annotations

import platform

import torch

from .. import quant
from ..context import RunContext

ID = "E7"
OPS = None


def run(ctx: RunContext) -> None:
    half_ulp = quant.half_ulp(ctx.fp8_dtype)
    args, kwargs = ctx.inputs(torch.float32)
    first = ctx.run(args, kwargs)
    if not bool(torch.isfinite(first).all()):
        raise AssertionError(f"{ctx.op.name}: the first deployment call produced non-finite values")
    second = ctx.run(args, kwargs)
    if not torch.equal(first, second):
        worst = (first - second).abs().max().item()
        raise AssertionError(
            f"{ctx.op.name}: two deployment calls on the same inputs differ by up to {worst:.3e}"
        )

    if not hasattr(torch, "use_deterministic_algorithms"):
        raise AssertionError("this torch cannot be asked for deterministic algorithms")
    if not half_ulp > 0:
        raise AssertionError(f"{ctx.fp8_dtype}: half an ULP is {half_ulp}")
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "backend": ctx.backend.name,
        "device_type": ctx.backend.device_type,
        "fp8": str(ctx.fp8_dtype),
        "half_ulp": half_ulp,
        "max_finite": quant.max_finite(ctx.fp8_dtype),
    }
    ctx.note(" ".join(f"{k}={v}" for k, v in info.items()) + " repeatable=yes")
