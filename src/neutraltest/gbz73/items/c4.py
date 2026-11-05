"""C4 -- the combinations a caller actually writes: which side is FP8, which
side is float, and what the compute dtype is."""

from __future__ import annotations

import torch

from .. import metrics, quant
from ..context import RunContext
from ._support import adapt

ID = "C4"
OPS = None


def run(ctx: RunContext) -> None:
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(64, 32, generator=generator)
    args64, kwargs64 = adapt(ctx.op.name, x.double())
    golden = ctx.op.forward(*args64, **kwargs64)
    floor = ctx.floor()

    combos = {
        "E4M3 in, E4M3 out": (torch.float8_e4m3fn, torch.float8_e4m3fn),
        "E5M2 in, E4M3 out": (torch.float8_e5m2, torch.float8_e4m3fn),
        "E4M3 in, float out": (torch.float8_e4m3fn, None),
        "E5M2 in, E5M2 out": (torch.float8_e5m2, torch.float8_e5m2),
    }
    for label, (in_dtype, out_dtype) in combos.items():
        if not hasattr(torch, str(in_dtype).replace("torch.", "")):
            continue
        scale = quant.dynamic_scale(x.reshape(-1), in_dtype)
        quantised = quant.roundtrip(x, scale, in_dtype)
        args, kwargs = adapt(ctx.op.name, quantised)
        out = ctx.op.forward(*args, **kwargs).to(torch.float32)
        if out_dtype is not None:
            # The output has its own magnitude, so it gets its own scale; reusing
            # the input scale would saturate every result.
            out = quant.roundtrip(out, quant.dynamic_scale(out.reshape(-1), out_dtype), out_dtype)
        diff = ctx.diff(out.double(), golden)
        # The floor of the coarser of the two formats.
        coarser = min((d for d in (in_dtype, out_dtype) if d is not None), key=quant.mantissa_bits)
        metrics.assert_within(
            diff, rel=max(floor, ctx.floor(coarser)), msg=f"{ctx.op.name}: {label}"
        )
        if not torch.isfinite(out).all():
            raise AssertionError(f"{ctx.op.name}: {label} produced a non-finite result")
