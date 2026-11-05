"""A10 -- what a scale is supposed to mean.

Four properties, each of them a way scales are got wrong in practice: the
explicit scale must be the only one applied, moving the scale and the data by
the same factor must change no bit, quantising one tensor must not disturb
another, and a dynamic scale must use the format's range rather than a corner
of it.
"""

from __future__ import annotations

import torch

from .. import quant
from ..context import RunContext
from ._support import within_half_ulp

ID = "A10"
OPS = None


def run(ctx: RunContext) -> None:
    dtype = ctx.fp8_dtype
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(512, generator=generator) * 3.0
    other = torch.randn(512, generator=generator) * 0.01

    # 1. The scale is applied once, not twice and not zero times.
    scale = quant.dynamic_scale(x, dtype)
    scaled = quant.dequantise(quant.quantise(x, scale, dtype), scale)
    if scaled.abs().max() > 1e3:
        raise AssertionError(
            f"{dtype}: the scale was not divided out (max {scaled.abs().max():.3e})"
        )
    if not within_half_ulp(scaled, x, dtype):
        raise AssertionError(
            f"{dtype}: quantising with an explicit scale lost more than half an ULP"
        )
    if quant.elements_are_byte_equal(
        quant.quantise(x, scale, dtype), quant.quantise(x, 1.0, dtype)
    ):
        raise AssertionError(f"{dtype}: the scale had no effect on the encoding")

    # 2. Scaling the data and the scale by the same factor changes no bit.
    same = quant.quantise(x * 4.0, 1.0, dtype)
    equivalent = quant.quantise(x, 0.25, dtype)
    if not quant.elements_are_byte_equal(same, equivalent):
        raise AssertionError(f"{dtype}: scaling data and scale together changed the encoding")

    # 3. Quantising one tensor must not touch another's encoding.
    alone = quant.quantise(other, 1.0, dtype)
    _ = quant.quantise(x, 1.0, dtype)
    if not quant.elements_are_byte_equal(alone, quant.quantise(other, 1.0, dtype)):
        raise AssertionError(f"{dtype}: quantising one tensor changed another tensor's bytes")

    # 4. A dynamic scale must reach the top of the format, not a fraction of it.
    dynamic = quant.dynamic_scale(x, dtype)
    top = quant.max_finite(dtype)
    reached = float((x / dynamic).abs().max())
    if not reached >= top * 0.98:
        raise AssertionError(
            f"{dtype}: dynamic scale reached {reached:.3f} of {top}, wasting the range"
        )
    ctx.note(f"dynamic scale {float(dynamic):.6f} reached {reached:.1f}/{top}")
