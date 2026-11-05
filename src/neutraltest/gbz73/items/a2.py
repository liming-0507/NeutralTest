"""A2 -- the quantise/dequantise round trip on its own.

The standard's operator clause compares a quantised result with a float32
reference and therefore cannot tell "the half-ULP the format costs" from "the
operator is wrong". Isolating the round trip gives that floor, and the check
here is that the floor is actually met: a conversion that rounds correctly can
never be further than half an ULP away, per element.
"""

from __future__ import annotations

import torch

from .. import quant
from ..context import RunContext
from ._support import within_half_ulp

ID = "A2"
OPS = None


def run(ctx: RunContext) -> None:
    generator = torch.Generator().manual_seed(ctx.seed)
    sample = torch.randn(4096, generator=generator) * 0.5
    dtype = ctx.fp8_dtype

    back = quant.roundtrip(sample, 1.0, dtype)
    diff = ctx.diff(back, sample)
    bound = quant.half_ulp(dtype)

    if not within_half_ulp(back, sample, dtype):
        raise AssertionError(
            f"{dtype}: an element moved further than half an ULP from its exponent bucket"
        )
    if not diff.rel < bound:
        raise AssertionError(
            f"{dtype}: aggregate error {diff.rel:.4e} is not below half an ULP {bound:.4e}"
        )

    # The floor is real: values that are not representable must move.
    if diff.exact:
        raise AssertionError(
            f"{dtype}: the round trip was exact, so the sample never left the format"
        )
    ctx.note(f"round trip: rel={diff.rel:.4e} max_rel={diff.max_rel:.4e} (half-ULP {bound:.4e})")
