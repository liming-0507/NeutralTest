"""A5 -- the rounding rule, tested at the ties.

Round-to-nearest-even has one signature: at an exact midpoint the result keeps
the even mantissa. A truncating conversion passes a plain accuracy check and
fails here, which is why the tie cases are constructed explicitly.
"""

from __future__ import annotations

import torch

from .. import quant
from ..context import RunContext
from ._support import within_half_ulp

ID = "A5"
OPS = None


def _mantissa_steps(value: float, dtype: torch.dtype) -> tuple[float, float]:
    """The two neighbouring representable values around ``value`` and the midpoint."""
    down = torch.tensor([value], dtype=torch.float32)
    step = abs(float(torch.tensor(value * 2.0 ** -quant.mantissa_bits(dtype), dtype=torch.float32)))
    low = float(quant.roundtrip(down, 1.0, dtype)[0])
    return low, low + step


def run(ctx: RunContext) -> None:
    dtype = ctx.fp8_dtype
    bits = quant.mantissa_bits(dtype)

    # 1.0 + half an ULP is the classic tie: it must land on the even neighbour.
    low, high = 1.0, 1.0 + 2.0**-bits
    midpoint = (low + high) / 2.0
    got = float(quant.roundtrip(torch.tensor([midpoint]), 1.0, dtype)[0])
    if got != low:
        raise AssertionError(
            f"{dtype}: tie {midpoint!r} rounded to {got!r}, expected the even neighbour {low!r}"
        )

    # Every value is within half an ULP; truncation would show ~1 ULP.
    generator = torch.Generator().manual_seed(ctx.seed)
    sample = torch.rand(8192, generator=generator) * 4.0 - 2.0
    back = quant.roundtrip(sample, 1.0, dtype)
    if not within_half_ulp(back, sample, dtype):
        worst = float((back - sample).abs().max())
        raise AssertionError(
            f"{dtype}: an element moved {worst:.4e} (> half an ULP {quant.half_ulp(dtype):.4e}); "
            "the conversion is not round-to-nearest"
        )
