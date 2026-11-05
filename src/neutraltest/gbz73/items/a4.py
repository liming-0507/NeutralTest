"""A4 -- below the smallest normal number: subnormals, then zero.

Two properties are checked: the conversion stays monotone on the way down (a
larger magnitude never becomes the smaller one), and everything below the
smallest subnormal collapses to zero rather than to noise.
"""

from __future__ import annotations

import torch

from .. import quant
from ..context import RunContext

ID = "A4"
OPS = None


def run(ctx: RunContext) -> None:
    dtype = ctx.fp8_dtype
    tiny = quant.min_subnormal(dtype)
    smallest_normal = float(torch.finfo(dtype).tiny)

    values = torch.tensor(
        [0.0, tiny / 4.0, tiny / 2.0, tiny, tiny * 2.0, smallest_normal * 0.5, smallest_normal]
    )
    back = quant.roundtrip(values, 1.0, dtype)

    if not torch.equal(back[:3], torch.zeros(3)):
        raise AssertionError(
            f"{dtype}: below the smallest subnormal gave {back[:3].tolist()}, expected zeros"
        )
    if not torch.equal(back[3], torch.tensor(tiny)):
        raise AssertionError(
            f"{dtype}: the smallest subnormal round-tripped to {back[3].item()}, expected {tiny}"
        )
    if torch.any(back[1:] < back[:-1]):
        raise AssertionError(f"{dtype}: conversion is not monotone near underflow: {back.tolist()}")
    if not torch.all(back.abs() <= values.abs() + tiny):
        raise AssertionError(f"{dtype}: underflow amplified a value: {back.tolist()}")
