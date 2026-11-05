"""A3 -- what happens above the largest finite value.

A finite input must produce a finite output: a format without an infinity has
to saturate at its maximum, and a silent NaN (or a wrap-around) is the failure
this row exists to catch.
"""

from __future__ import annotations

import torch

from .. import quant
from ..context import RunContext

ID = "A3"
OPS = None


def run(ctx: RunContext) -> None:
    dtype = ctx.fp8_dtype
    top = quant.max_finite(dtype)
    values = torch.tensor([top, top * 1.0001, top + 0.5, top * 2.0, top * 1000.0, -top * 1000.0])

    back = quant.roundtrip(values, 1.0, dtype)
    if torch.isnan(back).any():
        raise AssertionError(f"{dtype}: a finite input became NaN on saturation: {back.tolist()}")
    if torch.isinf(back).any():
        raise AssertionError(f"{dtype}: a finite input became Inf: {back.tolist()}")
    if not torch.all(back.abs() <= top):
        raise AssertionError(f"{dtype}: output left the representable range: {back.tolist()}")
    if not torch.equal(back[1:4], torch.full((3,), top)):
        raise AssertionError(
            f"{dtype}: over-range values did not saturate at {top}: {back[1:4].tolist()}"
        )
    if not back[-1] == -top:
        raise AssertionError(
            f"{dtype}: negative saturation gave {back[-1].item()}, expected {-top}"
        )
