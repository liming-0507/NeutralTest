"""A5 -- the rounding rule at the midpoint.

Check: the eight midpoints between consecutive E4M3 values in ``[1, 2)`` are fed
through quantise, and each must land on the even mantissa -- round-to-nearest,
ties-to-even is the rule the clause names. Values a quarter of an ulp either side
of a representable value must then land back on it, so an implementation that
rounds half-up passes the table but fails the offsets, and one that truncates
fails both. The cache column is a copy, not an arithmetic kernel, so its rounding
rule is "none": the rows it stores must come back bit-identical.

Tolerance: exact. Rounding is a discrete decision; there is no error budget to
spend, and any deviation is a different rule rather than a looser one.
"""

from __future__ import annotations

from typing import Any

import torch

from ...gbz73 import quant
from .._support import make_inputs, observe

ID = "A5"
COLUMNS = None

EXPECTED = (1.0, 1.25, 1.25, 1.5, 1.5, 1.75, 1.75, 2.0)
REPRESENTABLE = (1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875)


def run(ctx: Any) -> None:
    if ctx.family == "kv_cache":
        _cache(ctx)
        return
    dtype = quant.FP8_DTYPES[0]
    if dtype is not torch.float8_e4m3fn:  # pragma: no cover - only E4M3 has this table
        ctx.na(f"the tie table is written for float8_e4m3fn, not {dtype}")
    offsets = [(2 * index + 1) / 16.0 for index in range(len(EXPECTED))]
    ties = torch.tensor([1.0 + offset for offset in offsets], dtype=torch.float32)
    rounded = quant.dequantise(quant.quantise(ties, 1.0, dtype), 1.0)
    expected = torch.tensor(EXPECTED, dtype=torch.float32)
    if not torch.equal(rounded, expected):
        raise AssertionError(
            f"{ctx.column.name}: ties rounded to {rounded.tolist()}, round-to-nearest-even "
            f"requires {expected.tolist()}"
        )
    for sign in (-1.0, 1.0):
        base = torch.tensor([sign * value for value in REPRESENTABLE], dtype=torch.float32)
        probe = base + sign * 0.03125
        got = quant.dequantise(quant.quantise(probe, 1.0, dtype), 1.0)
        if not torch.equal(got, base):
            raise AssertionError(
                f"{ctx.column.name}: a quarter-ulp offset rounded to {got.tolist()}, "
                f"not back to {base.tolist()}"
            )


def _cache(ctx: Any) -> None:
    args, kwargs = make_inputs(family="kv_cache", seed=ctx.seed)
    keys, values, cache_k, cache_v, index = args
    written_k, written_v = observe(ctx, args, kwargs)
    for row, slot in enumerate(int(value) for value in index):
        if not torch.equal(written_k[slot], keys[row]):
            raise AssertionError(
                f"{ctx.column.name}: row {slot} of the key cache was rounded on the way in"
            )
        if not torch.equal(written_v[slot], values[row]):
            raise AssertionError(
                f"{ctx.column.name}: row {slot} of the value cache was rounded on the way in"
            )
    if not torch.equal(cache_k, args[2]) or not torch.equal(cache_v, args[3]):
        raise AssertionError(f"{ctx.column.name}: the caller's cache was modified")
