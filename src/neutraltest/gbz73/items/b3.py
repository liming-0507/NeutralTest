"""B3 -- zero elements, one element, one row.

Degenerate shapes are where shape arithmetic and reductions over an empty axis
break. An empty input may be refused, but it must not return a wrongly shaped
tensor.
"""

from __future__ import annotations

import torch

from ..context import RunContext
from ._support import adapt

ID = "B3"
OPS = None


def _expected_shape(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    if name == "amax":
        return tuple(shape[:-1])
    return tuple(shape)


def _check(ctx: RunContext, x: torch.Tensor) -> None:
    args, kwargs = adapt(ctx.op.name, x)
    try:
        out = ctx.op.forward(*args, **kwargs)
    except (RuntimeError, ValueError, IndexError) as exc:
        ctx.note(f"refused {tuple(x.shape)}: {type(exc).__name__}")
        return
    expected = _expected_shape(ctx.op.name, tuple(x.shape))
    if tuple(out.shape) != expected:
        raise AssertionError(
            f"{ctx.op.name}: {tuple(x.shape)} gave {tuple(out.shape)}, expected {expected}"
        )


def run(ctx: RunContext) -> None:
    width = 8
    _check(ctx, torch.empty(0, width))
    _check(ctx, torch.zeros(1, width))
    _check(ctx, torch.zeros(0, 1))
    _check(ctx, torch.zeros(1, 1))
    if ctx.op.name == "grouped_gemm":
        out = ctx.op.forward(
            torch.ones(0, width),
            torch.ones(1, width, width),
            torch.tensor([0, 0], dtype=torch.int32),
        )
        if out.numel() != 0:
            raise AssertionError(f"grouped_gemm: an empty group gave {tuple(out.shape)}")
