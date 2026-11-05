"""E1 -- run it twice, get the same bytes.

Determinism has to hold for the operator and for the harness around it: same
inputs, same seed, same result -- bit for bit, not within tolerance.
"""

from __future__ import annotations

import torch

from ..context import RunContext

ID = "E1"
OPS = None


def _device_run(ctx: RunContext):
    args, kwargs = ctx.inputs(torch.float32)
    return ctx.run(args, kwargs)


def run(ctx: RunContext) -> None:
    first = _device_run(ctx)
    second = _device_run(ctx)
    if not torch.equal(first, second):
        worst = float((first - second).abs().max()) if first.shape == second.shape else float("nan")
        raise AssertionError(f"{ctx.op.name}: two identical runs differ (max {worst:.3e})")

    # The generators must not leak state into each other either: a fresh
    # context with the same seed must rebuild the same inputs.
    first_args, first_kwargs = ctx.inputs(torch.float32)
    second_args, second_kwargs = ctx.inputs(torch.float32)
    for left, right in zip(first_args, second_args, strict=True):
        if isinstance(left, torch.Tensor) and not torch.equal(left, right):
            raise AssertionError(f"{ctx.op.name}: rebuilding the inputs gave different values")
    if first_kwargs != second_kwargs:
        raise AssertionError(
            f"{ctx.op.name}: rebuilding the keyword arguments gave different values"
        )
