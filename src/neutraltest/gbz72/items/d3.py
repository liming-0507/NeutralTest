"""D3 -- aliasing and in-place behaviour.

Check: the rotation is called with a view of its input and with a copy, and both
must produce the same tensor without either call writing to the caller's storage.
The returned tensor must not share storage with the input, the position tensor
must come back untouched, and calling the same view twice must give the same
answer, which rules out a kernel that caches its rotated output on the input. A
backward column is checked the same way: the analytic VJP must not modify its
arguments, and two consecutive calls must return identical gradients.

Tolerance: exact. Aliasing and mutation are structural facts, so the comparison is
bit for bit; there is no rounding difference to absorb between a view and a copy
of the same values.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import clone_args, flat, jacobian, make_inputs, observe

ID = "D3"
COLUMNS = None


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    before = clone_args(args)
    if ctx.backward:
        first, _ = ctx.gradients_with(args, kwargs)
        second, _ = ctx.gradients_with(args, kwargs)
        _unchanged(ctx, args, before)
        if not torch.equal(flat(first), flat(second)):
            raise AssertionError(
                f"{ctx.column.name}: two identical calls returned different gradients"
            )
        jacobian(ctx, args, kwargs)
        return
    produced = observe(ctx, args, kwargs)
    _unchanged(ctx, args, before)
    value = produced[0]
    if value.data_ptr() == args[0].data_ptr():
        raise AssertionError(f"{ctx.column.name}: the result aliases the input")
    view = args[0].view_as(args[0])
    again = observe(ctx, (view, args[1]), kwargs)
    if any(not torch.equal(a, b) for a, b in zip(again, produced, strict=True)):
        raise AssertionError(f"{ctx.column.name}: the view and the original disagreed")
    copied = observe(ctx, (args[0].clone(), args[1].clone()), kwargs)
    if any(not torch.equal(a, b) for a, b in zip(copied, produced, strict=True)):
        raise AssertionError(f"{ctx.column.name}: the copy and the original disagreed")


def _unchanged(ctx: Any, args: Any, before: Any) -> None:
    for current, original in zip(args, before, strict=True):
        if isinstance(current, torch.Tensor) and not torch.equal(current, original):
            raise AssertionError(f"{ctx.column.name}: the call modified its input")
