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

from .._support import clone_args, flat, jacobian, make_inputs, observe, value_slots

ID = "D3"
COLUMNS = None


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    before = clone_args(args)
    slots = value_slots(ctx.family)
    first_value = slots[0]
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
    if value.data_ptr() == args[first_value].data_ptr():
        raise AssertionError(f"{ctx.column.name}: the result aliases the input")
    # A view of the first value operand, then a full copy of the whole call:
    # every argument is swapped, so the kernel cannot tell it apart by identity
    # and a result that depends on the layout of one operand is caught here.
    view = list(args)
    view[first_value] = args[first_value].view_as(args[first_value])
    again = observe(ctx, tuple(view), kwargs)
    if any(not torch.equal(a, b) for a, b in zip(again, produced, strict=True)):
        raise AssertionError(f"{ctx.column.name}: the view and the original disagreed")
    copied = observe(ctx, _cloned(args), kwargs)
    if any(not torch.equal(a, b) for a, b in zip(copied, produced, strict=True)):
        raise AssertionError(f"{ctx.column.name}: the copy and the original disagreed")


def _cloned(args: Any) -> tuple[Any, ...]:
    """A copy of every tensor operand; tensors that are not tensors stay as they are."""
    return tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args)


def _unchanged(ctx: Any, args: Any, before: Any) -> None:
    for current, original in zip(args, before, strict=True):
        if isinstance(current, torch.Tensor) and not torch.equal(current, original):
            raise AssertionError(f"{ctx.column.name}: the call modified its input")
