"""B8 -- samples are isolated from one another.

Check: one element of one sample is shifted and the rest of the batch is required
to be bit-identical to the unperturbed result. For the attention that means the
other batch entries, for the rotation the other positions and the other batch
entries, and for the row-wise families every other row. A kernel that reduces
across the batch, writes through a shared accumulator, or indexes the wrong
sample shows up here as a change where there should be none. The perturbed region
must differ as well, so that a perturbation which never reaches the output cannot
pass.

The backward columns are judged the same way on the derivative: either the
derivative itself changes only where the perturbed sample lives, or -- for a
column whose derivative does not depend on its input at all -- it must not change
anywhere, which is the stronger statement of the same isolation.

Tolerance: exact. A sample that is not read cannot change, so the unaffected
region must compare equal bit for bit.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import clone_args, make_inputs, observe

ID = "B8"
COLUMNS = None


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    base = _result(ctx, args, kwargs)
    perturbed = clone_args(args)
    _shift(ctx, perturbed)
    changed = _result(ctx, perturbed, kwargs)
    if len(changed) != len(base):
        raise AssertionError(f"{ctx.column.name}: the perturbation changed the number of outputs")
    if all(torch.equal(got, want) for got, want in zip(changed, base, strict=True)):
        if ctx.backward:
            ctx.note("the derivative of this column is constant in its inputs")
            return
        raise AssertionError(f"{ctx.column.name}: the perturbation did not reach the output")
    first_got, first_want = changed[0], base[0]
    for got, want, label in _untouched(ctx, first_got, first_want):
        if not torch.equal(got, want):
            raise AssertionError(
                f"{ctx.column.name}: shifting one sample changed {label}, which does not read it"
            )


def _result(ctx: Any, args: Any, kwargs: Any) -> tuple[torch.Tensor, ...]:
    if ctx.backward:
        gradients, _ = ctx.gradients_with(args, kwargs)
        return tuple(gradients)
    return observe(ctx, args, kwargs)


def _shift(ctx: Any, args: Any) -> None:
    first = args[0]
    if ctx.family in {"flash_attention", "rope"}:
        first[0, 0, 0, 0] = first[0, 0, 0, 0] + 1.0
    else:
        first[0, 0] = first[0, 0] + 1.0


def _untouched(
    ctx: Any, changed: torch.Tensor, base: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
    if ctx.family == "rope":
        return [
            (changed[0, 1:], base[0, 1:], "another position"),
            (changed[1:], base[1:], "another batch entry"),
        ]
    if ctx.family == "flash_attention":
        return [(changed[1:], base[1:], "another batch entry")]
    return [(changed[1:], base[1:], "another sample")]
