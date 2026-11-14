"""A6 -- NaN and infinity propagate the way the reference propagates them.

Check: one element of the first operand is poisoned with NaN, then with +inf,
then with -inf, and the result is compared with the reference on the same poisoned
input. A kernel that writes a sentinel over the poison (or clamps it back into
range) reports a finite value where the reference is non-finite; a kernel that
lets a NaN leak out of a row that does not contain one is worse than the
framework. Both are visible in the finiteness pattern, which is compared
elementwise, plus the requirement that a NaN in the input reaches the output at
least once.

Tolerance: none on the pattern -- finiteness is a bit-level property -- and the
finite entries are not compared here: A1 owns their values.

One exception to "a NaN in, a NaN out": for attention the reference resolves the
softmax to no NaN at all when the NaN sits in ``v``. The scores are finite, the
weights are finite, and ``0 x NaN`` is NaN only if the zero is an exact zero --
so the operator's answer depends on whether a fully masked score exists. That is
the framework's own behaviour on the same call, not a kernel's, so the row asks
only for pollution of ``q``/``k`` (where every implementation must give up on
the whole row) and records the ``v`` case instead of failing on it.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import budget, clone_args, close, make_inputs, observe, reference

ID = "A6"
COLUMNS = None

POISONS = (float("nan"), float("inf"), float("-inf"))


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    clean = observe(ctx, args, kwargs)[0]
    if not bool(torch.isfinite(clean).all()):
        ctx.na("the unpoisoned output is already non-finite, so the poison cannot be told apart")
    for poison in POISONS:
        poisoned = list(clone_args(args))
        first = poisoned[0]
        if not isinstance(first, torch.Tensor):  # pragma: no cover - defensive
            continue
        hit = first.reshape(-1)[0]
        first.reshape(-1)[0] = hit * 0.0 + poison
        if not _must_go_nan(ctx, poisoned, kwargs, poison):
            ctx.note(
                f"a {poison} in the value operand leaves the reference finite on this call; "
                f"the pattern row does not claim it"
            )
            continue
        frozen = tuple(poisoned)
        got = observe(ctx, frozen, kwargs)[0]
        want = reference(ctx, frozen, kwargs, dtype=torch.float32)[0]
        if not torch.equal(torch.isnan(got), torch.isnan(want)):
            raise AssertionError(
                f"{ctx.column.name}: a {poison} input left a NaN pattern "
                f"{_pattern(got)} where the reference has {_pattern(want)}"
            )
        if not torch.equal(torch.isinf(got), torch.isinf(want)):
            raise AssertionError(
                f"{ctx.column.name}: a {poison} input left an infinity pattern "
                f"{_pattern(got)} where the reference has {_pattern(want)}"
            )
        if poison != poison and not bool(torch.isnan(got).any()):  # NaN input, NaN output
            raise AssertionError(f"{ctx.column.name}: a NaN input produced no NaN in the output")
    if not bool(torch.isfinite(observe(ctx, args, kwargs)[0]).all()):
        raise AssertionError(f"{ctx.column.name}: a second clean call is not finite")
    close(
        observe(ctx, args, kwargs),
        reference(ctx, args, kwargs, dtype=torch.float32),
        rel=budget(ctx),
        msg=f"{ctx.column.name}: clean run",
    )


def _must_go_nan(ctx: Any, args: Any, kwargs: Any, poison: float) -> bool:
    """Whether a reference that is finite here would still make this row meaningful."""
    if ctx.family != "flash_attention" or poison == poison:
        return True
    # The NaN sits in v (the third operand): the reference's own answer decides.
    want = reference(ctx, tuple(args), kwargs, dtype=torch.float32)[0]
    return bool(torch.isnan(want).any())


def _pattern(value: torch.Tensor) -> str:
    return (
        f"nan {int(torch.isnan(value).sum())}, "
        f"inf {int(torch.isinf(value).sum())}, n {value.numel()}"
    )
