"""C5 -- the result dtype follows the framework's promotion rule.

Check: every pair of operand widths is fed to the column and the result dtype is
compared with ``torch.promote_types``. A kernel that always widens to float32
returns the right values and the wrong dtype; one that keeps the first operand's
width silently drops the second operand's precision. Both are visible here
without relying on the numbers.

Tolerance: no numeric comparison for the dtype itself -- the promotion rule is
exact. The values are compared at the *narrow* operand's budget, not the promoted
one: a kernel that legitimately does its arithmetic in the narrow width and
promotes the result on the way out carries the narrow width's rounding, and
holding it to the wide budget would demand arithmetic the framework itself does
not promise.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import budget, close, make_inputs, observe, reference, value_slots

ID = "C5"
COLUMNS = None

PAIRS = (
    (torch.float16, torch.float16),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.float32),
)

#: Families whose operands are required to agree on dtype by the operation
#: itself, so there is no mixed-width call to make.  Attention's q/k/v must share
#: a width (the scores are one matmul), and the cache write's ``index_put``
#: demands that the source and the destination agree.  The column reports this
#: rather than fabricating a call the framework is right to reject.
SINGLE_WIDTH: dict[str, str] = {
    "flash_attention": "query, key and value must share a width before the scores can be formed",
    "kv_cache": "the slot write requires the new tokens and the cache rows to share a width",
}


def _narrow(first: torch.dtype, second: torch.dtype) -> torch.dtype:
    return first if first.itemsize <= second.itemsize else second


def run(ctx: Any) -> None:
    slots = value_slots(ctx.family)
    if len(slots) < 2:
        # One value operand means one width to choose: there is no pair to
        # promote, and the column's own answer (the output dtype) is C1's.
        ctx.note(f"{ctx.family} combines a single value operand; there is no pair to promote")
        return
    if ctx.family in SINGLE_WIDTH:
        ctx.note(f"mixed widths are not a call this operator accepts: {SINGLE_WIDTH[ctx.family]}")
        return
    for first, second in PAIRS:
        args, kwargs = make_inputs(
            family=ctx.family, dtype=first, seed=ctx.seed, variant=ctx.variant
        )
        # Move every value operand but the first to the second width; the rest
        # of the call -- positions, slot indices, the cache being written into --
        # keeps the width it was built with, because mixing *its* dtype is a
        # different question (C4) and changing it here would confound the two.
        widened = list(args)
        for slot in slots[1:]:
            widened[slot] = widened[slot].to(second)
        mixed = tuple(widened)
        answer = observe(ctx, mixed, kwargs)
        promoted = torch.promote_types(first, second)
        if answer[0].dtype != promoted:
            raise AssertionError(
                f"{ctx.column.name}: {first} with {second} returned {answer[0].dtype}, "
                f"not {promoted}"
            )
        close(
            answer,
            reference(ctx, mixed, kwargs, dtype=torch.float64),
            rel=budget(ctx, _narrow(first, second)),
            msg=f"{ctx.column.name}: {first} with {second}",
        )
