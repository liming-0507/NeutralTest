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

from .._support import budget, close, make_inputs, observe, reference

ID = "C5"
COLUMNS = None

PAIRS = (
    (torch.float16, torch.float16),
    (torch.float16, torch.float32),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.float32),
)


def _narrow(first: torch.dtype, second: torch.dtype) -> torch.dtype:
    return first if first.itemsize <= second.itemsize else second


def run(ctx: Any) -> None:
    for first, second in PAIRS:
        args, kwargs = make_inputs(
            family=ctx.family, dtype=first, seed=ctx.seed, variant=ctx.variant
        )
        mixed = (args[0], args[1].to(second))
        produced = observe(ctx, mixed, kwargs)
        promoted = torch.promote_types(first, second)
        if produced[0].dtype != promoted:
            raise AssertionError(
                f"{ctx.column.name}: {first} with {second} returned {produced[0].dtype}, "
                f"not {promoted}"
            )
        close(
            produced,
            reference(ctx, mixed, kwargs, dtype=torch.float64),
            rel=budget(ctx, _narrow(first, second)),
            msg=f"{ctx.column.name}: {first} with {second}",
        )
