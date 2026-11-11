"""C4 -- mixed-width operands promote instead of truncating.

Check: a narrow operand is combined with a wide one -- a float16 addend with a
float32 one, a bfloat16 activation with a float32 weight -- and the result is
required to come back in the promoted width. A path that returns the narrow dtype
has silently chosen to store the loss; a path that produces a third dtype has
invented its own promotion rule. The cache is the exception: storing a row means
writing into a cache of a fixed dtype, so a mixed-width write is not a promotion
question at all -- the column must refuse it rather than truncate one side.

Tolerance: the *narrow* operand's budget. The reference is evaluated in float64 on
the already-cast values, so the cast itself is not being measured; what is left is
the arithmetic, and a kernel that computes in the narrow width and promotes at the
end is within the narrow width's rounding, not the wide one's.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import budget, close, make_inputs, observe, reference

ID = "C4"
COLUMNS = None

CASES: dict[str, tuple[tuple[torch.dtype, torch.dtype], ...]] = {
    "add": ((torch.float16, torch.float32), (torch.bfloat16, torch.float32)),
    "kv_cache": ((torch.bfloat16, torch.float32), (torch.float16, torch.float32)),
    "rms_norm": ((torch.bfloat16, torch.float32), (torch.float16, torch.float32)),
}


def _narrow(first: torch.dtype, second: torch.dtype) -> torch.dtype:
    return first if first.itemsize <= second.itemsize else second


def run(ctx: Any) -> None:
    for narrow, wide in CASES[ctx.family]:
        args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
        mixed = _mix(ctx, args, narrow, wide)
        if ctx.family == "kv_cache":
            _refused(ctx, mixed, kwargs, narrow, wide)
            return
        produced = observe(ctx, mixed, kwargs)
        promoted = torch.promote_types(narrow, wide)
        for value in produced:
            if value.dtype != promoted:
                raise AssertionError(
                    f"{ctx.column.name}: {narrow} with {wide} produced {value.dtype}, "
                    f"not {promoted}"
                )
        close(
            produced,
            reference(ctx, mixed, kwargs, dtype=torch.float64),
            rel=budget(ctx, _narrow(narrow, wide)),
            msg=f"{ctx.column.name}: {narrow} with {wide}",
        )


def _refused(ctx: Any, args: Any, kwargs: Any, narrow: torch.dtype, wide: torch.dtype) -> None:
    """The cache write has one storage dtype; a mixed write must be rejected."""
    try:
        observe(ctx, args, kwargs)
    except RuntimeError as exc:
        if "dtype" not in str(exc).lower():
            raise
        ctx.note(f"a {narrow} row into a {wide} cache is refused: {exc}")
        return
    raise AssertionError(
        f"{ctx.column.name}: a {narrow} row was written into a {wide} cache without a refusal"
    )


def _mix(ctx: Any, args: Any, narrow: torch.dtype, wide: torch.dtype) -> Any:
    if ctx.family == "add":
        return (args[0].to(narrow), args[1].to(wide))
    if ctx.family == "kv_cache":
        keys, values, cache_k, cache_v, index = args
        return (keys.to(narrow), values.to(narrow), cache_k.to(wide), cache_v.to(wide), index)
    x, weight = args
    return (x.to(narrow), weight.to(wide))
