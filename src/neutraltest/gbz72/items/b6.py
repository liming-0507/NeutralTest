"""B6 -- broadcasting, and shapes that must not broadcast.

Check: the elementwise add accepts a second operand of shape ``(width,)``, which
the framework broadcasts over the rows, and must produce the same values as the
explicit ``(1, width)`` spelling; the written rows of the cache must accept a
cache wider than the index touches. A second operand whose trailing size does not
match the row, and a key whose width does not match the cache, must be refused
rather than silently re-shaped.

Tolerance: exact where the two spellings perform the same arithmetic (a
broadcast is an indexing decision, not a rounding one), and the column's float32
budget where a differently shaped but equivalent expression is compared with the
reference.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import budget, close, make_inputs, observe, reference

ID = "B6"
COLUMNS = None


def run(ctx: Any) -> None:
    if ctx.family == "add":
        _add(ctx)
    elif ctx.family == "kv_cache":
        _cache(ctx)
    else:
        ctx.na("broadcasting is only defined for the elementwise add and the cache widths")


def _add(ctx: Any) -> None:
    args, kwargs = make_inputs(family="add", seed=ctx.seed)
    a, b = args
    row = b.reshape(1, -1)
    broad = observe(ctx, args, kwargs)[0]
    explicit = observe(ctx, (a, row), kwargs)[0]
    if not torch.equal(broad, explicit):
        raise AssertionError(f"{ctx.column.name}: (width,) and (1, width) disagree")
    alpha = float(kwargs.get("alpha", 1.0))
    # Subtracting the first operand recovers the broadcast term only up to the
    # rounding of the add, so the recovery is held to the column budget rather
    # than to bit equality.
    close(
        broad - a,
        (b * alpha).reshape(1, -1).expand_as(broad),
        rel=budget(ctx),
        msg=f"{ctx.column.name}: the broadcast term is not constant across rows",
    )
    tall = observe(ctx, (a[:1], b), kwargs)[0]
    close(
        tall,
        reference(ctx, (a[:1], b), kwargs, dtype=torch.float32)[0],
        rel=budget(ctx),
        msg=f"{ctx.column.name}: one row against many rows",
    )
    try:
        observe(ctx, (a, b[:3]), kwargs)
    except (RuntimeError, ValueError):
        return
    raise AssertionError(f"{ctx.column.name}: a (3,) operand was accepted for a (8, 16) tensor")


def _cache(ctx: Any) -> None:
    args, kwargs = make_inputs(family="kv_cache", seed=ctx.seed, rows=32)
    keys, values, cache_k, cache_v, index = args
    wider = observe(ctx, args, kwargs)
    if not torch.equal(wider[0][index.to(torch.long)], keys):
        raise AssertionError(f"{ctx.column.name}: a wide cache did not accept the write")
    narrow = keys[:, : keys.shape[1] // 2]
    try:
        observe(ctx, (narrow, values, cache_k, cache_v, index), kwargs)
    except (RuntimeError, ValueError):
        return
    raise AssertionError(f"{ctx.column.name}: a key narrower than the cache row was accepted")
