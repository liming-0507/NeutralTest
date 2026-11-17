"""B3 -- degenerate shapes.

Check: every operator is asked for the smallest shapes it can be asked for and
the answer is held to the reference on the same input. Three degenerate forms
per family: an empty leading dimension, a single element, and -- where the
operator has a reduction or a matmul along an axis -- a length-one axis. A
kernel that hard-codes a tile size, assumes a non-empty batch, or divides by a
length it never checked fails on one of them; one that quietly returns an empty
tensor of the wrong shape fails on the shape comparison without needing numbers.

An empty tensor is the interesting case rather than the trivial one: it is where
a launcher computes a grid dimension of zero and either refuses to launch or
launches a block that reads past the end of nothing. A single element is where a
reduction has nothing to reduce and a normalisation divides by its own epsilon.

Tolerance: the column's budget, the same as the reference agreement it is a
special case of. The comparison is on the whole result, so shape is checked
before values, and an operator that returns a correctly-shaped empty tensor
passes without a numeric comparison.

A family whose shapes cannot degenerate -- the cache needs at least the rows it
is asked to write -- reports the limit it hit instead of claiming the shape was
tested.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import budget, close, make_inputs, observe, reference

ID = "B3"
COLUMNS = None

#: (label, keyword overrides) tried against ``make_inputs``. The labels are what
#: a failure says, so they describe the shape rather than the call.
CASES: tuple[tuple[str, dict[str, int]], ...] = (
    ("an empty leading dimension", {"batch": 0}),
    ("a single element", {"batch": 1, "heads": 1, "seq": 1, "rows": 1, "width": 2, "tokens": 1}),
    ("a single row", {"batch": 1, "heads": 1, "seq": 1, "rows": 1, "width": 2, "tokens": 1}),
)

#: The narrowest width each family accepts. The rotation works on pairs, so its
#: axis must be even and one is the narrowest *width* it can be asked for -- the
#: single-element case is still one element per row, not one column.
WIDTH_FLOOR: dict[str, int] = {"rope": 2}

#: Families whose shape cannot shrink to the cases above: the cache is asked to
#: hold ``tokens`` rows at distinct slots, and a zero-row cache has no slot to
#: write. These are reported as the operator's own limit.
NO_DEGENERATE: dict[str, str] = {
    "kv_cache": "the slot write names rows of the cache, so neither the token count "
    "nor the row count can be zero without the call becoming a different one",
}


def run(ctx: Any) -> None:
    if ctx.family in NO_DEGENERATE:
        ctx.note(
            f"degenerate shapes are not a call this operator accepts: {NO_DEGENERATE[ctx.family]}"
        )
        return
    tried = 0
    for label, overrides in CASES:
        shape = _shape_for(ctx.family, overrides)
        args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant, **shape)
        got = observe(ctx, args, kwargs)
        want = reference(ctx, args, kwargs, dtype=torch.float64)
        if len(got) != len(want):
            raise AssertionError(
                f"{ctx.column.name}: {label} returned {len(got)} tensors, "
                f"the reference returned {len(want)}"
            )
        for index, (produced, expected) in enumerate(zip(got, want, strict=True)):
            if produced.shape != expected.shape:
                raise AssertionError(
                    f"{ctx.column.name}: {label} output {index} has shape "
                    f"{tuple(produced.shape)}, the reference has {tuple(expected.shape)}"
                )
            if produced.numel() == 0:
                continue
            close(
                (produced,),
                (expected,),
                rel=budget(ctx),
                msg=f"{ctx.column.name}: {label}",
            )
            tried += 1
    ctx.note(f"degenerate shapes: {len(CASES)} forms, {tried} numeric comparisons")


def _shape_for(family: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """The subset of ``overrides`` that names a dimension ``family`` actually has.

    ``make_inputs`` takes the whole vocabulary of dimensions so that a script can
    reshape one; passing ``heads`` to ``softmax`` would be accepted and ignored,
    which would silently make two cases identical. Filtering by the dimension the
    family reads keeps the labels honest.
    """
    usable: dict[str, tuple[str, ...]] = {
        "add": ("rows", "width"),
        "flash_attention": ("batch", "heads", "seq", "width"),
        "softmax": ("rows", "width"),
        "rope": ("batch", "heads", "seq", "width"),
        "rms_norm": ("rows", "width"),
        "kv_cache": ("tokens", "rows", "width"),
    }
    allowed = usable.get(family, ())
    shape = {key: value for key, value in overrides.items() if key in allowed}
    floor = WIDTH_FLOOR.get(family)
    if floor is not None and "width" in shape:
        shape["width"] = max(shape["width"], floor)
    return shape
