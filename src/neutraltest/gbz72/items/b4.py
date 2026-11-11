"""B4 -- irregular shapes.

Check: every column is exercised on sizes that are not powers of two and not
multiples of the usual vector width, which is where a kernel that rounds a length
up, assumes divisibility, or writes a tail with the wrong mask goes wrong. The
shapes are per family and include a prime width, a one-row tensor, and a 977-wide
row.

Tolerance: the column's float32 budget. The shapes change what is computed, not
how precisely, so the same budget applies as for the regular shape.
"""

from __future__ import annotations

from typing import Any

from .._support import check, make_inputs

ID = "B4"
COLUMNS = None

SHAPES: dict[str, tuple[dict[str, Any], ...]] = {
    "add": ({"rows": 3, "width": 37}, {"rows": 7, "width": 13}, {"rows": 1, "width": 977}),
    "flash_attention": (
        {"batch": 1, "heads": 1, "seq": 5, "width": 7},
        {"batch": 2, "heads": 3, "seq": 13, "width": 17},
        {"batch": 1, "heads": 2, "seq": 1, "width": 3},
    ),
    "kv_cache": (
        {"rows": 1, "width": 8, "tokens": 3},
        {"rows": 37, "width": 13, "tokens": 5},
        {"rows": 977, "width": 16, "tokens": 7},
    ),
    "softmax": ({"rows": 3, "width": 37}, {"rows": 7, "width": 13}, {"rows": 1, "width": 977}),
    "rope": (
        {"batch": 1, "seq": 5, "heads": 1, "width": 6},
        {"batch": 2, "seq": 3, "heads": 3, "width": 10},
        {"batch": 1, "seq": 1, "heads": 1, "width": 16},
    ),
    "rms_norm": ({"rows": 3, "width": 37}, {"rows": 7, "width": 13}, {"rows": 1, "width": 977}),
}


def run(ctx: Any) -> None:
    for shape in SHAPES[ctx.family]:
        args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant, **shape)
        check(ctx, args, kwargs)
