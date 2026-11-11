"""B5 -- extreme sizes.

Check: the smallest tensor each family can be given (one element, one row, one
token) and the largest one this machine can hold without swapping, with a
non-power-of-two width in between. The boundary cases are where an empty tail, a
zero-length loop, or an unscaled first sample shows up; the large case is where an
index that is narrower than the tensor, or a stride that overflows, shows up.

Tolerance: the column's float32 budget. A narrow row reduces the number of
accumulated terms, so it cannot be looser than the regular shape; the large case
accumulates across a longer axis only where the family reduces over it, and the
budget's reduction term already accounts for that.
"""

from __future__ import annotations

from typing import Any

from .._support import check, make_inputs

ID = "B5"
COLUMNS = None

SHAPES: dict[str, tuple[dict[str, Any], ...]] = {
    "add": (
        {"rows": 1, "width": 1},
        {"rows": 1, "width": 16384},
        {"rows": 4096, "width": 3},
        {"rows": 64, "width": 1024},
    ),
    "softmax": (
        {"rows": 1, "width": 1},
        {"rows": 1, "width": 16384},
        {"rows": 4096, "width": 3},
        {"rows": 64, "width": 1024},
    ),
    "rms_norm": (
        {"rows": 1, "width": 1},
        {"rows": 1, "width": 16384},
        {"rows": 4096, "width": 3},
        {"rows": 64, "width": 1024},
    ),
    "flash_attention": (
        {"batch": 1, "heads": 1, "seq": 1, "width": 1},
        {"batch": 1, "heads": 2, "seq": 64, "width": 64},
        {"batch": 4, "heads": 1, "seq": 33, "width": 5},
    ),
    "kv_cache": (
        {"rows": 1, "width": 1, "tokens": 1},
        {"rows": 16384, "width": 8, "tokens": 3},
        {"rows": 17, "width": 1024, "tokens": 2},
    ),
    "rope": (
        {"batch": 1, "seq": 1, "heads": 1, "width": 2},
        {"batch": 1, "seq": 1, "heads": 1, "width": 4096},
        {"batch": 8, "seq": 17, "heads": 2, "width": 64},
    ),
}


def run(ctx: Any) -> None:
    for shape in SHAPES[ctx.family]:
        args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant, **shape)
        check(ctx, args, kwargs)
