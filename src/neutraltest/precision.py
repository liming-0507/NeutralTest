"""Pin the float32 arithmetic contract before measuring anything.

``torch.set_float32_matmul_precision`` is a public switch: at ``high`` and
``medium`` a float32 matmul is allowed to round its inputs to a narrower
format (TF32-class), which is a 10-bit mantissa, not 24.  A container image is
free to leave that switch at ``high``, and then a float32 operator looks two to
three orders of magnitude less accurate than its dtype promises.

That is a property of the session, not of the operator, so the suite pins it
once, records the value it pinned, and reports it beside the results.  Pin it
yourself with ``--fp32-precision`` when you deliberately want to measure the
reduced-precision path.
"""

from __future__ import annotations

import torch

LEVELS = ("highest", "high", "medium")
DEFAULT = "highest"


def pin(level: str = DEFAULT) -> str:
    """Apply ``level`` and return what torch actually reports afterwards."""
    if level not in LEVELS:
        raise ValueError(f"unknown float32 precision {level!r}; choose from {', '.join(LEVELS)}")
    torch.set_float32_matmul_precision(level)
    return describe()


def describe() -> str:
    """The live setting, e.g. ``highest (float32 matmul keeps a 24-bit mantissa)``."""
    level = torch.get_float32_matmul_precision()
    note = {
        "highest": "float32 matmul keeps a 24-bit mantissa",
        "high": "float32 matmul may use a TF32-class 10-bit mantissa",
        "medium": "float32 matmul may use a bfloat16-class 8-bit mantissa",
    }.get(level, "unknown level")
    return f"{level} ({note})"
