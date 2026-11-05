"""F5 -- the measurement protocol itself.

Every timing number in this suite comes from the same protocol; this row checks
the protocol rather than the operator: warm-up removes the cold path, one
measurement is not an answer, and two independent batches agree.
"""

from __future__ import annotations

from ..context import RunContext
from ._support import device_call, measure

ID = "F5"
OPS = None

_WARMUP = 3
_REPEAT = 9


def run(ctx: RunContext) -> None:
    call = device_call(ctx)

    cold = measure(call, warmup=0, repeat=1)
    warm = measure(call, warmup=_WARMUP, repeat=_REPEAT)
    if cold <= 0 or warm <= 0:
        raise AssertionError(
            f"{ctx.op.name}: a timing came back as zero (cold {cold:.3e}, warm {warm:.3e})"
        )

    first = measure(call, warmup=_WARMUP, repeat=_REPEAT)
    second = measure(call, warmup=_WARMUP, repeat=_REPEAT)
    spread = max(first, second) / max(min(first, second), 1e-12)
    if spread > 3.0:
        raise AssertionError(
            f"{ctx.op.name}: the protocol is not reproducible ({spread:.1f}x between batches)"
        )
    ctx.note(f"cold {cold * 1e6:.1f} us, warm {warm * 1e6:.1f} us, spread {spread:.2f}x")
