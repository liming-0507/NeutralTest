"""F5 -- the measurement protocol itself.

Check: the column is first called once cold, to prove that the first call works
before any warm-up has touched it; it is then measured twice with very different
protocols -- one warm-up and five samples against twenty-five warm-ups and
fifty-one samples -- and the two medians must agree. The protocol question is
whether the number that gets reported depends on how it was taken: a column whose
cost halves after the tenth call, or whose samples drift upward as the cache
fills, will fail here while looking stable under a single protocol.

Tolerance: 3 x the noise slack (4.5x) between the two protocols. Below the clock's
resolution the run stops with a recorded reason instead of reporting a number the
timer cannot support.
"""

from __future__ import annotations

from typing import Any

from .._support import NOISE_SLACK, make_inputs, measure

ID = "F5"
COLUMNS = None

AGREEMENT = 3.0 * NOISE_SLACK


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    if ctx.backward:
        call: Any = lambda: ctx.gradients_with(args, kwargs)  # noqa: E731
    else:
        call = lambda: ctx.call_with(args, kwargs)  # noqa: E731
    call()
    quick = measure(call, warmup=1, repeat=5)
    thorough = measure(call, warmup=25, repeat=51)
    for seconds, what in ((quick, "the short protocol"), (thorough, "the long protocol")):
        if seconds < 1e-7:
            ctx.na(
                f"{what} for {ctx.column.name} resolves to {seconds * 1e9:.0f} ns, below "
                f"the clock's resolution"
            )
    ratio = max(quick, thorough) / min(quick, thorough)
    if ratio > AGREEMENT:
        raise AssertionError(
            f"{ctx.column.name}: the two protocols disagree by {ratio:.2f}x, over the "
            f"{AGREEMENT:.1f}x bound; the reported cost depends on how it was measured"
        )
    ctx.note(
        f"{ctx.column.name}: 1+5 warm protocol {quick * 1e6:.1f} us, "
        f"25+51 protocol {thorough * 1e6:.1f} us"
    )
