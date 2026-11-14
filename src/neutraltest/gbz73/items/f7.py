"""F7 -- does the accelerator beat the CPU at all.

Every other performance row in this catalogue measures one implementation against
itself, so an accelerator that never reaches the CPU's time passes all of them.
An accelerator slower than the CPU on the operator's own shape is the finding, not a
footnote: the comparison is the same call, the same inputs and the same protocol on
both devices, and the ratio is reported whatever the verdict.

A ratio closer than `SAME_ORDER` is reported as such rather than claimed: two
devices within 1.5x of each other on a hosted machine are the same order.

A device call has a floor the CPU does not: getting the work to the accelerator
and back costs tens of microseconds before the operator does anything. Comparing
a 2 us CPU operator with the device's call cost therefore says nothing about the
operator -- the ratio is a launch latency. The verdict is only claimed where the
device spent longer than `LAUNCH_FLOOR`; a faster result is reported as such, and
a result inside the floor is reported with its numbers and left unclaimed. The
floor is far below any kernel this suite would call a win, so nothing passes by
being merely slow.
"""

from __future__ import annotations

from ..context import RunContext
from ._support import NOISE_SLACK, device_call, measure, require_resolvable

ID = "F7"
OPS = None

#: Timing this close is reported as "same order as the CPU", not as a win or a loss.
SAME_ORDER = 1.5

#: Only said for operators whose result nobody reads: pushing the inputs and
#: recording a launch is not the work, so the ratio is a lower-bound-shaped number.
#: Below this the device side is measuring launch and not the operator.
LAUNCH_FLOOR = 100e-6

NULL_ASYNC_HINT = (
    " (a null-ish operator: neither side has to bring the result back, so the device "
    "side may still be executing when the clock stops)"
)


def run(ctx: RunContext) -> None:
    if ctx.device.type == "cpu":
        ctx.na("the CPU is the reference here; there is no second device to compare with")

    from ...backends.builtin import CpuBackend

    device = require_resolvable(measure(device_call(ctx)), f"{ctx.device.type} {ctx.op.name}")
    cpu = require_resolvable(measure(device_call(ctx, backend=CpuBackend())), "cpu")

    ratio = device / cpu
    async_hint = "" if ctx.op.communicated else NULL_ASYNC_HINT
    ctx.note(
        f"{ctx.op.name}/{ctx.variant}: {device * 1e6:.1f} us on {ctx.device.type} vs "
        f"{cpu * 1e6:.1f} us on cpu ({ratio:.2f}x){async_hint}"
    )
    if ratio <= SAME_ORDER * NOISE_SLACK:
        return
    if device < LAUNCH_FLOOR and not ctx.op.communicated:
        ctx.note(
            f"{ctx.op.name}: the device call takes {device * 1e6:.1f} us, inside the "
            f"{LAUNCH_FLOOR * 1e6:.0f} us launch floor; the {ratio:.2f}x is call overhead, "
            f"not the operator"
        )
        return
    raise AssertionError(
        f"{ctx.op.name}: the accelerator takes {ratio:.2f}x the CPU's time on the same "
        f"call ({device * 1e6:.1f} us vs {cpu * 1e6:.1f} us) -- slower than the CPU is "
        f"the finding, not a footnote{async_hint}"
    )
