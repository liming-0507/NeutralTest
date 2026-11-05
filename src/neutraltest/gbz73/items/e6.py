"""E6 -- several devices in one process.

Reported per backend: with one device there is nothing to shard, and pretending
otherwise would hide exactly the case the clause cares about.
"""

from __future__ import annotations

import torch

from ..context import RunContext

ID = "E6"
OPS = None


def run(ctx: RunContext) -> None:
    count = ctx.backend.device_count()
    if count < 2:
        ctx.na(
            f"{ctx.backend.name} exposes {count} device(s); multi-device behaviour cannot be observed"
        )
        return
    ctx.note(f"{count} devices see the same inputs")
    args, kwargs = ctx.inputs(torch.float32)
    first = ctx.run(args, kwargs)
    other = ctx.backend.make_device(1)
    if other.type == first.device.type and other.index == first.device.index:
        raise AssertionError(f"{ctx.backend.name}: device 1 is the same device as device 0")
