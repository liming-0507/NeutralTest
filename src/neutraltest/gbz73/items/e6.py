"""E6 -- the same work on a second device.

Reported per backend: with one device there is nothing to shard, and pretending
otherwise would hide exactly the case the clause cares about -- so one device is
declared not applicable rather than passed.

With two or more devices the row does the work the clause is about: the same
inputs go to device 0 and device 1 and the two results have to agree. Merely
asking whether a second device exists would pass on a build where device 1 is
advertised and then computes the wrong thing, which is the deployment fault this
row is here to find.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..context import RunContext

ID = "E6"
OPS = None


def _on(ctx: RunContext, index: int, args: tuple, kwargs: dict) -> torch.Tensor:
    """Run ``ctx.op`` on device ``index`` and bring the result back."""
    if index != ctx.device_index:
        shard = replace(ctx, device_index=index)
        return shard.run(args, kwargs)
    return ctx.run(args, kwargs)


def run(ctx: RunContext) -> None:
    count = ctx.backend.device_count()
    if count < 2:
        ctx.na(
            f"{ctx.backend.name} exposes {count} device(s); multi-device behaviour cannot be observed"
        )
        return
    args, kwargs = ctx.inputs(torch.float32)
    first = _on(ctx, 0, args, kwargs)
    second = _on(ctx, 1, args, kwargs)
    if first.shape != second.shape:
        raise AssertionError(
            f"{ctx.op.name}: device 0 returned {tuple(first.shape)}, device 1 returned {tuple(second.shape)}"
        )
    if not torch.equal(first, second):
        worst = (first - second).abs().max().item()
        raise AssertionError(
            f"{ctx.op.name}: device 1 differs from device 0 by up to {worst:.3e} on the same inputs"
        )
    ctx.note(f"{count} devices returned identical results")
