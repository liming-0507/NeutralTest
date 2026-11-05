"""E3 -- concurrency and streams.

Two threads sharing one device is the normal serving case. There is no stream
API on a CPU-only build, so the check here is the part that matters everywhere:
concurrent calls must not share hidden state.
"""

from __future__ import annotations

import concurrent.futures

import torch

from ..context import RunContext

ID = "E3"
OPS = None


def run(ctx: RunContext) -> None:
    if not ctx.backend.has_streams():
        ctx.na(
            f"{ctx.backend.name} exposes no stream API, so multi-stream behaviour cannot be exercised"
        )
        return
    args, kwargs = ctx.inputs(torch.float32)
    reference = ctx.run(args, kwargs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: ctx.run(args, kwargs), range(4)))
    for result in results:
        if not torch.equal(result, reference):
            raise AssertionError(f"{ctx.op.name}: a concurrent call disagreed with the serial one")
