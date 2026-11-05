"""A1 -- every result is compared against the float32 and the float64 baseline."""

from __future__ import annotations

from .. import metrics
from ..context import RunContext

ID = "A1"
OPS = None


def run(ctx: RunContext) -> None:
    normal = ctx.normal()
    golden = ctx.golden()

    baseline = ctx.diff(normal, golden)
    metrics.assert_within(
        baseline, rel=1e-4, msg=f"{ctx.op.name}: float32 baseline vs float64 golden"
    )
    if not baseline.cosine > 0.9999:
        raise AssertionError(
            f"{ctx.op.name}: the two baselines disagree (cosine {baseline.cosine:.6f})"
        )

    quantised = ctx.quantised()
    ctx.check(quantised, golden, msg=f"{ctx.op.name}: quantised path vs golden")
    ctx.check(quantised, normal, msg=f"{ctx.op.name}: quantised path vs ordinary baseline")
