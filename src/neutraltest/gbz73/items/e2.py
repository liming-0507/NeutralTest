"""E2 -- seeds and distributions.

A seed that does not control the data is worse than no seed: the run looks
reproducible and is not. Different seeds must give different data, the same
seed must give the same data, and both must satisfy the operator.
"""

from __future__ import annotations

import torch

from ...seeding import make_generator, randn
from ..context import RunContext

ID = "E2"
OPS = None


def run(ctx: RunContext) -> None:
    shape = (64, 16)
    default = randn(shape, torch.float32, make_generator(ctx.seed))
    repeated = randn(shape, torch.float32, make_generator(ctx.seed))
    if not torch.equal(default, repeated):
        raise AssertionError(f"{ctx.op.name}: the same seed produced different inputs")

    other_seed = randn(shape, torch.float32, make_generator(ctx.seed + 1))
    if torch.equal(default, other_seed):
        raise AssertionError(
            f"{ctx.op.name}: two seeds produced identical inputs, so the seed is ignored"
        )

    if float(default.std()) == 0.0:
        raise AssertionError(f"{ctx.op.name}: the generated input has no spread")
    if not torch.isfinite(other_seed).all():
        raise AssertionError(f"{ctx.op.name}: the generated input contains non-finite values")

    # The operator must survive data drawn from a different seed too.
    args, kwargs = ctx.inputs(torch.float32)
    ctx.run(args, kwargs)
