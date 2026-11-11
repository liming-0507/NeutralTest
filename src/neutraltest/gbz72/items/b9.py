"""B9 -- the input distribution does not change the answer.

Check: each forward column is fed the same shape five times from different
distributions -- normal, uniform, constant, heavy-tailed, and a scaled copy of the
normal draw -- and each result is compared with the reference on the same values.
The distributions are the ones that separate a genuine computation from one that
was tuned on the default draw: a all-ones tensor exposes a division by the wrong
statistic, a heavy tail exposes an intermediate that overflows before the final
scale, and a scaled draw exposes a threshold that was written for unit variance.
The rotation keeps its positions, because those are indices rather than data.

Tolerance: the column's float32 budget for every distribution. The comparison is
normalised by the magnitude of the reference, so a scaled draw does not need a
wider budget; only a different shape would.
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import budget, clone_args, close, make_inputs, observe, reference

ID = "B9"
COLUMNS = None

DISTRIBUTIONS = ("normal", "uniform", "constant", "heavy", "scaled")


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    for name in DISTRIBUTIONS:
        candidate = list(clone_args(args))
        _draw(ctx, candidate, name)
        frozen = tuple(candidate)
        close(
            observe(ctx, frozen, kwargs),
            reference(ctx, frozen, kwargs, dtype=torch.float32),
            rel=budget(ctx),
            msg=f"{ctx.column.name}: {name} input",
        )


def _draw(ctx: Any, args: Any, name: str) -> None:
    positions = 1 if ctx.family == "rope" else min(2, len(args))
    for index in range(positions):
        value = args[index]
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            continue
        generator = torch.Generator().manual_seed(ctx.seed + index)
        args[index] = _transform(value, name, generator, ctx.family)


def _transform(
    value: torch.Tensor, name: str, generator: torch.Generator, family: str
) -> torch.Tensor:
    if name == "normal":
        return value
    if name == "uniform":
        return torch.rand(value.shape, generator=generator) * 2.0 - 1.0
    if name == "constant":
        return torch.ones_like(value)
    if name == "heavy":
        return value.sign() * value.abs().pow(3) * 10.0
    scale = 1e2 if family == "flash_attention" else 1e3
    return value * scale
