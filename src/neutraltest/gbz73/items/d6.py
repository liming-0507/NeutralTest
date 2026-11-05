"""D6 -- illegal input must fail loudly.

A wrong shape, a wrong length and an out-of-range axis are all caller mistakes.
The requirement is not a particular exception type but a clean failure: no
garbage result, no crash, no silent clamp.
"""

from __future__ import annotations

import torch

from ..context import RunContext
from ._support import adapt

ID = "D6"
OPS = None

_EXPECTED = (RuntimeError, ValueError, IndexError, TypeError)


def _must_raise(ctx: RunContext, label: str, fn) -> None:
    try:
        fn()
    except _EXPECTED:
        return
    except AssertionError:
        raise
    else:
        raise AssertionError(f"{ctx.op.name}: {label} was accepted")


def run(ctx: RunContext) -> None:
    name = ctx.op.name
    generator = torch.Generator().manual_seed(ctx.seed)

    if name == "grouped_gemm":
        a = torch.randn(4, 4, generator=generator)
        b = torch.randn(2, 4, 3, generator=generator)
        _must_raise(
            ctx,
            "a decreasing group_list",
            lambda: ctx.op.forward(a, b, torch.tensor([0, 3, 2], dtype=torch.int32)),
        )
        _must_raise(
            ctx,
            "an out-of-range offset",
            lambda: ctx.op.forward(a, b, torch.tensor([0, 2, 99], dtype=torch.int32)),
        )
        _must_raise(
            ctx,
            "a mismatched right-hand side",
            lambda: ctx.op.forward(
                a, torch.randn(2, 5, 3), torch.tensor([0, 2, 4], dtype=torch.int32)
            ),
        )
        return

    if name == "layer_norm":
        x = torch.randn(4, 8, generator=generator)
        _must_raise(
            ctx,
            "a weight of the wrong length",
            lambda: torch.nn.functional.layer_norm(x, (8,), torch.ones(7), torch.zeros(7)),
        )
        _must_raise(
            ctx,
            "a normalized shape that does not divide the input",
            lambda: torch.nn.functional.layer_norm(x, (3,)),
        )
        return

    if name == "gelu":
        x = torch.randn(4, generator=generator)
        _must_raise(
            ctx,
            "an unknown approximation",
            lambda: torch.nn.functional.gelu(x, approximate="bogus"),
        )
        return

    if name == "amax":
        x = torch.randn(4, 8, generator=generator)
        _must_raise(ctx, "an out-of-range dimension", lambda: torch.amax(x.abs(), dim=5))
        return

    if name == "relu":
        # relu accepts anything numeric; a string tensor is the clean way to
        # show that a non-numeric input is refused.
        _must_raise(ctx, "a non-numeric input", lambda: torch.relu(torch.tensor(["a", "b"])))
        return

    args, kwargs = adapt(name, torch.randn(4, 8, generator=generator))
    ctx.na(f"no illegal-input case defined for {name}")
