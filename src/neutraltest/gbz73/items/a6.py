"""A6 -- NaN and infinity must survive, or be refused, but never disappear.

A quantiser that maps NaN to zero turns a poisoned gradient into a plausible
number, which is the worst failure mode a training operator can have. Each
operator is checked for the propagation its mathematics actually implies.
"""

from __future__ import annotations

import torch

from ..context import RunContext

ID = "A6"
OPS = None


def run(ctx: RunContext) -> None:
    args, kwargs = ctx.inputs(torch.float32)
    floats = [
        i for i, a in enumerate(args) if isinstance(a, torch.Tensor) and a.is_floating_point()
    ]
    if not floats:
        ctx.na("no floating-point input to poison")

    index = floats[0]
    for poison, name in ((float("nan"), "NaN"), (float("inf"), "Inf")):
        poisoned = list(args)
        sample = poisoned[index].clone()
        sample.view(-1)[0] = poison
        poisoned[index] = sample
        try:
            out = ctx.op.forward(*poisoned, **kwargs)
        except Exception as exc:  # noqa: BLE001 - a refusal is a legitimate answer
            raise AssertionError(
                f"{ctx.op.name}: {name} in the input was refused with {type(exc).__name__}: {exc}"
            ) from exc

        if name == "NaN":
            if not torch.isnan(out).any():
                raise AssertionError(f"{ctx.op.name}: NaN in the input did not reach the output")
            continue

        if ctx.op.name == "layer_norm":
            # Normalising a set of infinities is undefined; NaN is the honest answer.
            if not torch.isnan(out).any():
                raise AssertionError(
                    "layer_norm: Inf input should produce NaN, got a finite result"
                )
            continue
        if not torch.isinf(out).any():
            raise AssertionError(f"{ctx.op.name}: Inf in the input did not reach the output")
