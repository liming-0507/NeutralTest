"""A12 -- are the gradients right, not merely finite?

Two independent checks: the float32 gradients must match the float64 golden
ones, and both must match a central finite difference of the same loss. The
second one would catch a gradient that is consistently wrong in the same way in
both precisions.
"""

from __future__ import annotations

import torch

from .. import metrics
from ..context import RunContext

ID = "A12"
OPS = None

_FD_STEPS = 2
_FD_EPS = 1e-6


def run(ctx: RunContext) -> None:
    narrow = ctx.gradients(torch.float32)
    wide = ctx.gradients(torch.float64)

    for index, (small, big) in enumerate(zip(narrow, wide, strict=True)):
        if small is None or big is None:
            raise AssertionError(f"{ctx.op.name}: input {index} received no gradient")
        diff = ctx.diff(small.double(), big)
        metrics.assert_within(
            diff, rel=1e-4, msg=f"{ctx.op.name}: float32 gradient {index} vs float64"
        )
        if not torch.isfinite(big).all():
            raise AssertionError(f"{ctx.op.name}: float64 gradient {index} is not finite")

    # Finite difference on the first floating-point input, in float64.
    args, kwargs = ctx.inputs(torch.float64)
    floats = [
        i for i, a in enumerate(args) if isinstance(a, torch.Tensor) and a.is_floating_point()
    ]
    if not floats:
        ctx.na("no floating-point input to differentiate")
    position = floats[0]
    base = args[position].detach().clone().requires_grad_(True)
    probe = list(args)
    probe[position] = base
    out = ctx.op.forward(*probe, **kwargs)
    # A weighted sum, so that the gradient is not the constant vector an
    # unweighted reduction would produce.
    weight = torch.randn(out.shape, generator=torch.Generator().manual_seed(ctx.seed))

    def loss_of(tensor: torch.Tensor) -> float:
        rebuilt = list(args)
        rebuilt[position] = tensor
        return float((ctx.op.forward(*rebuilt, **kwargs) * weight).sum())

    (analytic,) = torch.autograd.grad((out * weight).sum(), base)

    flat = torch.nonzero(weight.reshape(-1) != 0).reshape(-1)
    if flat.numel() == 0:
        ctx.na("the weighting was empty")
    picks = [int(flat[i]) for i in range(min(_FD_STEPS, int(flat.numel())))]

    for index in picks:
        plus = args[position].detach().clone().reshape(-1)
        minus = plus.clone()
        plus[index] += _FD_EPS
        minus[index] -= _FD_EPS
        numeric = (
            loss_of(plus.reshape(args[position].shape))
            - loss_of(minus.reshape(args[position].shape))
        ) / (2 * _FD_EPS)
        analytic_value = float(analytic.reshape(-1)[index])
        if not abs(numeric - analytic_value) <= 1e-3 * max(1.0, abs(numeric)):
            raise AssertionError(
                f"{ctx.op.name}: gradient entry {index} is {analytic_value:.6f}, finite difference says {numeric:.6f}"
            )
    ctx.note(f"finite difference agreed on {len(picks)} entries")
