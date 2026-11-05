"""A11 -- does quantisation error grow over steps, or blow up?

A single-step comparison can pass while a training loop diverges, because each
step multiplies the previous error by the weight matrix. Three steps of an
honest update, quantised versus float, is the smallest experiment that tells
the two apart.
"""

from __future__ import annotations

import torch

from .. import quant
from ..context import RunContext

ID = "A11"
OPS = None

_STEPS = 3
_RATE = 0.05


def run(ctx: RunContext) -> None:
    dtype = ctx.fp8_dtype
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(32, 16, generator=generator)
    target = torch.randn(32, 16, generator=generator)
    start = torch.randn(16, 16, generator=generator) * (1.0 / 16**0.5)

    def train(quantised: bool) -> list[float]:
        weight = start.clone().requires_grad_(True)
        losses: list[float] = []
        for _ in range(_STEPS):
            used = (
                quant.roundtrip(weight, quant.dynamic_scale(weight.detach(), dtype), dtype)
                if quantised
                else weight
            )
            out = x @ used
            loss = torch.nn.functional.mse_loss(out, target)
            losses.append(float(loss.detach()))
            (grad,) = torch.autograd.grad(loss, weight)
            weight = (weight - _RATE * grad).detach().requires_grad_(True)
        return losses

    plain = train(False)
    quantised_losses = train(True)

    if not all(value == value and abs(value) < 1e6 for value in quantised_losses):
        raise AssertionError(
            f"{dtype}: quantised training produced a non-finite loss: {quantised_losses}"
        )

    growth = quantised_losses[-1] / max(plain[-1], 1e-12)
    if not growth <= 1.5:
        raise AssertionError(
            f"{dtype}: after {_STEPS} steps the quantised loss is {growth:.2f}x the float loss "
            f"({quantised_losses[-1]:.4e} vs {plain[-1]:.4e})"
        )
    ctx.note(
        f"loss after {_STEPS} steps: float {plain[-1]:.4e}, quantised {quantised_losses[-1]:.4e}"
    )
