"""B6 -- broadcasting: the weight over a batch, the groups over the rows."""

from __future__ import annotations

import torch

from ..context import RunContext

ID = "B6"
OPS = None


def run(ctx: RunContext) -> None:
    name = ctx.op.name

    if name == "layer_norm":
        width = 16
        generator = torch.Generator().manual_seed(ctx.seed)
        x = torch.randn(2, 3, width, generator=generator)
        weight = torch.randn(width, generator=generator) * 0.1 + 1.0
        bias = torch.randn(width, generator=generator) * 0.1
        out = torch.nn.functional.layer_norm(x, (width,), weight, bias, 1e-5)
        manual = torch.stack(
            [
                torch.nn.functional.layer_norm(row, (width,), weight, bias, 1e-5)
                for row in x.reshape(-1, width)
            ]
        ).reshape(x.shape)
        if not torch.allclose(out, manual, atol=1e-6, rtol=0):
            raise AssertionError(
                "layer_norm: the weight does not broadcast over the leading dimensions"
            )
        return

    if name == "grouped_gemm":
        a = torch.randn(6, 4)
        b = torch.randn(2, 4, 3)
        offsets = torch.tensor([0, 2, 6], dtype=torch.int32)
        out = ctx.op.forward(a, b, offsets)
        first = a[:2] @ b[0]
        second = a[2:] @ b[1]
        if not torch.allclose(out, torch.cat([first, second]), atol=1e-6, rtol=0):
            raise AssertionError("grouped_gemm: the groups do not partition the rows as declared")
        return

    ctx.na(f"{name} is unary; there is no second operand to broadcast")
