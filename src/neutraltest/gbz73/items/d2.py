"""D2 -- parameter sweeps, each checked against the formula rather than against
a previous call of the same function."""

from __future__ import annotations

import torch

from ..context import RunContext

ID = "D2"
OPS = None


def run(ctx: RunContext) -> None:
    name = ctx.op.name
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(8, 16, generator=generator)

    if name == "layer_norm":
        weight = torch.randn(16, generator=generator) * 0.1 + 1.0
        bias = torch.randn(16, generator=generator) * 0.1
        for eps in (0.0, 1e-12, 1e-5, 1e-2, 1.0):
            out = torch.nn.functional.layer_norm(x, (16,), weight, bias, eps)
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            manual = (x - mean) / torch.sqrt(var + eps) * weight + bias
            if not torch.allclose(out, manual, atol=1e-6, rtol=0):
                raise AssertionError(f"layer_norm: eps={eps} does not follow the formula")
        return

    if name == "gelu":
        for approximate in ("none", "tanh"):
            out = torch.nn.functional.gelu(x, approximate=approximate)
            if approximate == "none":
                manual = 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
            else:
                inner = torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)
                manual = 0.5 * x * (1.0 + torch.tanh(inner))
            if not torch.allclose(out, manual, atol=1e-6, rtol=0):
                raise AssertionError(f"gelu: approximate={approximate} does not follow the formula")
        return

    if name == "amax":
        for dim in (0, 1, -1, -2):
            out = torch.amax(x.abs(), dim=dim)
            moved = x.abs().movedim(dim % x.dim(), 0)
            manual = moved.amax(dim=0)
            if out.shape != manual.shape or not torch.equal(out, manual):
                raise AssertionError(f"amax: dim={dim} does not agree with a manual reduction")
        return

    if name == "relu":
        for threshold in (0.0, -1.0, 0.5):
            out = torch.nn.functional.relu(x - threshold) + threshold
            if not bool((out >= threshold - 1e-7).all()):
                raise AssertionError(
                    f"relu: shifted by {threshold} let a value below the threshold through"
                )
        return

    if name == "grouped_gemm":
        a = torch.randn(6, 4, generator=generator)
        b = torch.randn(3, 4, 3, generator=generator)
        offsets = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
        out = ctx.op.forward(a, b, offsets)
        if out.shape != (6, 3):
            raise AssertionError(f"grouped_gemm: group sizes gave shape {tuple(out.shape)}")
        return

    ctx.na(f"no parameter sweep for {name}")
