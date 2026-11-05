"""B1 -- tensors larger than a 32-bit element count.

Allocating 2^31 elements of float32 needs 8 GiB, which no CI machine should
have to spare, so the size is exercised where it is actually decided: in the
shape arithmetic. A meta tensor carries the shape and the strides and no
storage, which is exactly enough to prove that the operator's index arithmetic
is 64-bit.
"""

from __future__ import annotations

import torch

from ..context import RunContext

ID = "B1"
OPS = None

#: Larger than int32 can index: 2^31 rows of 8 columns.
_ROWS = 2**31
_WIDTH = 8


def run(ctx: RunContext) -> None:
    meta = torch.device("meta")
    x = torch.empty(_ROWS, _WIDTH, device=meta)
    if x.numel() <= 2**31:
        raise AssertionError(f"the probe is not large enough: {x.numel()} elements")

    name = ctx.op.name
    expected: tuple[int, ...]
    if name in {"relu", "gelu"}:
        out = torch.relu(x) if name == "relu" else torch.nn.functional.gelu(x)
        expected = (x.shape[0], x.shape[1])
    elif name == "amax":
        out = torch.amax(x.abs(), dim=-1)
        expected = (x.shape[0],)
    elif name == "layer_norm":
        weight = torch.empty(_WIDTH, device=meta)
        bias = torch.empty(_WIDTH, device=meta)
        out = torch.nn.functional.layer_norm(x, (_WIDTH,), weight, bias, 1e-5)
        expected = (x.shape[0], x.shape[1])
    elif name == "grouped_gemm":
        rows = 1024
        a = torch.empty(rows, _WIDTH, device=meta)
        b = torch.empty(2, _WIDTH, _WIDTH, device=meta)
        out = a @ b[0]
        expected = (rows, _WIDTH)
    else:  # pragma: no cover - the catalogue has five columns
        ctx.na(f"no large-tensor probe for {name}")

    if tuple(out.shape) != expected:
        raise AssertionError(
            f"{ctx.op.name}: {x.numel()} elements gave shape {tuple(out.shape)}, expected {expected}"
        )
    if out.device.type != "meta":
        raise AssertionError(f"{ctx.op.name}: the probe left the meta device")
    ctx.note(f"shape arithmetic verified at {x.numel()} elements (>{2**31})")
