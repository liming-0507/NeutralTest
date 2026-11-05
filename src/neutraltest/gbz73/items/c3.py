"""C3 -- how fine the scale is: per tensor, per row, per column, per group.

Finer granularity cannot make the result worse, so the check is two-sided: each
arrangement stays inside the floor, and the per-row arrangement is not worse
than the per-tensor one.
"""

from __future__ import annotations

import torch

from .. import metrics, quant
from ..context import RunContext
from ._support import adapt

ID = "C3"
OPS = None

_GROUP = 4


def _scales(x: torch.Tensor, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    top = quant.max_finite(dtype)
    flat = x.reshape(-1)
    per_tensor = (flat.abs().max() / top).clamp_min(1e-12)
    rows = (x.abs().amax(dim=-1, keepdim=True) / top).clamp_min(1e-12)
    columns = (x.abs().amax(dim=0, keepdim=True) / top).clamp_min(1e-12)
    padded = -(-x.shape[0] // _GROUP) * _GROUP
    groups = torch.nn.functional.pad(x.abs(), (0, 0, 0, padded - x.shape[0])).reshape(
        -1, _GROUP, x.shape[-1]
    )
    per_group = (groups.amax(dim=(1, 2)) / top).clamp_min(1e-12)
    return {
        "per-tensor": per_tensor.reshape(1, 1),
        "per-row": rows,
        "per-column": columns,
        "per-group": per_group.reshape(-1, 1).repeat_interleave(_GROUP, dim=0)[: x.shape[0]],
    }


def run(ctx: RunContext) -> None:
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(64, 32, generator=generator) * 1.5
    args64, kwargs64 = adapt(ctx.op.name, x.double())
    golden = ctx.op.forward(*args64, **kwargs64)

    diffs = {}
    for label, scale in _scales(x, ctx.fp8_dtype).items():
        quantised = quant.roundtrip(x, scale, ctx.fp8_dtype)
        args, kwargs = adapt(ctx.op.name, quantised)
        out = ctx.op.forward(*args, **kwargs)
        diff = ctx.diff(out.double(), golden)
        metrics.assert_within(diff, rel=ctx.floor(), msg=f"{ctx.op.name}: {label} scale")
        diffs[label] = diff.rel

    if not diffs["per-row"] <= diffs["per-tensor"] * 1.25 + 1e-9:
        raise AssertionError(
            f"{ctx.op.name}: per-row scales ({diffs['per-row']:.3e}) are worse than per-tensor ({diffs['per-tensor']:.3e})"
        )
