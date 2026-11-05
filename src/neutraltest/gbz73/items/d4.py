"""D4 -- the variable-length structures: an empty group, a one-row group, a tail.

Only the grouped operator has a structure whose shape depends on data; the other
columns say so instead of inventing a check.
"""

from __future__ import annotations

import torch

from ..context import RunContext

ID = "D4"
OPS = None


def run(ctx: RunContext) -> None:
    if ctx.op.name != "grouped_gemm":
        ctx.na("the operator has a fixed shape; there is no variable-length structure to check")

    generator = torch.Generator().manual_seed(ctx.seed)
    cases = {
        "empty first group": ([0, 0, 4], 4),
        "single-row groups": ([0, 1, 2], 2),
        "tail group of one": ([0, 4, 5], 5),
        "one group only": ([0, 6], 6),
    }
    width = 4
    for label, (offsets, rows) in cases.items():
        a = torch.randn(rows, width, generator=generator)
        b = torch.randn(len(offsets) - 1, width, 3, generator=generator)
        out = ctx.op.forward(a, b, torch.tensor(offsets, dtype=torch.int32))
        expected = torch.cat(
            [a[offsets[g] : offsets[g + 1]] @ b[g] for g in range(len(offsets) - 1)]
        )
        if not torch.allclose(out, expected, atol=1e-6, rtol=0):
            raise AssertionError(f"grouped_gemm: {label} does not match the per-group computation")
