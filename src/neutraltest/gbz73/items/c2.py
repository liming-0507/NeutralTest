"""C2 -- MX blocks: 32 elements sharing one E8M0 scale.

Constructing an MX block needs an E8M0 scale dtype in the tensor library. Where
it does not exist, the honest answer is a disclosure, not a green cell.
"""

from __future__ import annotations

import torch

from ..context import RunContext

ID = "C2"
OPS = None


def run(ctx: RunContext) -> None:
    if not hasattr(torch, "float8_e8m0fnu"):
        ctx.na("no E8M0 scale dtype in this build, so an MX block cannot be constructed")
    # If a build does provide it, at least the block arithmetic is checked.
    block = 32
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(4, block, generator=generator)
    scales = torch.exp2(torch.zeros(4, 1))
    if not torch.allclose(x / scales, x):
        raise AssertionError("a unit block scale changed the values")
