"""D3 -- in-place forms, aliasing and views.

Two failure modes: an operator that writes into its input when it promised not
to, and a result that aliases its input so that a later write corrupts both.
"""

from __future__ import annotations

import torch

from ..context import RunContext
from ._support import adapt

ID = "D3"
OPS = None


def run(ctx: RunContext) -> None:
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(16, 32, generator=generator)

    args, kwargs = adapt(ctx.op.name, x)
    before = [t.clone() for t in args if isinstance(t, torch.Tensor)]
    out = ctx.op.forward(*args, **kwargs)
    after = [t for t in args if isinstance(t, torch.Tensor)]
    for original, current in zip(before, after, strict=True):
        if not torch.equal(original, current):
            raise AssertionError(f"{ctx.op.name}: the call modified its input")

    if out.data_ptr() == x.data_ptr() and out.shape == x.shape:
        raise AssertionError(f"{ctx.op.name}: the result aliases the input")

    # A view of the same storage must give the same answer.
    y = torch.randn(4, 8, generator=generator)
    view = y.view(2, 16)
    args_v, kwargs_v = adapt(ctx.op.name, view)
    from_view = ctx.op.forward(*args_v, **kwargs_v)
    args_c, kwargs_c = adapt(ctx.op.name, view.contiguous())
    from_copy = ctx.op.forward(*args_c, **kwargs_c)
    if not torch.equal(from_view, from_copy):
        raise AssertionError(f"{ctx.op.name}: a view gave a different result than a copy")

    if ctx.op.name == "relu":
        target = x.clone()
        expected = torch.relu(x)
        returned = torch.relu_(target)
        if returned.data_ptr() != target.data_ptr():
            raise AssertionError("relu_: the in-place form returned a new tensor")
        if not torch.equal(target, expected):
            raise AssertionError("relu_: the in-place form disagrees with the out-of-place form")
