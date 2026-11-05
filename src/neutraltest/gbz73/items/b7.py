"""B7 -- strided and transposed inputs.

A non-contiguous tensor has the same values and a different memory layout. The
result must be identical: if it is not, the kernel is reading the buffer
instead of the logical tensor.
"""

from __future__ import annotations

import torch

from ...tolerance import tolerance_for
from ..context import RunContext
from ._support import adapt

ID = "B7"
OPS = None


def run(ctx: RunContext) -> None:
    generator = torch.Generator().manual_seed(ctx.seed)
    base = torch.randn(16, 32, generator=generator)

    # Exact operators must be bit-identical.  A transcendental one may pick a
    # different vectorisation for a strided buffer -- on CPU as well -- so it is
    # held to the same budget the tolerance table gives that arithmetic class.
    transcendental = "transcendental" in ctx.op.tags
    budget = tolerance_for(torch.float32, "transcendental")
    layouts = {
        "every other column": base[:, ::2],
        "transposed": base.t(),
        "row slice": base[3:11],
    }
    for label, tensor in layouts.items():
        if tensor.numel() == 0:
            continue
        args, kwargs = adapt(ctx.op.name, tensor)
        out = ctx.op.forward(*args, **kwargs)
        contiguous = tensor.contiguous()
        c_args, c_kwargs = adapt(ctx.op.name, contiguous)
        reference = ctx.op.forward(*c_args, **c_kwargs)
        if out.shape != reference.shape:
            raise AssertionError(
                f"{ctx.op.name}: a {label} input gives {tuple(out.shape)}, contiguous gives {tuple(reference.shape)}"
            )
        if transcendental:
            if not torch.allclose(out, reference, rtol=budget.rtol, atol=budget.atol):
                worst = (out - reference).abs().max().item()
                raise AssertionError(
                    f"{ctx.op.name}: a {label} input changed the values by {worst:.3e}, "
                    f"beyond the {budget} budget for this arithmetic class"
                )
        elif not torch.equal(out, reference):
            raise AssertionError(f"{ctx.op.name}: a {label} input changed the values")
