"""D1 -- the calling contract: positional and keyword forms agree, defaults are
the documented ones, and an unknown keyword is refused rather than ignored."""

from __future__ import annotations

import torch

from ..context import RunContext
from ._support import adapt

ID = "D1"
OPS = None


def run(ctx: RunContext) -> None:
    name = ctx.op.name
    generator = torch.Generator().manual_seed(ctx.seed)
    x = torch.randn(16, 32, generator=generator)

    args, kwargs = adapt(name, x)
    positional = ctx.op.forward(*args, **kwargs)

    if name == "layer_norm":
        weight, bias = args[1], args[2]
        by_keyword = torch.nn.functional.layer_norm(
            x, (x.shape[-1],), weight=weight, bias=bias, eps=1e-5
        )
        defaulted = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias)
        if not torch.equal(by_keyword, defaulted):
            raise AssertionError("layer_norm: the default eps is not 1e-5")
    elif name == "gelu":
        by_keyword = torch.nn.functional.gelu(x, approximate="none")
    elif name == "amax":
        by_keyword = torch.amax(x.abs(), dim=-1)
    elif name == "relu":
        by_keyword = torch.relu(input=x)
    else:
        by_keyword = ctx.op.forward(args[0], b=args[1], group_list=args[2])

    if not torch.equal(positional, by_keyword):
        raise AssertionError(
            f"{name}: the keyword form gave a different result than the positional form"
        )

    try:
        ctx.op.forward(*args, no_such_argument=True, **kwargs)
    except TypeError:
        pass
    else:
        raise AssertionError(f"{name}: an unknown keyword argument was accepted")
