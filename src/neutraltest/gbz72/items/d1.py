"""D1 -- the call contract.

Check: the kernel's parameter list is compared with the clause's contract -- the
operand names, the order, and the default of every keyword -- and the function is
then exercised three ways: positionally, entirely by keyword, and with one
argument missing or one argument that does not exist. A parameter that was
renamed, reordered, made positional-only, or given a different default breaks the
first part; a call that only works positionally breaks the second; a call that
accepts an unknown keyword silently breaks the third.

Tolerance: no numeric comparison. The names, order and defaults are exact, the
keyword call must produce a bit-identical result to the positional one, and the
two invalid calls must raise ``TypeError`` specifically rather than any error.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch

from .._support import as_tuple, make_inputs, observe

ID = "D1"
COLUMNS = None

CONTRACTS: dict[str, tuple[tuple[str, ...], dict[str, object]]] = {
    "add": (("a", "b", "alpha"), {"alpha": None}),
    "flash_attention": (
        ("q", "k", "v", "is_causal", "softmax_scale"),
        {"is_causal": False, "softmax_scale": None},
    ),
    "softmax": (("x", "dim"), {"dim": -1}),
    "rope": (("x", "position", "interleaved"), {"interleaved": True}),
    "rms_norm": (("x", "weight", "eps"), {"eps": 1e-6}),
    "kv_cache_write": (("k", "v", "cache_k", "cache_v", "index", "quantise"), {"quantise": None}),
    "kv_cache_read": (("cache_k", "cache_v", "index", "quantise"), {"quantise": None}),
}


def run(ctx: Any) -> None:
    if ctx.family != "kv_cache":
        family = ctx.family
    else:
        family = "kv_cache_read" if ctx.backward else "kv_cache_write"
    expected, defaults = CONTRACTS[family]
    kernel = ctx.kernel()
    signature = inspect.signature(kernel)
    names = tuple(signature.parameters)
    if names != expected:
        raise AssertionError(
            f"{ctx.column.name}: the parameters are {names}, the clause says {expected}"
        )
    for name, default in defaults.items():
        actual = signature.parameters[name].default
        if actual != default and not (actual is None and default is None):
            raise AssertionError(
                f"{ctx.column.name}: the default of {name} is {actual!r}, not {default!r}"
            )
    args, kwargs = make_inputs(family=ctx.family, seed=ctx.seed, variant=ctx.variant)
    positional = observe(ctx, args, kwargs)
    called_args, called_kwargs = ctx.project(args, kwargs)
    mapping = dict(zip(expected[: len(called_args)], called_args, strict=True), **called_kwargs)
    by_keyword = as_tuple(kernel(**mapping))
    _same(ctx, positional, by_keyword, "the keyword call")
    for produced in positional:
        if not bool(torch.isfinite(produced).all()):
            raise AssertionError(f"{ctx.column.name}: a finite input produced a non-finite value")
    _expect_type_error(ctx, kernel, called_args[:-1], called_kwargs, "one argument short")
    _expect_type_error(
        ctx,
        kernel,
        called_args,
        {**called_kwargs, "there_is_no_such_argument": 0},
        "an unknown keyword",
    )


def _same(ctx: Any, left: Any, right: Any, what: str) -> None:
    if len(left) != len(right) or any(
        not torch.equal(a, b) for a, b in zip(left, right, strict=True)
    ):
        raise AssertionError(f"{ctx.column.name}: {what} differs from the positional call")


def _expect_type_error(ctx: Any, kernel: Any, args: Any, kwargs: dict[str, Any], what: str) -> None:
    try:
        kernel(*args, **kwargs)
    except TypeError:
        return
    raise AssertionError(f"{ctx.column.name}: {what} did not raise TypeError")
