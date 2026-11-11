"""A10 -- the scale parameter is applied, and only where it belongs.

Check: one parameter per marked family carries the multiplicative scale --
``alpha`` for the residual add, ``softmax_scale`` for attention, ``eps`` and
``weight`` for the normalisation, and the row index for the cache. For each, the
neutral value is compared exactly with the operation it should reduce to
(``alpha=1`` is a plain add, ``softmax_scale=None`` is ``1/sqrt(d)``), a
non-neutral value is compared with the reference at that value, and the two are
required to differ, so a parameter that is accepted and then ignored cannot pass.
The cache has no scalar scale; its contract is that the write lands on the named
rows and leaves every other row bit-identical.

Tolerance: exact where the two sides perform the same arithmetic (``alpha=1``
against ``torch.add``, the cache rows against their source), and the column's
float32 budget where a different but equivalent expression is used (the rotation
with a non-neutral ``eps`` against its reference, the doubled weight against
twice the result).
"""

from __future__ import annotations

from typing import Any

import torch

from .._support import budget, close, jacobian, make_inputs, observe, reference

ID = "A10"
COLUMNS = None


def run(ctx: Any) -> None:
    args, kwargs = make_inputs(
        family=ctx.family, dtype=torch.float32, seed=ctx.seed, variant=ctx.variant
    )
    if ctx.backward:
        jacobian(ctx, args, kwargs)
        altered = _altered(ctx, args, kwargs)
        if altered is not None:
            jacobian(ctx, *altered)
        return
    if ctx.family == "add":
        _add(ctx, args)
    elif ctx.family == "flash_attention":
        _attention(ctx, args, kwargs)
    elif ctx.family == "kv_cache":
        _cache(ctx, args)
    else:
        _rms(ctx, args)


def _add(ctx: Any, args: Any) -> None:
    a, b = args
    neutral = observe(ctx, args, {"alpha": 1.0})[0]
    if not torch.equal(neutral, a + b):
        raise AssertionError(f"{ctx.column.name}: alpha=1 is not a plain add")
    if not torch.equal(observe(ctx, args, {})[0], neutral):
        raise AssertionError(f"{ctx.column.name}: the default alpha is not 1")
    if not torch.equal(observe(ctx, args, {"alpha": 0.0})[0], a):
        raise AssertionError(f"{ctx.column.name}: alpha=0 did not return the first operand")
    doubled = observe(ctx, args, {"alpha": 2.0})[0]
    close(doubled, a + 2.0 * b, rel=1e-6, msg=f"{ctx.column.name}: alpha=2")
    if torch.equal(doubled, neutral):
        raise AssertionError(f"{ctx.column.name}: alpha had no effect")


def _attention(ctx: Any, args: Any, kwargs: Any) -> None:
    q, k, v = args
    width = q.shape[-1]
    default = observe(ctx, args, kwargs)[0]
    explicit = observe(ctx, args, dict(kwargs, softmax_scale=1.0 / width**0.5))[0]
    if not torch.equal(default, explicit):
        raise AssertionError(f"{ctx.column.name}: the default scale is not 1/sqrt(d)")
    scale = 2.0
    scaled = observe(ctx, args, dict(kwargs, softmax_scale=scale))[0]
    # Scaling the logits by ``scale`` is scaling q by ``scale * sqrt(d)`` before
    # the framework's own 1/sqrt(d); the reference has no scale argument, so the
    # equivalent call is built on the operand.
    expected = reference(
        ctx,
        (q * (scale * width**0.5), k, v),
        {"is_causal": kwargs.get("is_causal", False)},
        dtype=torch.float32,
    )[0]
    close(scaled, expected, rel=budget(ctx), msg=f"{ctx.column.name}: softmax_scale=2")
    if torch.equal(scaled, default):
        raise AssertionError(f"{ctx.column.name}: softmax_scale had no effect")


def _cache(ctx: Any, args: Any) -> None:
    keys, values, cache_k, cache_v, index = args
    slot = index.to(torch.long)
    out_k, out_v = observe(ctx, args, {})
    if not torch.equal(out_k[slot], keys) or not torch.equal(out_v[slot], values):
        raise AssertionError(f"{ctx.column.name}: the write did not land on the named rows")
    untouched = torch.ones(cache_k.shape[0], dtype=torch.bool)
    untouched[slot] = False
    if bool(untouched.any()):
        if not torch.equal(out_k[untouched], cache_k[untouched]):
            raise AssertionError(f"{ctx.column.name}: the write changed an unnamed cache row")
        if not torch.equal(out_v[untouched], cache_v[untouched]):
            raise AssertionError(f"{ctx.column.name}: the write changed an unnamed cache row")


def _rms(ctx: Any, args: Any) -> None:
    x, weight = args
    neutral = observe(ctx, args, {"eps": 1e-12})[0]
    close(
        neutral,
        reference(ctx, args, {"eps": 1e-12}, dtype=torch.float32)[0],
        rel=budget(ctx),
        msg=f"{ctx.column.name}: eps=1e-12",
    )
    damped = observe(ctx, args, {"eps": 1e-1})[0]
    close(
        damped,
        reference(ctx, args, {"eps": 1e-1}, dtype=torch.float32)[0],
        rel=budget(ctx),
        msg=f"{ctx.column.name}: eps=1e-1",
    )
    if torch.equal(damped, neutral):
        raise AssertionError(f"{ctx.column.name}: eps had no effect")
    doubled = observe(ctx, (x, weight * 2.0), {"eps": 1e-6})[0]
    close(
        doubled,
        2.0 * observe(ctx, args, {"eps": 1e-6})[0],
        rel=1e-6,
        msg=f"{ctx.column.name}: doubled weight",
    )


def _altered(ctx: Any, args: Any, kwargs: Any) -> tuple[Any, Any] | None:
    """The same backward call with one scale parameter moved off its default.

    The attention's scale lives on the logits, which the reference does not
    accept as a keyword, so the altered call folds it into ``q`` -- the same
    derivative as the kernel's ``softmax_scale`` argument.
    """
    if ctx.family == "add":
        return args, {"alpha": 2.0}
    if ctx.family == "rms_norm":
        return args, {"eps": 1e-1}
    if ctx.family == "flash_attention":
        q, k, v = args
        factor = 2.0 * q.shape[-1] ** 0.5
        return (q * factor, k, v), {"is_causal": kwargs.get("is_causal", False)}
    return None
