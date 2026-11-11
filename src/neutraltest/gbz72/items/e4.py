"""E4 -- nothing outside the used region may influence the result.

Check: attention is masked and then the key and value rows the mask removes are
overwritten with different numbers; the output must not move, which is only true
if those rows are never read. In the backward direction the causal window is used
the same way: perturbing the last key must leave the query gradient of every
earlier position bit-identical, because no earlier position may attend to it. The
rotation is checked at its negative angle, which must be the inverse rotation, and
its VJP must come back finite and shaped like the input.

Tolerance: exact equality for the mask and causal-window comparisons -- the two
runs differ only in values that must not reach the arithmetic, so any difference
is a real read of excluded memory. 1e-6 for the negative-angle identity, and the
usual finiteness and shape checks for the gradient.
"""

from __future__ import annotations

from typing import Any

import torch

from ...gbz73.inference import kernels, ops
from .._support import clone_args, jacobian, make_inputs

ID = "E4"
COLUMNS = None


def run(ctx: Any) -> None:
    if ctx.family == "flash_attention":
        _attention(ctx)
    elif ctx.family == "rope":
        _rotation(ctx)
    else:
        ctx.na(f"{ctx.column.name} has no out-of-bounds probe in this clause")


def _attention(ctx: Any) -> None:
    generator = torch.Generator().manual_seed(ctx.seed)
    q = torch.randn(2, 2, 8, 8, generator=generator)
    k = torch.randn(2, 2, 8, 8, generator=generator)
    v = torch.randn(2, 2, 8, 8, generator=generator)
    if ctx.backward:
        before = clone_args((q, k, v))
        kwargs = {"is_causal": True}
        first, _ = ctx.gradients_with((q, k, v), kwargs)
        for gradient in first:
            if not bool(torch.isfinite(gradient).all()):
                raise AssertionError(f"{ctx.column.name}: a gradient is not finite")
        perturbed = k.clone()
        perturbed[..., -1, :] = perturbed[..., -1, :] + 1.0
        second, _ = ctx.gradients_with((q, perturbed, v), kwargs)
        if not torch.equal(first[0][:, :, :-1], second[0][:, :, :-1]):
            raise AssertionError(
                f"{ctx.column.name}: an earlier query saw the last key under a causal mask"
            )
        if torch.equal(first[0], second[0]):
            raise AssertionError(f"{ctx.column.name}: the perturbed key changed nothing")
        jacobian(ctx, (q, k, v), kwargs)
        _unchanged(ctx, (q, k, v), before)
        return
    mask = torch.zeros(8, 8)
    mask[:, 5:] = float("-inf")
    base = kernels.masked_attention(q, k, v, mask=mask)
    k2 = k.clone()
    v2 = v.clone()
    k2[..., 5:, :] = 7.0
    v2[..., 5:, :] = -3.0
    if not torch.equal(base, kernels.masked_attention(q, k2, v2, mask=mask)):
        raise AssertionError(f"{ctx.column.name}: masked rows influenced the output")
    if not bool(torch.isfinite(base).all()):
        raise AssertionError(f"{ctx.column.name}: the masked attention is not finite")


def _rotation(ctx: Any) -> None:
    args, kwargs = make_inputs(family="rope", seed=ctx.seed)
    x, position = args
    before = clone_args(args)
    if ctx.backward:
        jacobian(ctx, args, kwargs)
        first, _ = ctx.gradients_with(args, kwargs)
        if first[0].shape != x.shape:
            raise AssertionError(f"{ctx.column.name}: the gradient is not shaped like the input")
        if not bool(torch.isfinite(first[0]).all()):
            raise AssertionError(f"{ctx.column.name}: the gradient is not finite")
    else:
        close = kernels.rope(x, -position)
        want = ops._rope_rotate(x, position, interleaved=True, inverse=True)
        if not torch.allclose(close, want, rtol=1e-6, atol=1e-6):
            raise AssertionError(
                f"{ctx.column.name}: a negative position is not the inverse rotation"
            )
    _unchanged(ctx, args, before)


def _unchanged(ctx: Any, args: Any, before: Any) -> None:
    for current, original in zip(args, before, strict=True):
        if isinstance(current, torch.Tensor) and not torch.equal(current, original):
            raise AssertionError(f"{ctx.column.name}: the call modified its input")
