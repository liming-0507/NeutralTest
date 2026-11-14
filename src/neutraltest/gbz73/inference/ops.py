"""Reference operators for the inference clause, and the Jacobians they declare.

Every column has two references, both computed on the CPU with plain ``torch``:

``forward``
    the operator itself, written so that it runs in float32 (ordinary baseline)
    and float64 (golden baseline) without special-casing;
``jacobian``
    the analytic derivative of that same operator, written by hand. It is not
    ``autograd`` on the reference: a backward test that took its expectation from
    autograd would only be testing autograd, and an implementation could ship a
    wrong backward and still pass. Where ``autograd`` *is* used, it is on a
    deliberately naive (detached) implementation, so the two paths are
    independent.

The Jacobian is ``freezes``/``index``-aware in a way the forward is not: routing
metadata is held fixed, and the slot write returns a cotangent for the newly
written rows only, since that is the only place the forward used ``k``/``v``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

Args = tuple[Any, ...]
Kwargs = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Sample:
    """One deterministic call of a column, on the CPU."""

    args: Args
    kwargs: Kwargs


# --------------------------------------------------------------------------
# add
# --------------------------------------------------------------------------
def add_forward(a: torch.Tensor, b: torch.Tensor, *, alpha: float = 1.0) -> torch.Tensor:
    return a + b * alpha


def add_jacobian(
    a: torch.Tensor, b: torch.Tensor, *, alpha: float = 1.0
) -> Callable[[torch.Tensor], list[torch.Tensor]]:
    """``d(a + alpha*b)`` -- the residual gradient passes straight through.

    A broadcast operand receives the *summed* cotangent, not a copy of the
    output-shaped one: ``b`` of shape ``(N,)`` added to ``a`` of shape ``(M, N)``
    has a gradient of shape ``(N,)``. This is the single most common backward
    defect, so it is modelled here rather than waved through.
    """

    def vjp(grad: torch.Tensor) -> list[torch.Tensor]:
        def shrink(cotangent: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
            leading = cotangent.dim() - like.dim()
            if leading > 0:
                cotangent = cotangent.sum(dim=tuple(range(leading)))
            axes = tuple(i for i, n in enumerate(like.shape) if n == 1 and cotangent.shape[i] != 1)
            if axes:
                cotangent = cotangent.sum(dim=axes, keepdim=True)
            return cotangent

        return [shrink(grad, a), shrink(grad * alpha, b)]

    return vjp


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------
def attention_forward(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool = False
) -> torch.Tensor:
    # One keyword for one thing: the kernel spells it ``is_causal``, so the
    # reference does too. Two names for the same flag is how call sites drift.
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def attention_jacobian(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool = False
) -> Callable[[torch.Tensor], list[torch.Tensor]]:
    """Hand-written ``dQ``/``dK``/``dV`` for softmax attention.

    Written out rather than taken from ``autograd`` so that the test has an
    independent expectation: the two are compared in the conformance suite.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-1, -2)) * scale
    if is_causal:
        length = scores.shape[-1]
        mask = torch.ones(length, length, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(-1)

    def vjp(grad: torch.Tensor) -> list[torch.Tensor]:
        dv = weights.transpose(-1, -2) @ grad
        dweights = grad @ v.transpose(-1, -2)
        # softmax Jacobian: dS = p * (dP - sum(dP * p))
        dscores = weights * (dweights - (dweights * weights).sum(-1, keepdim=True))
        dscores = dscores * scale
        dq = dscores @ k
        dk = dscores.transpose(-1, -2) @ q
        return [dq, dk, dv]

    return vjp


# --------------------------------------------------------------------------
# kv cache
# --------------------------------------------------------------------------
def kv_write_forward(
    keys: torch.Tensor,
    values: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    index: torch.Tensor,
) -> tuple:
    """Store the tokens in the rows ``index`` names, leaving the others alone.

    A token stored into a row that another token already claimed in the same
    call resolves last-writer-wins: two tokens pointing at one slot is a bug in
    the caller, and a scheduler that randomises the order would make the answer
    differ between runs. ``index_put`` defaults to the opposite (it accumulates),
    which turns a repeated slot into a doubled value and reads as a kernel
    mismatch in the store's row, where nothing is wrong with the store.
    """
    return (
        cache_k.index_put((index,), keys, accumulate=False),
        cache_v.index_put((index,), values, accumulate=False),
    )


def kv_write_jacobian(
    keys: torch.Tensor,
    values: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    index: torch.Tensor,
) -> Callable[[tuple], list[torch.Tensor]]:
    """Route each output gradient to the token stored there and to the row it replaced.

    The cache an operator returns is a new tensor here rather than an in-place
    update, so every row keeps a gradient: the written rows get the incoming
    gradient back on their token, and carry the untouched rows through.
    """
    rows = cache_k.shape[0]

    def vjp(grad: tuple) -> list[torch.Tensor]:
        grad_k, grad_v = grad
        written_k = grad_k.index_select(0, index)
        written_v = grad_v.index_select(0, index)
        untouched = torch.ones(rows, dtype=torch.bool)
        untouched[index] = False
        return [
            written_k,
            written_v,
            grad_k * untouched.unsqueeze(-1),
            grad_v * untouched.unsqueeze(-1),
        ]

    return vjp


def kv_read_forward(cache_k: torch.Tensor, cache_v: torch.Tensor, index: torch.Tensor) -> tuple:
    return cache_k[index], cache_v[index]


def kv_read_jacobian(
    cache_k: torch.Tensor, cache_v: torch.Tensor, index: torch.Tensor
) -> Callable[[tuple], list[torch.Tensor]]:
    """Each read row routes its gradient back to the row it came from."""

    def vjp(grad: tuple) -> list[torch.Tensor]:
        grad_k, grad_v = grad
        rows, width = cache_k.shape
        out_k = torch.zeros(rows, width, dtype=grad_k.dtype)
        out_v = torch.zeros(rows, width, dtype=grad_v.dtype)
        return [out_k.index_add(0, index, grad_k), out_v.index_add(0, index, grad_v)]

    return vjp


# --------------------------------------------------------------------------
# softmax
# --------------------------------------------------------------------------
def softmax_forward(x: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)


def softmax_jacobian(
    x: torch.Tensor, *, dim: int = -1
) -> Callable[[torch.Tensor], list[torch.Tensor]]:
    """``dsoftmax`` -- the diagonal form of the Jacobian, applied to a cotangent."""

    def vjp(grad: torch.Tensor) -> list[torch.Tensor]:
        p = torch.softmax(x, dim=dim)
        return [p * (grad - (grad * p).sum(dim, keepdim=True))]

    return vjp


# --------------------------------------------------------------------------
# rope
# --------------------------------------------------------------------------
def _rope_angles(
    position: torch.Tensor, width: int, dtype: torch.dtype, *, heads: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` of the rotation angles, laid out as ``(..., seq, 1, half)``.

    ``position`` is routing rather than a parameter: the angles are
    differentiable in it, and the Jacobian below deliberately returns no
    cotangent for it. The reference detaches for the same reason -- otherwise
    autograd would hand back a third gradient and the arity check would compare
    two different contracts.

    Layout: a rotation table has to broadcast against ``x`` of shape
    ``(batch, seq, heads, width)``. The sequence axis is the one the rotation
    steps over, the width axis is the one it rotates in, and heads are shared --
    so the table is ``(batch, seq, 1, half)`` for a 2-D position and
    ``(1, seq, 1, half)`` for a 1-D one. Doing this here, once, is what keeps the
    reference and the kernel from drifting apart.
    """
    half = width // 2
    inv = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float64) / half)
    angles = position.detach().to(torch.float64)[..., None] * inv
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)
    if cos.dim() == 2:  # (seq, half) -> (1, seq, 1, half)
        cos, sin = cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)
    else:  # (batch, seq, half) -> (batch, seq, 1, half)
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    return cos, sin


def _rope_pair(x: torch.Tensor, *, interleaved: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the rotated axis into the two members of each angle pair.

    ``interleaved`` pairs adjacent lanes ``(2i, 2i+1)``; otherwise the axis is
    split in half and lane ``i`` pairs with lane ``i + half``. Both layouts are
    the *same* rotation -- only which lanes form a pair changes -- so the caller
    never gets to choose how the two members combine.
    """
    if interleaved:
        return x[..., ::2], x[..., 1::2]
    half = x.shape[-1] // 2
    return x[..., :half], x[..., half:]


def _rope_join(a: torch.Tensor, b: torch.Tensor, *, interleaved: bool) -> torch.Tensor:
    """Inverse of :func:`_rope_pair`: interleave or concatenate two half tensors."""
    if interleaved:
        return torch.stack((a, b), dim=-1).flatten(-2)
    return torch.cat((a, b), dim=-1)


def _rope_rotate(
    x: torch.Tensor, position: torch.Tensor, *, interleaved: bool, inverse: bool = False
) -> torch.Tensor:
    """Rotate every angle pair of ``x`` by its angle.

    ``(a, b) -> (a*cos - b*sin, a*sin + b*cos)``, written once, on the pair. The
    norm of each pair is then ``(a^2 + b^2) * (cos^2 + sin^2)``, which the
    rotation has to preserve -- that identity is the acceptance test for this
    operator, and it can only hold if the four coefficients are produced
    together. Building ``cos`` and ``sin`` as two separate full-width tables (and
    hoping they line up) is how the reference drifted into ``sin^2`` before.
    ``inverse`` applies the transpose, which is the same rotation negated.
    """
    cos, sin = _rope_angles(position, x.shape[-1], x.dtype)
    a, b = _rope_pair(x, interleaved=interleaved)
    weight = cos if not inverse else cos
    sign = sin if not inverse else -sin
    return _rope_join(a * weight - b * sign, a * sign + b * weight, interleaved=interleaved)


def rope_forward(
    x: torch.Tensor, position: torch.Tensor, *, interleaved: bool = True
) -> torch.Tensor:
    return _rope_rotate(x, position, interleaved=interleaved)


def rope_jacobian(
    x: torch.Tensor, position: torch.Tensor, *, interleaved: bool = True
) -> Callable[[torch.Tensor], list[torch.Tensor]]:
    """The transpose rotation: ``dx = grad * cos + rot(grad) * sin``."""

    def vjp(grad: torch.Tensor) -> list[torch.Tensor]:
        # The Jacobian of a rotation is the rotation by the negated angle, i.e.
        # the same matrix transposed: cos stays, sin flips sign.
        return [_rope_rotate(grad, position, interleaved=interleaved, inverse=True)]

    return vjp


# --------------------------------------------------------------------------
# rms norm
# --------------------------------------------------------------------------
def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rms_norm_jacobian(
    x: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6
) -> Callable[[torch.Tensor], list[torch.Tensor]]:
    """``dx`` and ``dw`` with the RMS scale held at its forward value."""

    def vjp(grad: torch.Tensor) -> list[torch.Tensor]:
        inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        normed = x * inv_rms
        dweight = (grad * normed).sum(0) if grad.dim() > 1 else grad * normed
        g = grad * weight
        dx = inv_rms * (g - normed * (g * normed).mean(-1, keepdim=True))
        return [dx, dweight]

    return vjp


#: Which function implements each family's forward and Jacobian.
IMPLEMENTATIONS: dict[str, tuple[Callable[..., Any], Callable[..., Any]]] = {
    "add": (add_forward, add_jacobian),
    "flash_attention": (attention_forward, attention_jacobian),
    "kv_cache": (kv_read_forward, kv_read_jacobian),
    "softmax": (softmax_forward, softmax_jacobian),
    "rope": (rope_forward, rope_jacobian),
    "rms_norm": (rms_norm_forward, rms_norm_jacobian),
}


# --------------------------------------------------------------------------
# deterministic samples
# --------------------------------------------------------------------------
#: Small enough to stay on a CPU and large enough that a layout bug shows up.
_SHAPES: dict[str, dict[str, Any]] = {
    "add": {"rows": 8, "width": 16},
    "flash_attention": {"batch": 2, "heads": 4, "seq": 8, "head_dim": 8},
    "kv_cache": {"rows": 16, "width": 8, "tokens": 4},
    "softmax": {"rows": 8, "width": 16},
    "rope": {"batch": 2, "seq": 4, "heads": 2, "head_dim": 8},
    "rms_norm": {"rows": 8, "width": 16},
}


def _randn(shape: tuple[int, ...], dtype: torch.dtype, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator).to(dtype)


def sample(family: str, dtype: torch.dtype, *, seed: int = 0, variant: str = "") -> Sample:
    """One deterministic call of ``family`` in ``dtype``.

    Every column of a family gets the same shapes, so a forward and a backward
    are compared on the same problem rather than on two convenient ones.
    """
    generator = torch.Generator().manual_seed(seed)
    shapes = _SHAPES[family]
    if family == "add":
        a = _randn((shapes["rows"], shapes["width"]), dtype, generator)
        b = _randn((shapes["width"],), dtype, generator)
        return Sample((a, b), {"alpha": 2.0})
    if family == "flash_attention":
        q = _randn(
            (shapes["batch"], shapes["heads"], shapes["seq"], shapes["head_dim"]), dtype, generator
        )
        k = _randn(
            (shapes["batch"], shapes["heads"], shapes["seq"], shapes["head_dim"]), dtype, generator
        )
        v = _randn(
            (shapes["batch"], shapes["heads"], shapes["seq"], shapes["head_dim"]), dtype, generator
        )
        return Sample((q, k, v), {"is_causal": variant == "causal"})
    if family == "kv_cache":
        # k, v, cache_k, cache_v, index -- the full write call. The read column
        # picks the three of these it needs; both columns therefore see the same
        # cache and the same slots.
        tokens = shapes["tokens"]
        keys = _randn((tokens, shapes["width"]), dtype, generator)
        values = _randn((tokens, shapes["width"]), dtype, generator)
        cache_k = _randn((shapes["rows"], shapes["width"]), dtype, generator)
        cache_v = _randn((shapes["rows"], shapes["width"]), dtype, generator)
        index = torch.arange(tokens, dtype=torch.int64) * 3 % shapes["rows"]
        return Sample((keys, values, cache_k, cache_v, index), {})
    if family == "softmax":
        return Sample((_randn((shapes["rows"], shapes["width"]), dtype, generator),), {"dim": -1})
    if family == "rope":
        x = _randn(
            (shapes["batch"], shapes["seq"], shapes["heads"], shapes["head_dim"]), dtype, generator
        )
        # One position per sequence slot, shared by every batch and head: the
        # per-example rotary-table layout the kernel documents.
        sequence = torch.arange(shapes["seq"]).to(torch.float32)
        position = sequence.unsqueeze(0).expand(shapes["batch"], -1).contiguous()
        return Sample((x, position), {"interleaved": variant != "split"})
    if family == "rms_norm":
        x = _randn((shapes["rows"], shapes["width"]), dtype, generator)
        weight = (torch.randn(shapes["width"], generator=generator) * 0.1 + 1.0).to(dtype)
        return Sample((x, weight), {"eps": 1e-6})
    raise KeyError(f"no sample for family {family!r}")


def variants(family: str) -> tuple[str, ...]:
    """Second shapes/layouts worth exercising for a family, if any."""
    if family == "flash_attention":
        return ("", "causal")
    if family == "rope":
        return ("", "split")
    return ("",)
